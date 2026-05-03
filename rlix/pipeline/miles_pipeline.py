"""MilesPipeline — RLix-controlled MILES fullasync GRPO pipeline actor.

Parallel to RollFullFinetunePipeline but driving MILES's RolloutManager
(SGLang + Megatron) instead of ROLL's vLLM/SGLang workers.

Naming convention for Ray actors:
  coordinator: "rlix:coordinator:{pipeline_id}"  in namespace get_pipeline_namespace(pipeline_id)
  pipeline:    "rlix:pipeline:{pipeline_id}"     in namespace get_pipeline_namespace(pipeline_id)
  model_update_service: "rlix:miles_model_update_service:{pipeline_id}" (same namespace)

Milestone M11.1: cpu_serialize colocate transport only (vast.ai restricted container).
cuda_ipc colocate adapter is M11.2. Cross-node TP is M11.3.
"""
from __future__ import annotations

import logging
import os
import shutil
import threading
from typing import Any, Dict, List, Optional, Set

import ray

from rlix.protocol.types import (
    ACTOR_TRAIN_CLUSTER_NAME,
    COORDINATOR_ACTOR_NAME_PREFIX,
    GENERATION_CLUSTER_NAME,
    RLIX_NAMESPACE,
    SCHEDULER_ACTOR_NAME,
    ActionResponse,
    Priority,
    get_pipeline_namespace,
)
from rlix.utils.env import parse_env_timeout_s
from rlix.utils.ray import get_actor_or_raise

logger = logging.getLogger(__name__)

# DO_TIME_SHARING: True when MILES is running under RLix scheduler control.
# Drives build_cpu_bucket_cache / hooks / placement adapter paths.
DO_TIME_SHARING: bool = os.environ.get("RLIX_CONTROL_PLANE") == "rlix"


def _validate_rlix_topology(args: Any) -> None:
    """F10 — fail-fast topology validation for RLix MILES partial-overlap mode.

    All checks must pass before any actor allocation. Clear error messages
    so mis-configured launches fail immediately with actionable diagnostics.
    """
    train_devices: Set[int] = set(range(int(args.actor_num_nodes) * int(args.actor_num_gpus_per_node)))
    infer_devices: Set[int] = set(range(int(args.rollout_num_gpus)))
    infer_engine_count: int = int(args.rollout_num_gpus) // int(args.rollout_num_gpus_per_engine)

    # Partial overlap: train GPUs must be a proper strict subset of infer GPUs.
    assert train_devices != infer_devices, (
        f"F10: train_devices == infer_devices — partial overlap requires train ⊊ infer "
        f"(at least one non-overlap inference engine). "
        f"train={sorted(train_devices)} infer={sorted(infer_devices)}"
    )
    assert train_devices.issubset(infer_devices), (
        f"F10: partial overlap requires train_devices ⊂ infer_devices. "
        f"train={sorted(train_devices)} infer={sorted(infer_devices)}"
    )

    # At least 2 engines: need ≥1 overlap + ≥1 non-overlap engine.
    assert infer_engine_count >= 2, (
        f"F10: partial overlap requires >= 2 inference engines "
        f"(got {infer_engine_count} from rollout_num_gpus={args.rollout_num_gpus} / "
        f"rollout_num_gpus_per_engine={args.rollout_num_gpus_per_engine})"
    )

    # At least 1 full engine stays active after shrink (non-overlap guarantee).
    non_overlap = infer_devices - train_devices
    assert len(non_overlap) >= int(args.rollout_num_gpus_per_engine), (
        f"F10: at least 1 full inference engine must stay active after shrink. "
        f"non_overlap_gpus={sorted(non_overlap)} rollout_num_gpus_per_engine={args.rollout_num_gpus_per_engine}"
    )

    # Fullasync required (drives turn-level redispatch + weight staleness).
    rollout_fn = getattr(args, "rollout_function_path", "") or ""
    assert rollout_fn.endswith("fully_async_rollout.generate_rollout_fully_async"), (
        f"F10: RLix MILES partial overlap requires fullasync rollout function. "
        f"rollout_function_path must end with 'fully_async_rollout.generate_rollout_fully_async', "
        f"got {rollout_fn!r}"
    )

    # offload_train required: actor.sleep() has internal assert args.offload_train.
    assert getattr(args, "offload_train", False), (
        "F10: RLix mode requires offload_train=True. Without it, actor_train cannot "
        "release overlap GPU after each step → OOM at infer wake_up."
    )

    # M11.1: cpu_serialize only (vast.ai restricted container; no CAP_SYS_PTRACE / --ipc=host).
    transport = getattr(args, "model_update_transport", "cuda_ipc")
    assert transport == "cpu_serialize", (
        f"F10 M11.1: only cpu_serialize colocate transport supported on vast.ai restricted container "
        f"(got model_update_transport={transport!r}). cuda_ipc = M11.2 production cluster milestone."
    )

    # sglang_data_parallel_size must be 1 (RLix handles DP externally via engine scheduling).
    assert int(getattr(args, "sglang_data_parallel_size", 1)) == 1, (
        "F10: RLix mode requires sglang_data_parallel_size == 1 "
        "(RLix scheduler handles DP via engine index, not SGLang-internal DP)"
    )

    # No PD disaggregation.
    try:
        from miles.backends.sglang_utils.sglang_config import SglangConfig
        sglang_config = SglangConfig(args)
        assert not getattr(sglang_config, "has_pd_disaggregation", False), (
            "F10: PD disaggregation is out of scope for M11.1 RLix MILES"
        )
    except (ImportError, AttributeError):
        pass  # SglangConfig not importable or attribute absent; skip (validated at runtime)

    # No MoE / EP (F4 CPU bucket cache only covers dense Megatron).
    assert int(getattr(args, "expert_model_parallel_size", 1)) == 1, (
        "F10: MoE/EP is out of scope for M11.1 RLix MILES. "
        "F4 CPU bucket cache only covers dense Megatron parameters."
    )
    assert int(getattr(args, "moe_router_topk", 0)) == 0, (
        "F10: MoE configs not allowed in RLix mode (M11.1 scope = dense Megatron only)"
    )

    # async_save races with actor.sleep() torch_memory_saver.pause() → segfault.
    assert not getattr(args, "async_save", False), (
        "F10: async_save not supported in M11.1 — background ckpt flush races "
        "actor.sleep() torch_memory_saver.pause(). "
        "Fix: add maybe_finalize_async_save(blocking=True) + cuda.synchronize() "
        "in MegatronTrainRayActor.sleep() prologue (M11 follow-up)."
    )

    # No RadixTreeMiddleware (partial_rollout + radix_tree is a follow-up).
    # Use substring match to catch both short class names and fully-qualified paths.
    middleware_paths = getattr(args, "miles_router_middleware_paths", None) or []
    assert not any("RadixTreeMiddleware" in m for m in middleware_paths), (
        "F10: RLix mode disables RadixTreeMiddleware. "
        "partial_rollout + radix_tree is a follow-up after turn-level redispatch is stable."
    )

    # Single updateable model + server (no critic/reward refit in M11.1).
    assert getattr(args, "critic_model_path", None) is None, (
        "F10: RLix M11.1 requires single updateable model (critic_model_path not supported)"
    )
    assert getattr(args, "reward_model_path", None) is None, (
        "F10: RLix M11.1 requires single updateable model (reward_model_path not supported)"
    )
    assert int(getattr(args, "sglang_secondary_server_count", 0)) == 0, (
        "F10: RLix M11.1 requires single SGLang server group (secondary_server_count must be 0)"
    )

    # Megatron parallelism divisibility.
    tp = int(getattr(args, "tensor_model_parallel_size", 1))
    pp = int(getattr(args, "pipeline_model_parallel_size", 1))
    cp = int(getattr(args, "context_parallel_size", 1))
    ep = int(getattr(args, "expert_model_parallel_size", 1))
    megatron_product = tp * pp * cp * ep
    n_train = len(train_devices)
    assert n_train % megatron_product == 0, (
        f"F10: train device_mapping size ({n_train}) must divide evenly by "
        f"tp*pp*cp*ep ({megatron_product})"
    )
    n_infer = len(infer_devices)
    gpus_per_engine = int(args.rollout_num_gpus_per_engine)
    assert n_infer % gpus_per_engine == 0, (
        f"F10: infer device_mapping size ({n_infer}) must divide evenly by "
        f"rollout_num_gpus_per_engine ({gpus_per_engine})"
    )

    # M11.1: cross-node TP not supported (M11.3). Multi-node DP is OK.
    num_gpus_per_node = int(getattr(args, "num_gpus_per_node", getattr(args, "actor_num_gpus_per_node", 8)))
    assert gpus_per_engine <= num_gpus_per_node, (
        f"F10 M11.1: cross-node TP not supported (engine spans {gpus_per_engine} GPUs "
        f"but node has {num_gpus_per_node}). Cross-node TP = M11.3 milestone."
    )

    # First-build: sorted contiguous infer_device_mapping only.
    infer_device_mapping = list(range(n_infer))
    assert infer_device_mapping == sorted(infer_device_mapping), (
        "F10: first build requires sorted contiguous infer_device_mapping. "
        "Non-contiguous mapping needs explicit scheduler_dp_rank→engine_index adapter (F12 follow-up)."
    )
    for engine_idx, start in enumerate(range(0, n_infer, gpus_per_engine)):
        group = infer_device_mapping[start : start + gpus_per_engine]
        expected = list(range(group[0], group[0] + gpus_per_engine))
        assert group == expected, (
            f"F10: infer engine {engine_idx} must occupy contiguous GPUs in first build; "
            f"got {group}, expected {expected}"
        )

    # Streaming generate must be disabled (router metadata injection requires JSON body).
    assert not getattr(args, "rollout_force_stream", False), (
        "F10: RLix mode requires non-streaming generate; "
        "metadata injection requires JSON body (SSE has no meta_info field)"
    )

    # cpu_serialize: /dev/shm must be writable and large enough.
    bucket_mb = int(getattr(args, "miles_model_update_bucket_size_mb", 512))
    bucket_bytes = bucket_mb * 1024 * 1024
    assert os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK), (
        "F10: cpu_serialize transport requires writable /dev/shm. "
        "Container may need --shm-size (Docker) or tmpfs mount."
    )
    shm_free = shutil.disk_usage("/dev/shm").free
    required_bytes = bucket_bytes + 256 * 1024 * 1024  # bucket + 256 MB safety margin
    assert shm_free >= required_bytes, (
        f"F10: /dev/shm free space ({shm_free} bytes) < required ({required_bytes} bytes "
        f"= bucket_size + 256 MB margin). Increase --shm-size (Docker) or reduce "
        f"args.miles_model_update_bucket_size_mb (currently {bucket_mb} MB)."
    )

    logger.info(
        "[MilesPipeline] F10 topology validation passed: "
        "train_devices=%s infer_devices=%s engine_count=%d tp=%d pp=%d cp=%d",
        sorted(train_devices), sorted(infer_devices), infer_engine_count, tp, pp, cp,
    )


class MilesPipeline:
    """RLix-managed MILES fullasync GRPO pipeline actor.

    Lifecycle (M11.1):
      initialize_pipeline()  — allocate clusters, build base CPU cache, bootstrap engines
      run()                  — main training loop (fullasync GRPO)
      resize_infer()         — called by coordinator when scheduler shrinks/expands engines
      shrink_engines()       — delegate to RolloutManager (Phase B)
      expand_engines()       — delegate to RolloutManager (Phase B)
    """

    def __init__(self, *, pipeline_id: str, pipeline_config: Any):
        if not isinstance(pipeline_id, str) or not pipeline_id:
            raise ValueError("pipeline_id must be non-empty str")
        self._pipeline_id: str = pipeline_id
        self._pipeline_config: Any = pipeline_config
        self._initialized: bool = False
        self._init_lock = threading.Lock()

        self._rlix_scheduler = get_actor_or_raise(
            SCHEDULER_ACTOR_NAME,
            RLIX_NAMESPACE,
            error_context="MilesPipeline requires the central scheduler actor at startup.",
        )
        self._actor_train_cluster_id: str = f"{pipeline_id}_{ACTOR_TRAIN_CLUSTER_NAME}"
        self._actor_infer_cluster_id: str = f"{pipeline_id}_{GENERATION_CLUSTER_NAME}"

        # Lazily resolved coordinator handle.
        self._coordinator_handle: Any = None

        # State from Phase C initialization.
        self._cache_ready_step: Optional[int] = None
        self._current_weight_version: Optional[int] = None
        self._cache_owner_actor: Any = None  # Megatron worker that owns the CPU bucket cache
        self.actor_train: Any = None          # RayTrainGroup
        self.actor_infer: Any = None          # RolloutManager

    def _get_coordinator_handle(self) -> Any:
        if self._coordinator_handle is not None:
            return self._coordinator_handle
        namespace = get_pipeline_namespace(self._pipeline_id)
        actor_name = f"{COORDINATOR_ACTOR_NAME_PREFIX}{self._pipeline_id}"
        self._coordinator_handle = get_actor_or_raise(
            actor_name, namespace,
            error_context=f"Coordinator required for pipeline_id={self._pipeline_id!r}.",
        )
        return self._coordinator_handle

    def _request_cluster_gpus(
        self, *, cluster_id: str, priority: Any, global_step: int
    ) -> List[int]:
        allocated = ray.get(
            self._rlix_scheduler.request_gpus.remote(
                cluster_id=str(cluster_id),
                priority=priority,
                global_step=global_step,
            )
        )
        if not isinstance(allocated, list):
            raise RuntimeError(f"rlix:scheduler.request_gpus returned non-list: {type(allocated).__name__}")
        allocated = [int(x) for x in allocated]
        if not allocated:
            raise RuntimeError(f"rlix:scheduler allocated empty GPU list for cluster_id={cluster_id!r}")
        return allocated

    def _notify_release_cluster_gpus(self, *, cluster_id: str, global_step: int) -> None:
        ray.get(
            self._rlix_scheduler.notify_release_gpus.remote(
                cluster_id=str(cluster_id), global_step=global_step
            )
        )

    def initialize_pipeline(self) -> ActionResponse:
        """Initialize MILES pipeline under RLix scheduler control.

        Steps follow spec §F4 init bootstrap order exactly:
          Phase 1 (train):
            Step 1:   request actor_train GPUs (INITIALIZATION priority)
            Step 1b:  create RayTrainGroup with worker_placements (Phase E) or pg (standalone)
            Step 2:   run(actor_train.init())
            Step 3:   run(actor_train.onload())        [wake needed for cache build]
            Step 4:   run(actor_train.build_cpu_bucket_cache(step=-1))
            Step 5:   run(actor_train.offload())       [release overlap GPUs + destroy NCCL]
            finally:  M4 hard cleanup on failure + release train scheduler
          Phase 2 (infer + service bootstrap):
            Step 6.5: collect cache_owner_actor handle
            Step 7:   request actor_infer GPUs (GENERATION priority) + M1 full-alloc assert
            Step 8:   create RolloutManager with worker_placements
            Step 9:   get_engine_count() sanity check
            Step 10:  bootstrap_active_engines + register_model_update_resources

        Phase A stub: F10 validation runs fully. Steps 1-10 are implemented as stubs
        that will be filled in Phase C (cpu_bucket_cache) and Phase E (placement adapter).
        """
        with self._init_lock:
            if self._initialized:
                return ActionResponse(success=True)

            args = self._pipeline_config
            init_global_step = -1

            # F10: fail-fast topology validation (runs immediately, before any GPU allocation).
            _validate_rlix_topology(args)

            # ----------------------------------------------------------------
            # Phase 1: actor_train allocation + base cache build
            # ----------------------------------------------------------------
            self._request_cluster_gpus(
                cluster_id=self._actor_train_cluster_id,
                priority=Priority.INITIALIZATION,
                global_step=init_global_step,
            )

            train_init_succeeded = False
            try:
                from miles.ray.actor_group import RayTrainGroup
                from miles.utils.async_utils import run
                from miles.ray.placement_group import create_placement_groups

                # Step 1b: Create RayTrainGroup.
                # Phase E: RLix mode uses MilesPlacementProvider.get_train_workers() instead of create_placement_groups.
                # Phase C (first build): use standalone placement_groups as placeholder.
                pgs = create_placement_groups(args)
                self.actor_train = RayTrainGroup(
                    args=args,
                    num_nodes=int(getattr(args, "actor_num_nodes", 1)),
                    num_gpus_per_node=int(getattr(args, "actor_num_gpus_per_node", 1)),
                    pg=pgs["actor"],
                    num_gpus_per_actor=0.01,  # RLix mode: tiny reservation, real isolation via PG + CVD
                    role="actor",
                    with_ref=(getattr(args, "kl_coef", 0) != 0 or getattr(args, "use_kl_loss", False)),
                )

                # Step 2: init (loads model to GPU).
                run(self.actor_train.init())

                # Step 3: onload (wake if init auto-slept; needed for cache build).
                if getattr(args, "offload_train", False):
                    run(self.actor_train.onload())

                # Step 4: build base CPU bucket cache (GPU has weights; all ranks gather).
                run(self.actor_train.build_cpu_bucket_cache(step=-1))
                self._cache_ready_step = -1

                # Step 5: offload training GPU + destroy NCCL groups (partial overlap: release overlap GPUs).
                run(self.actor_train.offload())

                train_init_succeeded = True

            except Exception:
                # M4: on failure, hard-kill train actors before finally release.
                if self.actor_train is not None:
                    for h in getattr(self.actor_train, "_actor_handles", []):
                        try:
                            ray.kill(h, no_restart=True)
                        except Exception:
                            pass
                    self.actor_train = None
                raise
            finally:
                if not train_init_succeeded and self.actor_train is not None:
                    for h in getattr(self.actor_train, "_actor_handles", []):
                        try:
                            ray.kill(h, no_restart=True)
                        except Exception:
                            pass
                    self.actor_train = None
                import sys as _sys
                _active_exc = _sys.exc_info()[0] is not None
                if _active_exc:
                    # Failure path: swallow release error to preserve the original exception.
                    try:
                        self._notify_release_cluster_gpus(
                            cluster_id=self._actor_train_cluster_id,
                            global_step=init_global_step,
                        )
                    except Exception:
                        logger.exception(
                            "[MilesPipeline] scheduler train release failed; original init error takes precedence"
                        )
                else:
                    # Success path: let release failures propagate (scheduler state must be consistent).
                    self._notify_release_cluster_gpus(
                        cluster_id=self._actor_train_cluster_id,
                        global_step=init_global_step,
                    )

            # ----------------------------------------------------------------
            # Phase 2: actor_infer allocation + service bootstrap
            # ----------------------------------------------------------------
            actor_infer_allocated = False
            try:
                from miles.ray.placement_group import create_placement_groups, create_rollout_manager
                from miles.utils.async_utils import run

                # Step 6.5: collect cache owner actor handle (must run before infer allocation).
                roles = run(self.actor_train.collect_cache_owner_roles())
                # roles: list of (global_rank, is_owner, actor_handle)
                cache_owner_handles = [(r, h) for r, is_owner, h in roles if is_owner]
                assert len(cache_owner_handles) == 1, (
                    f"Expected exactly 1 cache owner (pp0+dp0+tp0+cp0); got {len(cache_owner_handles)}"
                )
                _, self._cache_owner_actor = cache_owner_handles[0]

                # Step 7: request actor_infer GPUs (GENERATION priority; long-lived allocation).
                allocated_infer_gpus = self._request_cluster_gpus(
                    cluster_id=self._actor_infer_cluster_id,
                    priority=Priority.GENERATION,
                    global_step=init_global_step,
                )
                actor_infer_allocated = True

                n_infer = int(args.rollout_num_gpus)
                gpus_per_engine = int(args.rollout_num_gpus_per_engine)
                engine_count = n_infer // gpus_per_engine
                # M1: assert full GENERATION allocation (partial allocation bootstrap = follow-up).
                expected_infer_gpus = set(range(n_infer))
                assert set(allocated_infer_gpus) == expected_infer_gpus, (
                    f"M1: first build requires full GENERATION allocation; "
                    f"got {sorted(allocated_infer_gpus)} vs declared {sorted(expected_infer_gpus)}."
                )

                # Step 8: create RolloutManager (Phase E: use MilesPlacementProvider; Phase C: standalone).
                from miles.ray.rollout import RolloutManager as _RolloutManager
                pgs = create_placement_groups(args)
                self.actor_infer = _RolloutManager.options(
                    namespace=get_pipeline_namespace(self._pipeline_id),
                    name=f"rlix:rollout_manager:{self._pipeline_id}",
                ).remote(args, pgs["rollout"])

                # Step 9: engine_count sanity check.
                actual_count = ray.get(self.actor_infer.get_engine_count.remote())
                assert actual_count == engine_count, (
                    f"RolloutManager engine_count mismatch: got {actual_count}, expected {engine_count}"
                )

                # Initialize engine lifecycle map for F2 sleep/wake.
                ray.get(self.actor_infer.initialize_rlix_engine_map.remote())

                # Step 10: bootstrap coordinator active set + register model update resources.
                coordinator = self._get_coordinator_handle()
                ray.get(coordinator.bootstrap_active_engines.remote(set(range(engine_count))))
                ray.get(coordinator.register_model_update_resources.remote(
                    cache_owner_actor=self._cache_owner_actor,
                    rollout_manager=self.actor_infer,
                ))

                logger.info(
                    "[MilesPipeline] initialize_pipeline complete. "
                    "engine_count=%d cache_ready_step=%d pipeline_id=%s",
                    engine_count, self._cache_ready_step, self._pipeline_id,
                )

            except Exception:
                if self.actor_train is not None:
                    for h in getattr(self.actor_train, "_actor_handles", []):
                        try:
                            ray.kill(h, no_restart=True)
                        except Exception:
                            pass
                    self.actor_train = None
                if self.actor_infer is not None:
                    try:
                        ray.get(self.actor_infer.shutdown_hard.remote(), timeout=10)
                    except Exception:
                        logger.exception("[MilesPipeline] shutdown_hard failed; falling back to ray.kill")
                    try:
                        ray.kill(self.actor_infer, no_restart=True)
                    except Exception:
                        pass
                    self.actor_infer = None
                if actor_infer_allocated:
                    try:
                        self._notify_release_cluster_gpus(
                            cluster_id=self._actor_infer_cluster_id,
                            global_step=init_global_step,
                        )
                    except Exception:
                        logger.exception("[MilesPipeline] scheduler infer release failed")
                raise

            self._initialized = True
            return ActionResponse(success=True)

    def run(self) -> None:
        """Main training loop — implemented in Phase C."""
        raise NotImplementedError(
            "MilesPipeline.run() is implemented in Phase C (weight sync integration). "
            f"pipeline_id={self._pipeline_id!r}"
        )

    def resize_infer(self, dp_ranks_to_remove: List[int], dp_ranks_to_add: List[int]) -> ActionResponse:
        """Resize dispatch from coordinator — delegate to shrink_engines/expand_engines (Phase B)."""
        from rlix.pipeline.utils import validate_resize_params
        validate_resize_params(dp_ranks_to_remove, dp_ranks_to_add)
        if dp_ranks_to_remove:
            self.shrink_engines(dp_ranks_to_remove)
        else:
            self.expand_engines(dp_ranks_to_add)
        return ActionResponse(success=True)

    def shrink_engines(self, engine_indices: List[int]) -> None:
        """Sleep overlap engines (Phase B — requires RolloutManager)."""
        if self.actor_infer is None:
            raise RuntimeError(
                f"shrink_engines called before actor_infer initialized. "
                f"pipeline_id={self._pipeline_id!r}"
            )
        ray.get(self.actor_infer.shrink_engines.remote(engine_indices))

    def expand_engines(self, engine_indices: List[int]) -> None:
        """Wake overlap engines + sync weights + finalize + version publish + activate routing (F5+6)."""
        self._expand_workers(engine_indices)

    def _after_training(self, step: int) -> None:
        """Post-training hook (F5+6):
          1. build_cpu_bucket_cache(step) — all ranks gather, cache owner stores
          2. _cache_ready_step = step
          3. offload training GPU + destroy NCCL groups
          4. sync_base_weights_to_active → active refresh for non-overlap engines
          5. finalize_weight_update on synced engines
          6. version publish + notify_release_cluster_gpus(actor_train)

        Ordering invariant: notify_release_cluster_gpus MUST be after sync+finalize
        to ensure active engines have consistent weights before GPU is released.
        """
        from miles.utils.async_utils import run

        # Step 1-2: build cache (all ranks participate in HF gather).
        run(self.actor_train.build_cpu_bucket_cache(step))
        self._cache_ready_step = step

        # Step 3: offload overlap GPUs + destroy Megatron NCCL groups.
        run(self.actor_train.offload())

        # Step 4: active refresh — push to non-overlap engines that are still serving.
        coordinator = self._get_coordinator_handle()
        synced_engines: List[int] = ray.get(coordinator.sync_base_weights_to_active.remote())

        # Step 5: finalize_weight_update on synced engines (pipeline-owned, per spec §F5+6).
        if synced_engines:
            finalize_refs = [
                self.actor_infer.finalize_engine.remote(idx)
                for idx in synced_engines
            ]
            ray.get(finalize_refs)

        # Step 6: version publish.
        self._current_weight_version = self._cache_ready_step
        if self.actor_infer is not None:
            ray.get(self.actor_infer.set_weight_version.remote(self._current_weight_version))

        # Release actor_train GPUs to scheduler (must be AFTER sync+finalize).
        self._notify_release_cluster_gpus(
            cluster_id=self._actor_train_cluster_id,
            global_step=step,
        )
        logger.info(
            "[MilesPipeline] _after_training complete step=%d version=%d pipeline_id=%s",
            step, self._current_weight_version, self._pipeline_id,
        )

    def _expand_workers(self, engine_indices: List[int]) -> None:
        """Expand path (F5+6): wake engines → sync weights → finalize → version publish → activate routing.

        Version validation: confirms the cache version matches `_cache_ready_step` before
        syncing — prevents a newly woken engine from receiving a stale or inconsistent cache.

        Called by resize_infer when coordinator receives dp_ranks_to_add from scheduler.
        Version publish must happen BEFORE routing activates (spec §F5+6 ordering).
        """
        if self.actor_infer is None:
            raise RuntimeError(f"_expand_workers called before actor_infer initialized. pipeline_id={self._pipeline_id!r}")

        # Version validation: confirm cache is consistent before syncing.
        if self._cache_ready_step is None:
            raise RuntimeError(
                f"_expand_workers called before cache is built (cache_ready_step is None). "
                f"pipeline_id={self._pipeline_id!r}"
            )
        actual_cache_step = ray.get(self._cache_owner_actor.get_cache_ready_step.remote())
        if actual_cache_step != self._cache_ready_step:
            raise RuntimeError(
                f"Cache version mismatch: pipeline expects _cache_ready_step={self._cache_ready_step} "
                f"but cache owner reports {actual_cache_step}. "
                f"Sync state inconsistency. pipeline_id={self._pipeline_id!r}"
            )

        # F2: wake engines (resume_memory_occupation + enable router admission).
        ray.get(self.actor_infer.wake_partial.remote(engine_indices))

        # Sync weights from CPU cache to newly-woken engines.
        coordinator = self._get_coordinator_handle()
        import uuid as _uuid
        sync_id = f"{self._pipeline_id}_expand_{_uuid.uuid4().hex[:8]}"
        # MilesModelUpdateService.sync_selected_workers handles transport routing.
        svc_name = f"rlix:miles_model_update_service:{self._pipeline_id}"
        namespace = get_pipeline_namespace(self._pipeline_id)
        svc = ray.get_actor(svc_name, namespace=namespace)
        ray.get(svc.sync_selected_workers.remote(
            sync_id=sync_id,
            tgt_engine_indices=engine_indices,
        ))

        # Finalize weight update on newly-synced engines.
        finalize_refs = [self.actor_infer.finalize_engine.remote(idx) for idx in engine_indices]
        ray.get(finalize_refs)

        # Version publish BEFORE routing activates (spec ordering invariant).
        if self._current_weight_version is not None:
            ray.get(self.actor_infer.set_weight_version.remote(
                self._current_weight_version, engine_indices
            ))

        logger.info(
            "[MilesPipeline] _expand_workers complete engines=%s version=%s pipeline_id=%s",
            engine_indices, self._current_weight_version, self._pipeline_id,
        )

    def _finalize_weight_update(self, engine_indices: List[int]) -> None:
        """Call finalize_weight_update on each target engine."""
        if self.actor_infer is None:
            return
        refs = [
            self.actor_infer.finalize_engine.remote(idx)
            for idx in engine_indices
        ]
        ray.get(refs)
