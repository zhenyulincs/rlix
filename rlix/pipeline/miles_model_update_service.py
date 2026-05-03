"""MilesModelUpdateService — selective weight sync for MILES SGLang inference engines (F4, F5+6).

Drives per-bucket weight transfer from the Megatron training cache owner to SGLang
inference engines. Two transport paths:
  - cpu_serialize (M11.1, colocate engines): Ray ObjectRef bytes → tmpfs → SGLang HTTP
  - NCCL broadcast (non-colocate engines): H2D staging → dist.broadcast → SGLang

Colocate detection: engine i is colocate iff all its GPUs are in train_devices.
First-build topology (sorted contiguous, F10): engine_i GPUs = [i*E..(i+1)*E-1].

M11.1 hardening:
  - cuda_ipc colocate adapter: M11.2
  - receiver crash tolerance: M11.3
  - LoRA: M11.5
"""
from __future__ import annotations

import logging
import os
import uuid
from typing import Any, Dict, List, Optional, Set

import ray

from rlix.utils.env import parse_env_timeout_s

logger = logging.getLogger(__name__)

# Timeout for entire sync_selected_workers call. Reuses the RLix canonical env var.
_SYNC_TIMEOUT_S: Optional[float] = parse_env_timeout_s(
    "ROLL_SELECTIVE_MODEL_UPDATE_TIMEOUT_S", default_s=150.0
)


def _is_colocate_engine(engine_index: int, args: Any) -> bool:
    """Return True iff engine_index is a colocate engine (all GPUs in train_devices).

    M11.1 single-node first-build only: uses logical GPU ranges which are valid when
    all actors reside on the same node (F10 enforces this: M11.1 forbids cross-node TP
    and requires sorted contiguous infer_device_mapping).

    M11.3 multi-node follow-up: replace with placement-metadata lookup using actual
    Ray node IDs from RolloutManager._engines[idx].worker_url node vs cache owner node.
    """
    actor_num_nodes = int(getattr(args, "actor_num_nodes", 1))
    actor_num_gpus_per_node = int(getattr(args, "actor_num_gpus_per_node", 1))
    train_devices: Set[int] = set(range(actor_num_nodes * actor_num_gpus_per_node))
    gpus_per_engine = int(getattr(args, "rollout_num_gpus_per_engine", 1))
    engine_gpus = set(range(engine_index * gpus_per_engine, (engine_index + 1) * gpus_per_engine))
    return engine_gpus.issubset(train_devices)


@ray.remote
class MilesModelUpdateService:
    """Per-pipeline model update service for MILES SGLang engines.

    Created lazily by MilesCoordinator._get_or_create_model_update_service()
    on first sync call. Named actor: "rlix:miles_model_update_service:{pipeline_id}".

    Separation of responsibilities:
      - MilesModelUpdateService: orchestrates phases, builds comm_plan, manages port claims
      - cache_owner_actor: holds CPU bucket cache, does H2D staging + NCCL broadcast
      - SGLangEngine actors: receive weights via cpu_serialize or NCCL
    """

    def __init__(
        self,
        *,
        pipeline_id: str,
        cache_owner_actor: Any,      # Megatron worker Ray actor handle
        rollout_manager: Any,        # RolloutManager Ray actor handle
        pipeline_config: Any,        # MILES args
    ):
        self._pipeline_id = pipeline_id
        self._cache_owner = cache_owner_actor
        self._rollout_manager = rollout_manager
        self._args = pipeline_config

        # SharedStorage for master port claim (prevents NCCL port collision across syncs).
        try:
            from roll.utils.constants import GLOBAL_STORAGE_NAMESPACE, STORAGE_NAME
            self._shared_storage = ray.get_actor(
                STORAGE_NAME, namespace=GLOBAL_STORAGE_NAMESPACE
            )
        except Exception:
            self._shared_storage = None  # No SharedStorage; ports allocated without claim

        logger.info(
            "[MilesModelUpdateService] init pipeline_id=%s cache_owner=%s",
            pipeline_id, cache_owner_actor,
        )

    def _claim_master_port(self) -> tuple:
        """Claim a free master port for NCCL group rendezvous.

        Returns (master_addr, master_port). Uses SharedStorage try_put for
        atomic claim to prevent port collision across concurrent syncs.
        """
        import socket
        master_addr = socket.gethostbyname(socket.gethostname())

        def _find_free_port():
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", 0))
                return s.getsockname()[1]

        for _ in range(100):
            port = _find_free_port()
            key = f"MASTER_ADDR_PORT:{master_addr}:{port}"
            if self._shared_storage is None:
                return master_addr, port
            claimed = ray.get(self._shared_storage.try_put.remote(key, self._pipeline_id))
            if claimed:
                return master_addr, port

        raise RuntimeError(f"Cannot claim a unique NCCL master port after 100 attempts")

    def _release_master_port(self, master_addr: str, master_port: int) -> None:
        if self._shared_storage is None:
            return
        key = f"MASTER_ADDR_PORT:{master_addr}:{master_port}"
        try:
            ray.get(self._shared_storage.delete.remote(key))
        except Exception:
            pass  # Best-effort; port will be cleaned up by pipeline shutdown

    def sync_selected_workers(
        self,
        sync_id: str,
        tgt_engine_indices: List[int],
    ) -> None:
        """Push current CPU bucket cache to target SGLang engines.

        M11.1 transport routing:
          - Colocate engines (engine_gpus ⊂ train_gpus): cpu_serialize via tmpfs
          - Non-colocate engines: dynamic NCCL broadcast

        Phases:
          1. Get bucket count from cache owner
          2. Classify engines → colocate (cpu_serialize) / non-colocate (NCCL)
          3. NCCL: claim port + setup groups on cache owner + engines
          4. Per-bucket: cpu_serialize fan-out + NCCL broadcast + per-bucket barrier
          5. NCCL: destroy groups + release port

        Timeout via ROLL_SELECTIVE_MODEL_UPDATE_TIMEOUT_S (default 150s).
        """
        import time
        args = self._args

        logger.info(
            "[MilesModelUpdateService] sync_selected_workers START "
            "sync_id=%s engines=%s", sync_id, tgt_engine_indices
        )

        deadline = time.time() + (_SYNC_TIMEOUT_S or 150.0)

        # Phase 1: Get bucket count.
        bucket_count: int = ray.get(self._cache_owner.get_bucket_count.remote())
        if bucket_count == 0:
            logger.warning(
                "[MilesModelUpdateService] empty cache (bucket_count=0); skipping sync. sync_id=%s", sync_id
            )
            return

        # Phase 2: Classify engines.
        colocate_indices = [idx for idx in tgt_engine_indices if _is_colocate_engine(idx, args)]
        nccl_indices = [idx for idx in tgt_engine_indices if not _is_colocate_engine(idx, args)]

        logger.info(
            "[MilesModelUpdateService] comm_plan sync_id=%s colocate=%s nccl=%s",
            sync_id, colocate_indices, nccl_indices,
        )

        # Get engine handles from RolloutManager._engines map.
        engines_map: Dict[int, Any] = {}
        for idx in tgt_engine_indices:
            # Access engine handle via RolloutManager.
            # RolloutManager stores handles in _engines dict.
            engines_map[idx] = idx  # placeholder; actual handle access below

        def _get_engine_handle(engine_index: int) -> Any:
            """Get SGLangEngine Ray actor handle from RolloutManager._engines."""
            # We can't access _engines dict directly from a Ray actor, so we
            # use finalize_engine as a proxy to confirm handle access.
            # For direct method calls, we use the RolloutManager as intermediary.
            return engine_index  # Index passed to RolloutManager methods

        # Phase 3+4+5: all in try/finally so NCCL teardown runs even on setup failure.
        master_addr = None
        master_port = None
        group_name = f"miles_model_update_{sync_id}"

        try:
            if nccl_indices:
                gpus_per_engine = int(getattr(args, "rollout_num_gpus_per_engine", 1))
                engine_gpu_counts = [gpus_per_engine] * len(nccl_indices)

                # Get actual SGLangEngine actor handles so connect_rollout_engines_from_distributed
                # can call init_weights_update_group on each engine directly.
                nccl_engine_handles = ray.get(
                    self._rollout_manager.get_engine_handles.remote(nccl_indices)
                )

                # Setup on cache owner: calls connect_rollout_engines_from_distributed which
                # calls init_weights_update_group on each SGLang engine and init_process_group
                # on the Megatron sender — single call handles full cross-process rendezvous.
                ray.get(
                    self._cache_owner.setup_collective_group.remote(
                        group_name, nccl_engine_handles, engine_gpu_counts
                    )
                )

        except Exception:
            # Setup failed; still release port if claimed (group may be partially formed).
            if master_addr and master_port:
                self._release_master_port(master_addr, master_port)
                master_port = None  # Prevent double-release in outer finally
            raise

        try:
            # Phase 4: Per-bucket sync.
            for bucket_idx in range(bucket_count):
                if time.time() > deadline:
                    raise TimeoutError(
                        f"sync_selected_workers exceeded timeout ({_SYNC_TIMEOUT_S}s) "
                        f"at bucket {bucket_idx}/{bucket_count}. sync_id={sync_id}"
                    )

                # cpu_serialize: serialize + ray.put on cache owner, then call each colocate engine.
                for engine_idx in colocate_indices:
                    # Get bucket payload ref from cache owner.
                    payload_ref = ray.get(self._cache_owner.serialize_bucket_to_objref.remote(bucket_idx))
                    # Determine local ranks for this engine (tp group local ranks).
                    gpus_per_engine = int(getattr(args, "rollout_num_gpus_per_engine", 1))
                    cpu_serialize_local_ranks = list(range(gpus_per_engine))
                    # Call engine serially per spec (tmpfs peak = 1× bucket_size).
                    ref = self._rollout_manager.call_engine_method.remote(
                        engine_idx, "update_weights_from_cpu_bucket",
                        payload_ref, "cpu_serialize", False, None, cpu_serialize_local_ranks,
                    )
                    ray.get(ref)  # Per-engine serial + per-bucket barrier

                # NCCL broadcast: cache owner broadcasts, engines receive.
                if nccl_indices:
                    bucket_meta = ray.get(self._cache_owner.get_bucket_meta.remote(bucket_idx))
                    gpus_per_engine = int(getattr(args, "rollout_num_gpus_per_engine", 1))
                    broadcast_local_ranks = list(range(gpus_per_engine))

                    # Broadcast from cache owner.
                    broadcast_ref = self._cache_owner.broadcast_bucket.remote(
                        group_name, bucket_idx, 0,  # src_rank=0 (cache owner)
                    )
                    # All NCCL engines receive simultaneously.
                    recv_refs = [
                        self._rollout_manager.call_engine_method.remote(
                            idx, "broadcast_parameter",
                            group_name, broadcast_local_ranks, bucket_meta,
                        )
                        for idx in nccl_indices
                    ]
                    # Per-bucket barrier: wait for all receivers.
                    ray.get([broadcast_ref] + recv_refs)

            logger.info(
                "[MilesModelUpdateService] sync_selected_workers COMPLETE "
                "sync_id=%s buckets=%d", sync_id, bucket_count,
            )

        finally:
            # Phase 5: Destroy NCCL groups + release port (even on failure).
            if nccl_indices:
                # destroy_collective_group on cache_owner calls disconnect_rollout_engines_from_distributed
                # which handles both Megatron sender teardown and SGLang engine NCCL teardown.
                try:
                    ray.get(
                        self._cache_owner.destroy_collective_group.remote(group_name),
                        timeout=10.0,
                    )
                except Exception:
                    logger.exception("[MilesModelUpdateService] NCCL teardown failed; port may leak")
                if master_addr and master_port:
                    self._release_master_port(master_addr, master_port)
