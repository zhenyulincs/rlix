"""MilesCoordinator — per-pipeline coordinator for MILES fullasync GRPO pipelines.

Does NOT subclass PipelineCoordinator to avoid triggering ROLL config validators
(_validate_config_schema, _validate_cpu_only_reward, _validate_vllm_sleep_level,
_validate_offload_nccl) which expect ROLL WorkerConfig attributes absent from MILES args.

Implements the Coordinator ABC. Named actor: "rlix:coordinator:{pipeline_id}"
in namespace get_pipeline_namespace(pipeline_id).
"""
from __future__ import annotations

import logging
import math
import os
import threading
import time
from typing import Any, Dict, List, Optional, Set

import ray

from rlix.pipeline.utils import validate_resize_params
from rlix.protocol.coordinator import Coordinator
from rlix.protocol.types import (
    COORDINATOR_ACTOR_NAME_PREFIX,
    PIPELINE_ACTOR_NAME_PREFIX,
    RLIX_NAMESPACE,
    SCHEDULER_ACTOR_NAME,
    ActionResponse,
    ProgressReport,
    get_pipeline_namespace,
)
from rlix.utils.env import pipeline_identity_env_vars
from rlix.utils.ray import get_actor_or_raise

logger = logging.getLogger(__name__)

# Max concurrent RPCs (progress fire-and-forget + resize must not block each other).
MILES_COORDINATOR_MAX_CONCURRENCY: int = 4

# Timeout for _resize_sync_lock acquisition (seconds). Matches ROLL default.
_RESIZE_LOCK_TIMEOUT_S: float = float(os.environ.get("RLIX_RESIZE_LOCK_TIMEOUT_S", "180"))


class MilesCoordinator(Coordinator):
    """Per-pipeline coordinator actor for MILES pipelines.

    Responsibilities:
    - Translate scheduler resize_infer RPCs into RolloutManager.shrink_engines /
      expand_engines calls (F2, first-build identity mapping: dp_rank == engine_index).
    - Coordinate sync_base_weights_to_active / sync_selected_workers via
      MilesModelUpdateService (F5+F6, Phase C).
    - Aggregate progress reports and forward to central scheduler (F9, Phase D).
    - Bootstrap and maintain _active_engine_indices set.
    """

    def __init__(self, *, pipeline_id: str, pipeline_config: Any):
        from rlix.protocol.validation import validate_pipeline_id

        validate_pipeline_id(pipeline_id)
        self._pipeline_id: str = pipeline_id
        self._ray_namespace: str = get_pipeline_namespace(pipeline_id)
        self._pipeline_config: Any = pipeline_config

        # Pipeline identity env vars forwarded to child actors.
        self._pipeline_env_vars: Dict[str, str] = pipeline_identity_env_vars(
            pipeline_id=pipeline_id, ray_namespace=self._ray_namespace
        )

        # Singleton ResourceManager shared across all pipelines — provides shared PGs.
        from roll.distributed.scheduler.resource_manager import RollResourceManagerProxy

        num_gpus_per_node = getattr(pipeline_config, "num_gpus_per_node", 4)
        self._resource_manager_proxy = RollResourceManagerProxy(num_gpus_per_node=num_gpus_per_node)
        self._resource_manager_node0_pg = self._resource_manager_proxy.node2pg.get(0)

        # Serializes resize_infer and sync_base_weights_to_active.
        self._resize_sync_lock = threading.Lock()
        # Active engine indices (engine_index == scheduler dp_rank in first-build identity topology).
        # Bootstrapped by initialize_pipeline Step 10; updated by resize_infer.
        self._active_engine_indices: Set[int] = set()

        # Progress aggregation (Phase D).
        self._progress_lock = threading.Lock()
        self._scheduler_reports: Dict[str, ProgressReport] = {}
        self._coord_progress_last_bucket: Optional[int] = None

        # Pipeline actor handle (created by create_pipeline_actor).
        self._pipeline_actor: Any = None

        # MilesModelUpdateService lazy slot (Phase C).
        # Populated by register_model_update_resources() called from initialize_pipeline.
        self._model_update_resources: Optional[Dict[str, Any]] = None
        self._model_update_service: Any = None

        # Guard for bootstrap_active_engines: prevents silent double-call overwrite.
        self._bootstrapped: bool = False

        # Central scheduler actor.
        self._rlix_scheduler = get_actor_or_raise(
            SCHEDULER_ACTOR_NAME,
            RLIX_NAMESPACE,
            error_context="MilesCoordinator requires the central scheduler actor at startup.",
        )

    # -------------------------------------------------------------------------
    # Coordinator ABC — pipeline actor lifecycle
    # -------------------------------------------------------------------------

    def create_pipeline_actor(self, *, pipeline_config: Any) -> Any:
        """Create (idempotent) and return the MilesPipeline Ray actor."""
        if self._pipeline_actor is not None:
            return self._pipeline_actor

        from rlix.pipeline.miles_pipeline import MilesPipeline
        from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

        PipelineActor = ray.remote(MilesPipeline)
        self._pipeline_actor = PipelineActor.options(
            name=f"{PIPELINE_ACTOR_NAME_PREFIX}{self._pipeline_id}",
            namespace=self._ray_namespace,
            get_if_exists=True,
            max_restarts=0,
            max_task_retries=0,
            max_concurrency=2,  # main loop + inbound stop/health-check
            runtime_env={"env_vars": self._pipeline_env_vars},
            num_gpus=0.01,
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=self._resource_manager_node0_pg,
            ),
        ).remote(pipeline_id=self._pipeline_id, pipeline_config=pipeline_config)
        return self._pipeline_actor

    # -------------------------------------------------------------------------
    # Coordinator ABC — resize
    # -------------------------------------------------------------------------

    def resize_infer(self, dp_ranks_to_remove: List[int], dp_ranks_to_add: List[int]) -> ActionResponse:
        """Translate scheduler dp_rank resize into MILES engine index shrink/expand.

        First-build identity mapping: dp_rank == engine_index (sorted contiguous
        infer_device_mapping enforced by F10). Non-contiguous mapping adapter is
        a follow-up (F12 spec §Cut 1').
        """
        validate_resize_params(dp_ranks_to_remove, dp_ranks_to_add)

        acquired = self._resize_sync_lock.acquire(timeout=_RESIZE_LOCK_TIMEOUT_S)
        if not acquired:
            raise RuntimeError(
                f"resize_infer timed out waiting for _resize_sync_lock after {_RESIZE_LOCK_TIMEOUT_S}s. "
                f"pipeline_id={self._pipeline_id!r}"
            )
        try:
            if self._pipeline_actor is None:
                raise RuntimeError(
                    f"Pipeline actor not yet created for resize_infer. pipeline_id={self._pipeline_id!r}"
                )
            # Identity mapping: dp_rank == engine_index in first-build topology.
            engine_indices_to_remove = list(dp_ranks_to_remove)
            engine_indices_to_add = list(dp_ranks_to_add)

            if engine_indices_to_remove:
                ray.get(self._pipeline_actor.shrink_engines.remote(engine_indices_to_remove))
                self._active_engine_indices -= set(engine_indices_to_remove)
            else:
                ray.get(self._pipeline_actor.expand_engines.remote(engine_indices_to_add))
                self._active_engine_indices |= set(engine_indices_to_add)
        finally:
            self._resize_sync_lock.release()

        return ActionResponse(success=True)

    # -------------------------------------------------------------------------
    # Coordinator ABC — weight sync
    # -------------------------------------------------------------------------

    def sync_lora_weights(self, *, loras_to_sync: List[str]) -> None:
        """LoRA weight sync — not supported in M11.1 (LoRA out of scope until M11.5)."""
        raise NotImplementedError(
            "sync_lora_weights is not supported in M11.1 (LoRA scope = M11.5). "
            f"pipeline_id={self._pipeline_id!r}"
        )

    def sync_base_weights_to_active(self) -> List[int]:
        """Push trained base model weights to all currently-awake SGLang engines.

        Holds _resize_sync_lock for the entire sync to prevent active_engine_indices
        from changing mid-broadcast (same invariant as PipelineCoordinator.sync_base_weights_to_active).

        Phase C fills in the actual MilesModelUpdateService invocation.
        """
        acquired = self._resize_sync_lock.acquire(timeout=_RESIZE_LOCK_TIMEOUT_S)
        if not acquired:
            raise RuntimeError(
                f"sync_base_weights_to_active timed out waiting for _resize_sync_lock "
                f"after {_RESIZE_LOCK_TIMEOUT_S}s. pipeline_id={self._pipeline_id!r}"
            )
        try:
            active_engines = sorted(self._active_engine_indices)
            if not active_engines:
                return []
            svc = self._get_or_create_model_update_service()
            import uuid
            sync_id = f"{self._pipeline_id}_active_{uuid.uuid4().hex[:8]}"
            ray.get(svc.sync_selected_workers.remote(sync_id=sync_id, tgt_engine_indices=active_engines))
            return active_engines
        finally:
            self._resize_sync_lock.release()

    # -------------------------------------------------------------------------
    # Coordinator ABC — progress
    # -------------------------------------------------------------------------

    def report_progress_from_scheduler(self, report: ProgressReport) -> None:
        """Aggregate per-scheduler progress and forward to central scheduler (Phase D)."""
        metrics = report.metrics if isinstance(report.metrics, dict) else {}
        if "collected" not in metrics:
            raise ValueError(f"ProgressReport missing 'collected' metric. pipeline_id={report.pipeline_id!r}")
        if "remaining" in metrics:
            raise ValueError(
                f"ProgressReport contains wire-level 'remaining'; send 'collected' instead. "
                f"pipeline_id={report.pipeline_id!r}"
            )
        mode = str(metrics.get("mode", "train"))
        adapter_id = metrics.get("adapter_id")
        scheduler_key = f"{mode}:{adapter_id if adapter_id is not None else '__fft__'}"
        is_new_batch = bool(metrics.get("new_batch", False))

        with self._progress_lock:
            self._scheduler_reports[scheduler_key] = report
            self._aggregate_and_emit(force=is_new_batch)

    def clear_progress_stream(self, *, mode: str, adapter_id: Optional[str]) -> None:
        """Remove a progress stream from aggregation (Phase D)."""
        scheduler_key = f"{mode}:{adapter_id if adapter_id is not None else '__fft__'}"
        with self._progress_lock:
            removed = self._scheduler_reports.pop(scheduler_key, None)
            if removed is None:
                return
            if self._scheduler_reports:
                self._aggregate_and_emit(force=True)
            else:
                self._coord_progress_last_bucket = None
                self._rlix_scheduler.clear_progress.remote(pipeline_id=self._pipeline_id)

    def _aggregate_and_emit(self, *, force: bool) -> None:
        """Sum clamped completion across streams and emit to scheduler. Must hold _progress_lock."""
        if not self._scheduler_reports:
            return
        total_required = 0.0
        total_completed = 0.0
        total_collected = 0.0
        for rpt in self._scheduler_reports.values():
            rpt_metrics = rpt.metrics if isinstance(rpt.metrics, dict) else {}
            step_target = float(max(int(rpt.step_target_trajectories), 1))
            collected_raw = max(0.0, float(rpt_metrics.get("collected", 0)))
            completed_clamped = min(collected_raw, step_target)
            total_required += step_target
            total_completed += completed_clamped
            total_collected += collected_raw
        if total_required <= 0:
            return
        percent = min(total_completed / total_required, 1.0)
        bucket = math.floor(percent * 50)
        if not force and bucket == self._coord_progress_last_bucket:
            return
        self._coord_progress_last_bucket = bucket
        aggregated = ProgressReport(
            pipeline_id=str(self._pipeline_id),
            step_target_trajectories=int(total_required),
            fifo_timestamp=time.time(),
            metrics={
                "mode": "aggregated",
                "collected": int(total_collected),
                "completed": int(total_completed),
                "bucket": int(bucket),
                "new_batch": force,
            },
        )
        self._rlix_scheduler.report_progress.remote(aggregated)

    # -------------------------------------------------------------------------
    # MILES-specific methods
    # -------------------------------------------------------------------------

    def bootstrap_active_engines(self, engine_indices: Set[int]) -> None:
        """Set initial active engine set after actor_infer GENERATION allocation.

        Called exactly once by MilesPipeline.initialize_pipeline() Step 10.
        Subsequent updates flow through resize_infer().
        Raises RuntimeError if called a second time (prevents silent overwrite
        of a live set during resize).
        """
        with self._resize_sync_lock:
            if self._bootstrapped:
                raise RuntimeError(
                    f"bootstrap_active_engines called twice: "
                    f"existing={self._active_engine_indices}, new={engine_indices}. "
                    f"pipeline_id={self._pipeline_id!r}"
                )
            self._active_engine_indices = set(engine_indices)
            self._bootstrapped = True
        logger.info(
            "[MilesCoordinator] bootstrap_active_engines pipeline_id=%s engines=%s",
            self._pipeline_id,
            sorted(engine_indices),
        )

    def register_model_update_resources(
        self,
        *,
        cache_owner_actor: Any,
        rollout_manager: Any,
    ) -> None:
        """Cache sender + receiver handles for lazy MilesModelUpdateService construction.

        Called by MilesPipeline.initialize_pipeline() Step 10. Defers service actor
        creation to first sync call to avoid extra round-trip during init bootstrap.
        """
        self._model_update_resources = {
            "cache_owner_actor": cache_owner_actor,
            "rollout_manager": rollout_manager,
        }
        logger.info(
            "[MilesCoordinator] register_model_update_resources pipeline_id=%s", self._pipeline_id
        )

    def _get_or_create_model_update_service(self) -> Any:
        """Lazily create MilesModelUpdateService named actor on first sync call (Phase C)."""
        if self._model_update_service is not None:
            return self._model_update_service
        if self._model_update_resources is None:
            raise RuntimeError(
                "register_model_update_resources() must be called before sync. "
                f"pipeline_id={self._pipeline_id!r}"
            )
        from rlix.pipeline.miles_model_update_service import MilesModelUpdateService  # Phase C

        svc_name = f"rlix:miles_model_update_service:{self._pipeline_id}"
        self._model_update_service = MilesModelUpdateService.options(
            name=svc_name,
            namespace=self._ray_namespace,
            get_if_exists=True,
            lifetime="detached",
        ).remote(
            pipeline_id=self._pipeline_id,
            cache_owner_actor=self._model_update_resources["cache_owner_actor"],
            rollout_manager=self._model_update_resources["rollout_manager"],
            pipeline_config=self._pipeline_config,
        )
        ray.get(self._model_update_service.__ray_ready__.remote())
        return self._model_update_service
