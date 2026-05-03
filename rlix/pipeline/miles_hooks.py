"""MilesRLixHooks — RLix-side implementation of the RLixHooks protocol (F9).

Import seam: fully_async_rollout.py only calls methods on RLixHooks instances.
It never imports ProgressReport or touches RLix actor handles. All RLix wire
construction and coordinator RPCs live here.

Standalone mode uses miles.utils.rlix_hooks.NoOpRLixHooks (all no-ops).
"""
from __future__ import annotations

import math
import logging
from typing import Optional

from rlix.protocol.types import ProgressReport

logger = logging.getLogger(__name__)


class MilesRLixHooks:
    """RLix-side RLixHooks implementation for MILES fullasync rollout.

    Translates plain-kwargs progress calls from fully_async_rollout.py into
    ProgressReport wire messages forwarded to MilesCoordinator (fire-and-forget).

    Local state (per wait window):
      _progress_target_step:   weight version we are collecting groups for
      _step_target_groups:     total groups needed to close the demand window
      _local_completed:        groups collected so far matching target version
      _progress_last_bucket:   last 2% bucket sent (0..50); -1 = not yet sent
    """

    def __init__(self, *, pipeline_id: str, coordinator_handle) -> None:
        self._pipeline_id = pipeline_id
        self._coordinator = coordinator_handle

        self._progress_target_step: int = -1
        self._step_target_groups: int = 1
        self._local_completed: int = 0
        self._progress_last_bucket: int = -1
        self._active: bool = False

    def begin_progress_batch(
        self,
        *,
        target_weight_version: int,
        step_target_groups: int,
        initial_completed: int,
        mode: Optional[str] = None,
        adapter_id: Optional[str] = None,
    ) -> None:
        """Open generation demand window. Must be called before the wait loop."""
        self._progress_target_step = target_weight_version
        self._step_target_groups = max(step_target_groups, 1)
        self._local_completed = initial_completed  # batch-open snapshot (NOT reset to 0)
        self._progress_last_bucket = -1            # force first emit
        self._active = True

        # Emit opening snapshot with new_batch=True.
        initial_bucket = math.floor(self._local_completed / self._step_target_groups * 50)
        self.report_progress(
            collected=self._local_completed,
            bucket=initial_bucket,
            current_train_step=target_weight_version,
            new_batch=True,
            mode=mode,
            adapter_id=adapter_id,
        )
        self._progress_last_bucket = initial_bucket

    def bump_completed(self, *, target_weight_version: int) -> None:
        """Increment local counter when a group is pushed to data (hot path)."""
        if not self._active:
            return
        if target_weight_version != self._progress_target_step:
            return  # Cross-step group; don't count toward current demand window

        self._local_completed += 1
        bucket = math.floor(self._local_completed / self._step_target_groups * 50)
        remaining = max(self._step_target_groups - self._local_completed, 0)
        # 2% gate: only emit when bucket changes or demand window closes.
        if bucket == self._progress_last_bucket and remaining > 0:
            return
        self._progress_last_bucket = bucket
        self.report_progress(
            collected=self._local_completed,
            bucket=bucket,
            current_train_step=self._progress_target_step,
            new_batch=False,
        )

    def end_progress_batch(self) -> None:
        """Close the demand window (call in finally to guarantee cleanup)."""
        if not self._active:
            return
        self._active = False
        self.clear_progress()

    def clear_progress(self) -> None:
        """Signal coordinator to remove this stream from aggregation."""
        try:
            self._coordinator.clear_progress_stream.remote(
                mode="train", adapter_id=None
            )
        except Exception as exc:
            logger.warning("[MilesRLixHooks] clear_progress RPC failed: %s", exc)

    def report_progress(
        self,
        *,
        collected: int,
        bucket: int,
        current_train_step: int,
        new_batch: bool,
        mode: Optional[str] = None,
        adapter_id: Optional[str] = None,
    ) -> None:
        """Fire-and-forget ProgressReport to MilesCoordinator."""
        report = ProgressReport(
            pipeline_id=self._pipeline_id,
            step_target_trajectories=self._step_target_groups,
            metrics={
                "collected": collected,
                "bucket": bucket,
                "current_train_step": current_train_step,
                "new_batch": new_batch,
                "mode": mode or "train",
                "adapter_id": adapter_id,
            },
        )
        try:
            self._coordinator.report_progress_from_scheduler.remote(report)
        except Exception as exc:
            logger.warning("[MilesRLixHooks] report_progress RPC failed: %s", exc)
