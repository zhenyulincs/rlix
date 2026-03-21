from __future__ import annotations

import asyncio
import importlib
import sys
import threading
import types
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
RLIX_ROOT = REPO_ROOT / "rlix"
ROLL_ROOT = REPO_ROOT / "external" / "ROLL_rlix"


# ---------------------------------------------------------------------------
# Import helpers (reuse pattern from test_scheduler_apply_plan_invariants.py)
# ---------------------------------------------------------------------------


def _install_import_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    for module_name in list(sys.modules):
        if module_name == "ray" or module_name.startswith("rlix"):
            monkeypatch.delitem(sys.modules, module_name, raising=False)

    ray_stub = types.ModuleType("ray")

    def _remote(*args, **kwargs):
        def _decorate(obj):
            return obj

        return _decorate

    ray_stub.remote = _remote
    ray_stub.get_actor = lambda *args, **kwargs: None
    ray_stub.get = lambda value: value
    monkeypatch.setitem(sys.modules, "ray", ray_stub)

    package_roots = {
        "rlix": RLIX_ROOT,
        "rlix.protocol": RLIX_ROOT / "protocol",
        "rlix.scheduler": RLIX_ROOT / "scheduler",
        "rlix.utils": RLIX_ROOT / "utils",
    }
    for module_name, module_path in package_roots.items():
        package_module = types.ModuleType(module_name)
        package_module.__path__ = [str(module_path)]  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, module_name, package_module)


def _load_scheduler_modules(monkeypatch: pytest.MonkeyPatch):
    _install_import_stubs(monkeypatch)
    scheduler_module = importlib.import_module("rlix.scheduler.scheduler")
    protocol_types = importlib.import_module("rlix.protocol.types")
    scheduler_types = importlib.import_module("rlix.scheduler.types")
    return scheduler_module, protocol_types, scheduler_types


def _load_protocol_types(monkeypatch: pytest.MonkeyPatch):
    _install_import_stubs(monkeypatch)
    return importlib.import_module("rlix.protocol.types")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_progress_report(
    protocol_types: Any,
    *,
    pipeline_id: str = "ft_000000000000",
    step_target: int = 100,
    metrics: Optional[Dict[str, Any]] = None,
) -> Any:
    return protocol_types.ProgressReport(
        pipeline_id=pipeline_id,
        step_target_trajectories=step_target,
        fifo_timestamp=1.0,
        metrics=metrics,
    )


class _FakeRemoteMethod:
    """Mimics a Ray actor method handle: calling .remote(args) captures the call."""

    def __init__(self, capture_fn: Any) -> None:
        self._capture_fn = capture_fn

    def remote(self, *args: Any, **kwargs: Any) -> None:
        self._capture_fn(*args, **kwargs)


class _FakeSchedulerHandle:
    """Captures calls to report_progress.remote() and clear_progress.remote()."""

    def __init__(self) -> None:
        self.emitted_reports: List[Any] = []
        self.cleared_pipeline_ids: List[str] = []
        self.report_progress = _FakeRemoteMethod(self._capture_report)
        self.clear_progress = _FakeRemoteMethod(self._capture_clear)

    def _capture_report(self, report: Any) -> None:
        self.emitted_reports.append(report)

    def _capture_clear(self, *, pipeline_id: str) -> None:
        self.cleared_pipeline_ids.append(pipeline_id)


def _install_coordinator_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Extend import stubs so rlix.pipeline.coordinator can be imported.

    The coordinator module imports from rlix.pipeline.utils, rlix.protocol.coordinator,
    rlix.utils.env, and rlix.utils.ray — all need package stubs registered.
    """
    extra_packages = {
        "rlix.pipeline": RLIX_ROOT / "pipeline",
    }
    for module_name, module_path in extra_packages.items():
        if module_name not in sys.modules:
            package_module = types.ModuleType(module_name)
            package_module.__path__ = [str(module_path)]  # type: ignore[attr-defined]
            monkeypatch.setitem(sys.modules, module_name, package_module)


def _make_coordinator_for_test(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple:
    """Build a minimal PipelineCoordinator instance for aggregation tests.

    Returns (protocol_types, coordinator, fake_scheduler).
    Imports the real class but bypasses __init__ (which needs Ray actors),
    then sets up the minimal fields needed for aggregation methods.
    """
    _install_import_stubs(monkeypatch)
    _install_coordinator_stubs(monkeypatch)

    protocol_types = importlib.import_module("rlix.protocol.types")
    coord_module = importlib.import_module("rlix.pipeline.coordinator")
    coordinator_cls = coord_module.PipelineCoordinator

    fake_scheduler = _FakeSchedulerHandle()

    # Bypass __init__ (which requires Ray actor handles) and set fields directly.
    coordinator = object.__new__(coordinator_cls)
    coordinator._pipeline_id = "ft_000000000000"
    coordinator._rlix_scheduler = fake_scheduler
    coordinator._scheduler_reports = {}
    coordinator._coord_progress_last_bucket = None
    coordinator._progress_lock = threading.Lock()

    return protocol_types, coordinator, fake_scheduler


# ===========================================================================
# Scheduler derivation tests
# ===========================================================================


class TestDeriveRemainingFromReport:
    """Tests for SchedulerImpl._derive_remaining_from_report."""

    def test_under_target(self, monkeypatch: pytest.MonkeyPatch) -> None:
        scheduler_module, protocol_types, _ = _load_scheduler_modules(monkeypatch)
        report = _make_progress_report(
            protocol_types, step_target=100, metrics={"completed": 30, "mode": "aggregated"}
        )
        remaining = scheduler_module.SchedulerImpl._derive_remaining_from_report(report)
        assert remaining == 70.0

    def test_exact_target(self, monkeypatch: pytest.MonkeyPatch) -> None:
        scheduler_module, protocol_types, _ = _load_scheduler_modules(monkeypatch)
        report = _make_progress_report(
            protocol_types, step_target=100, metrics={"completed": 100, "mode": "aggregated"}
        )
        remaining = scheduler_module.SchedulerImpl._derive_remaining_from_report(report)
        assert remaining == 0.0

    def test_overshoot(self, monkeypatch: pytest.MonkeyPatch) -> None:
        scheduler_module, protocol_types, _ = _load_scheduler_modules(monkeypatch)
        report = _make_progress_report(
            protocol_types, step_target=100, metrics={"completed": 120, "mode": "aggregated"}
        )
        remaining = scheduler_module.SchedulerImpl._derive_remaining_from_report(report)
        assert remaining == 0.0

    def test_missing_completed_defaults_to_full_remaining(self, monkeypatch: pytest.MonkeyPatch) -> None:
        scheduler_module, protocol_types, _ = _load_scheduler_modules(monkeypatch)
        report = _make_progress_report(protocol_types, step_target=100, metrics={"mode": "aggregated"})
        remaining = scheduler_module.SchedulerImpl._derive_remaining_from_report(report)
        assert remaining == 100.0

    @pytest.mark.parametrize(
        "completed,step_target",
        [
            (0, 1),
            (0, 50),
            (0, 100),
            (50, 1),
            (50, 50),
            (50, 100),
            (100, 1),
            (100, 50),
            (100, 100),
            (200, 1),
            (200, 50),
            (200, 100),
        ],
    )
    def test_percent_remaining_never_exceeds_one(
        self, monkeypatch: pytest.MonkeyPatch, completed: int, step_target: int
    ) -> None:
        scheduler_module, protocol_types, _ = _load_scheduler_modules(monkeypatch)
        report = _make_progress_report(
            protocol_types,
            step_target=step_target,
            metrics={"completed": completed, "mode": "aggregated"},
        )
        remaining = scheduler_module.SchedulerImpl._derive_remaining_from_report(report)
        assert remaining >= 0.0
        assert remaining <= float(step_target)
        percent_remaining = remaining / float(step_target)
        assert percent_remaining <= 1.0


# ===========================================================================
# Scheduler ingress validation tests
# ===========================================================================


class TestSchedulerIngressValidation:
    """Tests for report_progress() ingress validation."""

    def test_missing_completed_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        scheduler_module, protocol_types, _ = _load_scheduler_modules(monkeypatch)
        scheduler = scheduler_module.SchedulerImpl()
        scheduler._state.pipeline_registry["ft_000000000000"] = {
            "cluster_configs": {"actor_infer": {"tp_size": 1}},
        }
        report = _make_progress_report(
            protocol_types,
            step_target=100,
            metrics={"mode": "aggregated", "collected": 50},
        )
        with pytest.raises(ValueError, match="missing required 'completed' metric"):
            asyncio.get_event_loop().run_until_complete(scheduler.report_progress(report))

    def test_remaining_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        scheduler_module, protocol_types, _ = _load_scheduler_modules(monkeypatch)
        scheduler = scheduler_module.SchedulerImpl()
        scheduler._state.pipeline_registry["ft_000000000000"] = {
            "cluster_configs": {"actor_infer": {"tp_size": 1}},
        }
        report = _make_progress_report(
            protocol_types,
            step_target=100,
            metrics={"mode": "aggregated", "completed": 50, "remaining": 50},
        )
        with pytest.raises(ValueError, match="contains wire-level 'remaining'"):
            asyncio.get_event_loop().run_until_complete(scheduler.report_progress(report))


# ===========================================================================
# Coordinator aggregation tests
# ===========================================================================


class TestCoordinatorAggregation:
    """Tests for coordinator _aggregate_and_emit with new collected-based contract."""

    def test_single_stream_bucket_cadence(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Single stream: emit on bucket transition, suppress within same bucket."""
        protocol_types, coordinator, fake_scheduler = _make_coordinator_for_test(monkeypatch)

        # First report: 10% done (bucket 5).
        report_10pct = _make_progress_report(
            protocol_types, step_target=100, metrics={"collected": 10, "mode": "train", "new_batch": False}
        )
        coordinator._scheduler_reports["train:__fft__"] = report_10pct
        coordinator._aggregate_and_emit(force=False)
        assert len(fake_scheduler.emitted_reports) == 1

        # Same bucket: should not emit.
        coordinator._aggregate_and_emit(force=False)
        assert len(fake_scheduler.emitted_reports) == 1

        # Advance to 14% (bucket 7): should emit.
        report_14pct = _make_progress_report(
            protocol_types, step_target=100, metrics={"collected": 14, "mode": "train", "new_batch": False}
        )
        coordinator._scheduler_reports["train:__fft__"] = report_14pct
        coordinator._aggregate_and_emit(force=False)
        assert len(fake_scheduler.emitted_reports) == 2

    def test_multi_stream_sums_all_active(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Two active streams: aggregated = sum of both."""
        protocol_types, coordinator, fake_scheduler = _make_coordinator_for_test(monkeypatch)

        coordinator._scheduler_reports["train:adapter_a"] = _make_progress_report(
            protocol_types, step_target=100, metrics={"collected": 80, "mode": "train", "new_batch": False}
        )
        coordinator._scheduler_reports["train:adapter_b"] = _make_progress_report(
            protocol_types, step_target=100, metrics={"collected": 20, "mode": "train", "new_batch": False}
        )
        coordinator._aggregate_and_emit(force=True)

        assert len(fake_scheduler.emitted_reports) == 1
        agg = fake_scheduler.emitted_reports[0]
        assert agg.step_target_trajectories == 200
        assert agg.metrics["completed"] == 100  # 80 + 20
        assert agg.metrics["collected"] == 100  # 80 + 20 (both under target, so same)

    def test_aggregate_with_overshoot_stream(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Overshoot stream: completed clamped per-stream, collected raw sum."""
        protocol_types, coordinator, fake_scheduler = _make_coordinator_for_test(monkeypatch)

        coordinator._scheduler_reports["train:adapter_a"] = _make_progress_report(
            protocol_types, step_target=100, metrics={"collected": 150, "mode": "train", "new_batch": False}
        )
        coordinator._scheduler_reports["train:adapter_b"] = _make_progress_report(
            protocol_types, step_target=100, metrics={"collected": 20, "mode": "train", "new_batch": False}
        )
        coordinator._aggregate_and_emit(force=True)

        assert len(fake_scheduler.emitted_reports) == 1
        agg = fake_scheduler.emitted_reports[0]
        assert agg.step_target_trajectories == 200
        assert agg.metrics["completed"] == 120  # min(150,100) + 20
        assert agg.metrics["collected"] == 170  # 150 + 20 (raw sum)

    def test_clear_progress_stream_removes_contribution(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """After clearing one stream, aggregation uses only the remaining stream."""
        protocol_types, coordinator, fake_scheduler = _make_coordinator_for_test(monkeypatch)

        coordinator._scheduler_reports["train:adapter_a"] = _make_progress_report(
            protocol_types, step_target=100, metrics={"collected": 80, "mode": "train", "new_batch": False}
        )
        coordinator._scheduler_reports["train:adapter_b"] = _make_progress_report(
            protocol_types, step_target=100, metrics={"collected": 20, "mode": "train", "new_batch": False}
        )

        # Clear adapter_a.
        coordinator.clear_progress_stream(mode="train", adapter_id="adapter_a")

        # Should have force-emitted with only adapter_b.
        last_report = fake_scheduler.emitted_reports[-1]
        assert last_report.step_target_trajectories == 100
        assert last_report.metrics["completed"] == 20
        assert last_report.metrics["collected"] == 20

    def test_aggregate_percent_completed_never_exceeds_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Even with overshoot, percent_completed is clamped to [0, 1]."""
        protocol_types, coordinator, fake_scheduler = _make_coordinator_for_test(monkeypatch)

        coordinator._scheduler_reports["train:__fft__"] = _make_progress_report(
            protocol_types, step_target=100, metrics={"collected": 200, "mode": "train", "new_batch": False}
        )
        coordinator._aggregate_and_emit(force=True)

        agg = fake_scheduler.emitted_reports[0]
        # completed clamped to 100, percent = 100/100 = 1.0, bucket = 50.
        assert agg.metrics["completed"] == 100
        assert agg.metrics["bucket"] <= 50

    def test_per_stream_clamping_from_raw_collected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Stream with collected > target contributes completed_clamped, not raw collected."""
        protocol_types, coordinator, fake_scheduler = _make_coordinator_for_test(monkeypatch)

        coordinator._scheduler_reports["train:__fft__"] = _make_progress_report(
            protocol_types, step_target=100, metrics={"collected": 150, "mode": "train", "new_batch": False}
        )
        coordinator._aggregate_and_emit(force=True)

        agg = fake_scheduler.emitted_reports[0]
        assert agg.metrics["completed"] == 100  # clamped
        assert agg.metrics["collected"] == 150  # raw

    def test_missing_collected_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Report without 'collected' metric raises ValueError."""
        protocol_types, coordinator, _ = _make_coordinator_for_test(monkeypatch)

        report = _make_progress_report(protocol_types, step_target=100, metrics={"mode": "train", "new_batch": False})
        with pytest.raises(ValueError, match="missing required 'collected' metric"):
            coordinator.report_progress_from_scheduler(report)

    def test_wire_remaining_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Report with wire-level 'remaining' raises ValueError."""
        protocol_types, coordinator, _ = _make_coordinator_for_test(monkeypatch)

        report = _make_progress_report(
            protocol_types,
            step_target=100,
            metrics={"collected": 50, "remaining": 50, "mode": "train", "new_batch": False},
        )
        with pytest.raises(ValueError, match="contains wire-level 'remaining'"):
            coordinator.report_progress_from_scheduler(report)


# ===========================================================================
# Planner weight tests
# ===========================================================================


class TestPlannerWeights:
    """Tests that gap-ratio weights correctly reflect derived remaining."""

    def test_overshot_pipeline_gets_zero_weight(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Pipeline with completed > target gets zero remaining, thus zero weight."""
        scheduler_module, protocol_types, scheduler_types = _load_scheduler_modules(monkeypatch)

        report_overshot = _make_progress_report(
            protocol_types, step_target=100, metrics={"completed": 200, "mode": "aggregated"}
        )
        report_incomplete = _make_progress_report(
            protocol_types,
            pipeline_id="ft_111111111111",
            step_target=100,
            metrics={"completed": 50, "mode": "aggregated"},
        )

        remaining_overshot = scheduler_module.SchedulerImpl._derive_remaining_from_report(report_overshot)
        remaining_incomplete = scheduler_module.SchedulerImpl._derive_remaining_from_report(report_incomplete)

        assert remaining_overshot == 0.0
        assert remaining_incomplete == 50.0

        # Simulate weight calculation: remaining * tp_size.
        tp_size = 2
        weight_overshot = remaining_overshot * tp_size
        weight_incomplete = remaining_incomplete * tp_size
        total_weight = weight_overshot + weight_incomplete

        assert weight_overshot == 0.0
        assert total_weight > 0.0
        ratio_overshot = weight_overshot / total_weight if total_weight > 0 else 0.0
        ratio_incomplete = weight_incomplete / total_weight if total_weight > 0 else 0.0
        assert ratio_overshot == 0.0
        assert ratio_incomplete == 1.0

    def test_pending_request_inflation_only_way_to_exceed_baseline(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without pending request, completed == target means zero remaining.

        With pending request, remaining inflates by step_target (scheduler policy).
        """
        scheduler_module, protocol_types, _ = _load_scheduler_modules(monkeypatch)

        report = _make_progress_report(
            protocol_types, step_target=100, metrics={"completed": 100, "mode": "aggregated"}
        )
        baseline_remaining = scheduler_module.SchedulerImpl._derive_remaining_from_report(report)
        assert baseline_remaining == 0.0

        # Simulate pending-request inflation (scheduler.py lines 2234-2238).
        step_target = 100.0
        inflated_remaining = baseline_remaining + step_target
        assert inflated_remaining == 100.0
        assert inflated_remaining > baseline_remaining
