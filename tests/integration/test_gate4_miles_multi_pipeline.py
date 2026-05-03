"""Gate 4 MILES multi-pipeline scheduler integration test.

Spec covered here: 4 GPU node, two MILES fullasync pipelines, and Pipeline B's
overlapping rollout engine sleeps while Pipeline A trains on GPUs 0,1.  When
Pipeline A releases those GPUs, Pipeline B's engine 0 wakes through scheduler
``resize_infer`` and receives the current weight version.
"""
from __future__ import annotations

import asyncio
import importlib
import sys
import types
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
RLIX_ROOT = REPO_ROOT / "rlix"


def _install_import_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    for module_name in list(sys.modules):
        if module_name == "ray" or module_name.startswith("rlix"):
            monkeypatch.delitem(sys.modules, module_name, raising=False)

    ray_stub = types.ModuleType("ray")

    def _remote(*args: Any, **kwargs: Any):
        def _decorate(obj: Any) -> Any:
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
        package_module.__path__ = [str(module_path)]
        monkeypatch.setitem(sys.modules, module_name, package_module)


def _load_scheduler_modules(monkeypatch: pytest.MonkeyPatch):
    _install_import_stubs(monkeypatch)
    scheduler_module = importlib.import_module("rlix.scheduler.scheduler")
    protocol_types = importlib.import_module("rlix.protocol.types")
    scheduler_types = importlib.import_module("rlix.scheduler.types")
    return scheduler_module, protocol_types, scheduler_types


class _ResizeInferRemote:
    def __init__(self, coordinator: "_FakeMilesCoordinator") -> None:
        self._coordinator = coordinator

    def remote(self, **kwargs: Any):
        async def _call() -> None:
            self._coordinator.resize_infer_impl(**kwargs)

        return _call()


class _FakeMilesRuntime:
    """Small MILES-runtime double for Gate 4 sleep/wake + version assertions."""

    def __init__(self, *, pipeline_id: str, namespace: str, initial_weight_version: int) -> None:
        self.pipeline_id = pipeline_id
        self.namespace = namespace
        self.active_engine_indices = {0, 1}
        self.cache_ready_step = initial_weight_version
        self.current_weight_version = initial_weight_version
        self.engine_weight_versions = {0: initial_weight_version, 1: initial_weight_version}
        self.events: list[tuple[str, tuple[int, ...]]] = []

    def shrink_engines(self, engine_indices: list[int]) -> None:
        self.events.append(("shrink_engines", tuple(engine_indices)))
        self.active_engine_indices -= set(engine_indices)

    def expand_engines(self, engine_indices: list[int]) -> None:
        self.events.append(("wake_partial", tuple(engine_indices)))
        self.events.append(("sync_selected_workers", tuple(engine_indices)))
        for engine_index in engine_indices:
            self.engine_weight_versions[engine_index] = self.cache_ready_step
        self.events.append(("finalize_engine", tuple(engine_indices)))
        for engine_index in engine_indices:
            self.engine_weight_versions[engine_index] = self.current_weight_version
        self.events.append(("set_weight_version", tuple(engine_indices)))
        self.active_engine_indices |= set(engine_indices)


class _FakeMilesCoordinator:
    def __init__(self, runtime: _FakeMilesRuntime) -> None:
        self.runtime = runtime
        self.resize_calls: list[dict[str, list[int]]] = []
        self.resize_infer = _ResizeInferRemote(self)

    def resize_infer_impl(self, *, dp_ranks_to_remove: list[int], dp_ranks_to_add: list[int]) -> None:
        self.resize_calls.append(
            {
                "dp_ranks_to_remove": list(dp_ranks_to_remove),
                "dp_ranks_to_add": list(dp_ranks_to_add),
            }
        )
        if dp_ranks_to_remove:
            self.runtime.shrink_engines(list(dp_ranks_to_remove))
        if dp_ranks_to_add:
            self.runtime.expand_engines(list(dp_ranks_to_add))


def _assert_exclusive_allocations(active_allocations: dict[str, Any]) -> None:
    owners_by_gpu: dict[int, str] = {}
    for cluster_id, allocation in active_allocations.items():
        for gpu_id in allocation.gpu_ids:
            assert gpu_id not in owners_by_gpu, (
                f"GPU {gpu_id} allocated to both {owners_by_gpu[gpu_id]} and {cluster_id}"
            )
            owners_by_gpu[gpu_id] = cluster_id


def test_gate4_miles_two_pipelines_sleep_wake_and_weight_sync(monkeypatch: pytest.MonkeyPatch) -> None:
    scheduler_module, protocol_types, scheduler_types = _load_scheduler_modules(monkeypatch)

    Priority = protocol_types.Priority
    ProgressReport = protocol_types.ProgressReport
    get_pipeline_namespace = protocol_types.get_pipeline_namespace
    ClusterAllocation = scheduler_types.ClusterAllocation
    PendingRequest = scheduler_types.PendingRequest
    Request = scheduler_types.Request

    scheduler = scheduler_module.SchedulerImpl()
    scheduler._topology_ready.set()
    scheduler._num_gpus = 4
    scheduler._required_gpus_per_node = 4
    scheduler._state.idle_gpus = {0, 1, 2, 3}

    pipeline_a = "ft_aaaaaaaaaaaa"
    pipeline_b = "ft_bbbbbbbbbbbb"
    namespace_a = get_pipeline_namespace(pipeline_a)
    namespace_b = get_pipeline_namespace(pipeline_b)

    async def _register_and_admit() -> None:
        for pipeline_id, namespace in ((pipeline_a, namespace_a), (pipeline_b, namespace_b)):
            await scheduler.register_pipeline(
                pipeline_id=pipeline_id,
                ray_namespace=namespace,
                cluster_tp_configs={"actor_train": 2, "actor_infer": 2},
                cluster_device_mappings={
                    "actor_train": [0, 1],
                    "actor_infer": [0, 1, 2, 3],
                },
            )
            await scheduler.admit_pipeline(pipeline_id=pipeline_id)

    asyncio.run(_register_and_admit())

    # Invariant 1: both pipelines are registered in separate namespaces.
    assert scheduler._state.pipeline_registry[pipeline_a]["namespace"] == namespace_a
    assert scheduler._state.pipeline_registry[pipeline_b]["namespace"] == namespace_b
    assert namespace_a != namespace_b

    runtime_a = _FakeMilesRuntime(pipeline_id=pipeline_a, namespace=namespace_a, initial_weight_version=3)
    runtime_b = _FakeMilesRuntime(pipeline_id=pipeline_b, namespace=namespace_b, initial_weight_version=6)
    coordinator_a = _FakeMilesCoordinator(runtime_a)
    coordinator_b = _FakeMilesCoordinator(runtime_b)
    scheduler._coordinator_handle_cache[pipeline_a] = (namespace_a, coordinator_a)
    scheduler._coordinator_handle_cache[pipeline_b] = (namespace_b, coordinator_b)

    b_infer_cluster = f"{pipeline_b}_actor_infer"
    a_train_cluster = f"{pipeline_a}_actor_train"
    scheduler._state.active_allocations[b_infer_cluster] = ClusterAllocation(
        cluster_id=b_infer_cluster,
        gpu_ids=[0, 1, 2, 3],
        priority=Priority.GENERATION,
        active_dp_ranks={0, 1},
        dp_rank_to_gpus={0: [0, 1], 1: [2, 3]},
    )
    scheduler._state.idle_gpus = set()

    pending_train = PendingRequest(
        request=Request(cluster_id=a_train_cluster, priority=Priority.ACTOR_TRAINING, timestamp=1.0),
        event=asyncio.Event(),
    )
    scheduler._state.pending_bucket(Priority.ACTOR_TRAINING).append(pending_train)

    asyncio.run(scheduler.scheduling_cycle())

    # Invariant 2: scheduler allocates A train and B infer exclusively; no GPU overlap remains.
    assert pending_train.event.is_set()
    assert pending_train.result == [0, 1]
    assert scheduler._state.active_allocations[a_train_cluster].gpu_ids == [0, 1]
    assert scheduler._state.active_allocations[b_infer_cluster].gpu_ids == [2, 3]
    _assert_exclusive_allocations(scheduler._state.active_allocations)

    # Invariant 3a: B engine 0 slept only through the scheduler resize_infer RPC path.
    assert coordinator_b.resize_calls == [{"dp_ranks_to_remove": [0], "dp_ranks_to_add": []}]
    assert ("shrink_engines", (0,)) in runtime_b.events
    assert runtime_b.active_engine_indices == {1}

    # Simulate B's latest trained/cache version while its overlap engine is asleep.
    runtime_b.cache_ready_step = 7
    runtime_b.current_weight_version = 7
    runtime_b.engine_weight_versions[1] = 7
    scheduler._state.latest_progress_by_pipeline[pipeline_b] = {
        "train": {
            "__full_finetune__": ProgressReport(
                pipeline_id=pipeline_b,
                step_target_trajectories=100,
                metrics={"completed": 0},
            )
        }
    }

    asyncio.run(scheduler.notify_release_gpus(cluster_id=a_train_cluster, global_step=7))
    asyncio.run(scheduler.scheduling_cycle())

    # Invariant 3b: B engine 0 woke through scheduler resize_infer after A released GPUs.
    assert coordinator_b.resize_calls[-1] == {"dp_ranks_to_remove": [], "dp_ranks_to_add": [0]}
    assert ("wake_partial", (0,)) in runtime_b.events
    assert runtime_b.active_engine_indices == {0, 1}
    assert scheduler._state.active_allocations[b_infer_cluster].gpu_ids == [0, 1, 2, 3]
    _assert_exclusive_allocations(scheduler._state.active_allocations)

    # Invariant 4: the woken engine is synced to B's current cache/weight version.
    assert ("sync_selected_workers", (0,)) in runtime_b.events
    assert ("set_weight_version", (0,)) in runtime_b.events
    assert runtime_b.engine_weight_versions[0] == runtime_b.current_weight_version == runtime_b.cache_ready_step
    assert all(
        runtime_b.engine_weight_versions[engine] == runtime_b.current_weight_version
        for engine in runtime_b.active_engine_indices
    )
