"""Unit tests for CPUBucketCache integration in ModelUpdateService.

These tests verify the new cache-aware layer of ModelUpdateService:
- ModelUpdateService.populate_cache_from_workers(): calls each PP rank worker to
  extract and push weights into the owner's CPUBucketCache.
- ModelUpdateService.sync_from_cache(): reads dirty buckets from the cache and
  dispatches a BucketUpdateRequest to each target infer worker.
- Dirty-tracking round-trip: after sync, buckets are marked clean; after
  mark_all_dirty(), they become eligible again.

All Ray remote actors are replaced with synchronous fakes, so no GPU or Ray
cluster is required.
"""
from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, call, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


def _make_torch_stub() -> types.ModuleType:
    torch_stub = types.ModuleType("torch")

    class _Tensor:
        def __init__(self, data):
            self._data = list(data) if not isinstance(data, _Tensor) else data._data

        def cpu(self):
            return self

        def clone(self):
            return _Tensor(self._data[:])

        def copy_(self, other):
            self._data = other._data[:]
            return self

        def __eq__(self, other):
            if isinstance(other, _Tensor):
                return self._data == other._data
            return NotImplemented

        def __repr__(self):
            return f"_Tensor({self._data})"

    torch_stub.Tensor = _Tensor  # type: ignore[attr-defined]
    torch_stub.tensor = lambda data: _Tensor(data)  # type: ignore[attr-defined]
    torch_stub.cat = lambda ts, dim=0: _Tensor([x for t in ts for x in t._data])  # type: ignore[attr-defined]
    return torch_stub


def _install_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear rlix modules and install lightweight stubs for Ray, ROLL, torch."""
    for key in list(sys.modules):
        if key.startswith("rlix") or key == "ray":
            monkeypatch.delitem(sys.modules, key, raising=False)

    # torch
    monkeypatch.setitem(sys.modules, "torch", _make_torch_stub())

    # ray stub — bare minimum
    ray_stub = types.ModuleType("ray")
    ray_stub.remote = lambda *a, **kw: (lambda cls: cls)  # decorator no-op
    ray_stub.get = lambda refs, **kw: [r() if callable(r) else r for r in (refs if isinstance(refs, list) else [refs])]
    ray_stub.get_actor = MagicMock(return_value=MagicMock())
    monkeypatch.setitem(sys.modules, "ray", ray_stub)

    # ROLL stubs
    for mod_name in (
        "roll",
        "roll.distributed",
        "roll.distributed.executor",
        "roll.distributed.executor.cluster",
        "roll.utils",
        "roll.utils.constants",
        "roll.utils.logging",
    ):
        m = types.ModuleType(mod_name)
        monkeypatch.setitem(sys.modules, mod_name, m)

    constants_mod = sys.modules["roll.utils.constants"]
    constants_mod.GLOBAL_STORAGE_NAMESPACE = "global"  # type: ignore[attr-defined]
    constants_mod.STORAGE_NAME = "storage"  # type: ignore[attr-defined]

    logging_mod = sys.modules["roll.utils.logging"]
    logging_mod.get_logger = lambda: MagicMock()  # type: ignore[attr-defined]

    # rlix package stubs
    rlix_root = REPO_ROOT / "rlix"
    for pkg in ("rlix", "rlix.pipeline", "rlix.utils"):
        mod = types.ModuleType(pkg)
        mod.__path__ = [str(rlix_root / pkg.replace("rlix.", "").replace(".", "/"))]  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, pkg, mod)

    # rlix.utils.env stub
    env_stub = types.ModuleType("rlix.utils.env")
    env_stub.parse_env_timeout_s = lambda *a, **kw: 150.0  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "rlix.utils.env", env_stub)

    sys.path.insert(0, str(REPO_ROOT))


def _load_modules(monkeypatch: pytest.MonkeyPatch):
    import importlib

    _install_stubs(monkeypatch)
    bucket_cache = importlib.import_module("rlix.pipeline.bucket_cache")
    bucket_receiver = importlib.import_module("rlix.pipeline.bucket_receiver")
    mus = importlib.import_module("rlix.pipeline.model_update_service_cached")
    return bucket_cache, bucket_receiver, mus


# ---------------------------------------------------------------------------
# Fake worker/cluster helpers
# ---------------------------------------------------------------------------


class _FakeWorker:
    """Minimal synchronous fake for a ROLL/vLLM worker remote actor."""

    def __init__(self, rank: int, pp_rank: int, dp_rank: int, tp_rank: int, cp_rank: int = 0):
        self.rank = rank
        self.pp_rank = pp_rank
        self.dp_rank = dp_rank
        self.tp_rank = tp_rank
        self.cp_rank = cp_rank
        # Simulated model weights for this PP shard
        self.weights: Dict[str, Any] = {}
        self.received_requests: List[Any] = []

    def get_pp_weight_shards(self) -> Dict[str, Any]:
        """Return this worker's PP layer weights (simulates remote call)."""
        return dict(self.weights)

    def receive_weight_update(self, request: Any) -> Any:
        """Accept a BucketUpdateRequest (simulates infer worker)."""
        self.received_requests.append(request)
        return MagicMock(ok=True, applied=len(request.buckets), failed=0, errors=[])


@dataclass
class _FakeWorkerRankInfo:
    pp_rank: int
    dp_rank: int
    tp_rank: int
    cp_rank: int = 0


def _make_cluster(workers: List[_FakeWorker]) -> MagicMock:
    cluster = MagicMock()
    cluster.workers = workers
    cluster.rank2worker = {w.rank: w for w in workers}
    cluster.world_size = len(workers)
    cluster.worker_rank_info = [
        _FakeWorkerRankInfo(pp_rank=w.pp_rank, dp_rank=w.dp_rank, tp_rank=w.tp_rank, cp_rank=w.cp_rank)
        for w in workers
    ]
    return cluster


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mods(monkeypatch):
    return _load_modules(monkeypatch)


@pytest.fixture()
def tensor():
    torch = sys.modules["torch"]
    return lambda data: torch.tensor(data)


# ---------------------------------------------------------------------------
# ModelUpdateServiceCached construction
# ---------------------------------------------------------------------------


def test_construction_creates_cache(mods):
    bc, br, mus = mods
    src_cluster = _make_cluster([
        _FakeWorker(0, pp_rank=0, dp_rank=0, tp_rank=0),
    ])
    tgt_cluster = _make_cluster([
        _FakeWorker(0, pp_rank=0, dp_rank=0, tp_rank=0),
    ])
    svc = mus.ModelUpdateServiceCached(
        pipeline_id="test-pipeline",
        src_cluster=src_cluster,
        tgt_cluster=tgt_cluster,
    )
    assert svc.cache is not None
    assert svc.cache.size() == 0


# ---------------------------------------------------------------------------
# populate_cache_from_workers()
# ---------------------------------------------------------------------------


def test_populate_cache_single_pp_rank(mods, tensor):
    bc, br, mus = mods
    worker = _FakeWorker(0, pp_rank=0, dp_rank=0, tp_rank=0)
    worker.weights = {
        "layer.0.weight": tensor([1.0, 2.0]),
        "layer.0.bias": tensor([0.1]),
    }
    src_cluster = _make_cluster([worker])
    tgt_cluster = _make_cluster([_FakeWorker(0, pp_rank=0, dp_rank=0, tp_rank=0)])
    svc = mus.ModelUpdateServiceCached(
        pipeline_id="pipe-a",
        src_cluster=src_cluster,
        tgt_cluster=tgt_cluster,
    )
    svc.populate_cache_from_workers()
    # All params should now be in cache, all shards from shard_id=0
    assert svc.cache.size() == 2
    all_buckets = svc.cache.get_all_buckets()
    assert ("layer.0.weight", 0) in all_buckets
    assert ("layer.0.bias", 0) in all_buckets


def test_populate_cache_multi_pp_ranks(mods, tensor):
    """PP gather: 2 PP ranks → each param gets 2 shards in the cache."""
    bc, br, mus = mods
    w0 = _FakeWorker(0, pp_rank=0, dp_rank=0, tp_rank=0)
    w0.weights = {"layers.0.weight": tensor([1.0, 2.0])}
    w1 = _FakeWorker(1, pp_rank=1, dp_rank=0, tp_rank=0)
    w1.weights = {"layers.1.weight": tensor([3.0, 4.0])}
    src_cluster = _make_cluster([w0, w1])
    tgt_cluster = _make_cluster([_FakeWorker(0, pp_rank=0, dp_rank=0, tp_rank=0)])
    svc = mus.ModelUpdateServiceCached(
        pipeline_id="pipe-b",
        src_cluster=src_cluster,
        tgt_cluster=tgt_cluster,
    )
    svc.populate_cache_from_workers()
    # 2 params, 1 shard each (different param names from different PP ranks)
    assert svc.cache.size() == 2
    keys = set(svc.cache.get_all_buckets().keys())
    assert ("layers.0.weight", 0) in keys
    assert ("layers.1.weight", 1) in keys


def test_populate_marks_all_dirty(mods, tensor):
    bc, br, mus = mods
    w0 = _FakeWorker(0, pp_rank=0, dp_rank=0, tp_rank=0)
    w0.weights = {"w": tensor([1.0])}
    src_cluster = _make_cluster([w0])
    tgt_cluster = _make_cluster([_FakeWorker(0, pp_rank=0, dp_rank=0, tp_rank=0)])
    svc = mus.ModelUpdateServiceCached(
        pipeline_id="pipe-c",
        src_cluster=src_cluster,
        tgt_cluster=tgt_cluster,
    )
    svc.populate_cache_from_workers()
    assert len(svc.cache.get_dirty_buckets()) == 1


# ---------------------------------------------------------------------------
# sync_from_cache()
# ---------------------------------------------------------------------------


def test_sync_from_cache_dispatches_to_tgt_workers(mods, tensor):
    bc, br, mus = mods
    w0 = _FakeWorker(0, pp_rank=0, dp_rank=0, tp_rank=0)
    w0.weights = {"weight": tensor([1.0])}
    src_cluster = _make_cluster([w0])

    tgt_w0 = _FakeWorker(0, pp_rank=0, dp_rank=0, tp_rank=0)
    tgt_cluster = _make_cluster([tgt_w0])

    svc = mus.ModelUpdateServiceCached(
        pipeline_id="pipe-d",
        src_cluster=src_cluster,
        tgt_cluster=tgt_cluster,
    )
    svc.populate_cache_from_workers()
    svc.sync_from_cache(tgt_dp_ranks=[0])

    # Infer worker must have received exactly one request
    assert len(tgt_w0.received_requests) == 1
    req = tgt_w0.received_requests[0]
    assert len(req.buckets) == 1
    assert req.buckets[0].param_name == "weight"


def test_sync_from_cache_marks_buckets_clean(mods, tensor):
    bc, br, mus = mods
    w0 = _FakeWorker(0, pp_rank=0, dp_rank=0, tp_rank=0)
    w0.weights = {"a": tensor([1.0]), "b": tensor([2.0])}
    src_cluster = _make_cluster([w0])
    tgt_w0 = _FakeWorker(0, pp_rank=0, dp_rank=0, tp_rank=0)
    tgt_cluster = _make_cluster([tgt_w0])
    svc = mus.ModelUpdateServiceCached(
        pipeline_id="pipe-e",
        src_cluster=src_cluster,
        tgt_cluster=tgt_cluster,
    )
    svc.populate_cache_from_workers()
    assert len(svc.cache.get_dirty_buckets()) == 2
    svc.sync_from_cache(tgt_dp_ranks=[0])
    assert len(svc.cache.get_dirty_buckets()) == 0


def test_sync_from_cache_skips_clean_buckets(mods, tensor):
    """After sync, re-sync without repopulate must not send any buckets."""
    bc, br, mus = mods
    w0 = _FakeWorker(0, pp_rank=0, dp_rank=0, tp_rank=0)
    w0.weights = {"w": tensor([1.0])}
    src_cluster = _make_cluster([w0])
    tgt_w0 = _FakeWorker(0, pp_rank=0, dp_rank=0, tp_rank=0)
    tgt_cluster = _make_cluster([tgt_w0])
    svc = mus.ModelUpdateServiceCached(
        pipeline_id="pipe-f",
        src_cluster=src_cluster,
        tgt_cluster=tgt_cluster,
    )
    svc.populate_cache_from_workers()
    svc.sync_from_cache(tgt_dp_ranks=[0])
    # Second sync without new populate: no dirty buckets → no dispatch
    svc.sync_from_cache(tgt_dp_ranks=[0])
    assert len(tgt_w0.received_requests) == 1  # only first sync dispatched


def test_sync_after_mark_all_dirty_sends_again(mods, tensor):
    """mark_all_dirty() then sync_from_cache() must re-send all buckets."""
    bc, br, mus = mods
    w0 = _FakeWorker(0, pp_rank=0, dp_rank=0, tp_rank=0)
    w0.weights = {"w": tensor([1.0])}
    src_cluster = _make_cluster([w0])
    tgt_w0 = _FakeWorker(0, pp_rank=0, dp_rank=0, tp_rank=0)
    tgt_cluster = _make_cluster([tgt_w0])
    svc = mus.ModelUpdateServiceCached(
        pipeline_id="pipe-g",
        src_cluster=src_cluster,
        tgt_cluster=tgt_cluster,
    )
    svc.populate_cache_from_workers()
    svc.sync_from_cache(tgt_dp_ranks=[0])
    svc.cache.mark_all_dirty()
    svc.sync_from_cache(tgt_dp_ranks=[0])
    assert len(tgt_w0.received_requests) == 2


def test_sync_with_empty_cache_no_dispatch(mods):
    bc, br, mus = mods
    src_cluster = _make_cluster([_FakeWorker(0, pp_rank=0, dp_rank=0, tp_rank=0)])
    tgt_w0 = _FakeWorker(0, pp_rank=0, dp_rank=0, tp_rank=0)
    tgt_cluster = _make_cluster([tgt_w0])
    svc = mus.ModelUpdateServiceCached(
        pipeline_id="pipe-h",
        src_cluster=src_cluster,
        tgt_cluster=tgt_cluster,
    )
    # Don't call populate; sync_from_cache with empty cache → nothing sent
    svc.sync_from_cache(tgt_dp_ranks=[0])
    assert tgt_w0.received_requests == []
