"""Unit tests for CPUBucketCache — CPU-resident bucket cache for PP gather + selective sync.

Tests are fully self-contained: no Ray, no ROLL, no CUDA required.
The module under test only depends on the stdlib and (optionally) torch,
which is stubbed if unavailable.
"""
from __future__ import annotations

import sys
import threading
import types
from typing import Any
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Torch stub — allows tests to run without a GPU environment
# ---------------------------------------------------------------------------


def _make_torch_stub() -> types.ModuleType:
    torch_stub = types.ModuleType("torch")

    class _Tensor:
        def __init__(self, data: list | Any, *, dtype=None):
            self._data = list(data) if not isinstance(data, _Tensor) else data._data
            self.dtype = dtype

        def cpu(self):
            return self

        def clone(self):
            return _Tensor(self._data[:], dtype=self.dtype)

        def __eq__(self, other):  # type: ignore[override]
            if isinstance(other, _Tensor):
                return self._data == other._data
            return NotImplemented

        def __repr__(self):
            return f"_Tensor({self._data})"

    torch_stub.Tensor = _Tensor  # type: ignore[attr-defined]

    def _tensor(data, *, dtype=None):
        return _Tensor(data, dtype=dtype)

    torch_stub.tensor = _tensor  # type: ignore[attr-defined]
    return torch_stub


# ---------------------------------------------------------------------------
# Import helper — loads CPUBucketCache with stubbed deps
# ---------------------------------------------------------------------------

_BUCKET_CACHE_MODULE = "rlix.pipeline.bucket_cache"


def _load_bucket_cache(monkeypatch: pytest.MonkeyPatch):
    """Load rlix.pipeline.bucket_cache with all heavy deps stubbed."""
    # Remove prior imports so each test gets a fresh module state.
    for key in list(sys.modules):
        if key.startswith("rlix"):
            monkeypatch.delitem(sys.modules, key, raising=False)

    if "torch" not in sys.modules:
        monkeypatch.setitem(sys.modules, "torch", _make_torch_stub())

    import importlib
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    rlix_root = repo_root / "rlix"

    # Minimal package stubs so importlib can resolve rlix.pipeline.bucket_cache
    for pkg in ("rlix", "rlix.pipeline"):
        mod = types.ModuleType(pkg)
        mod.__path__ = [str(rlix_root / pkg.replace("rlix.", "").replace(".", "/"))]  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, pkg, mod)

    import sys as _sys
    _sys.path.insert(0, str(repo_root))

    return importlib.import_module(_BUCKET_CACHE_MODULE)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mod(monkeypatch):
    return _load_bucket_cache(monkeypatch)


@pytest.fixture()
def cache(mod):
    return mod.CPUBucketCache()


@pytest.fixture()
def tensor(mod):
    """Return a factory for test tensors."""
    import sys as _sys
    torch = _sys.modules["torch"]

    def _make(data):
        return torch.tensor(data)

    return _make


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_new_cache_is_empty(cache):
    assert cache.size() == 0


def test_get_all_buckets_empty(cache):
    assert cache.get_all_buckets() == {}


def test_get_dirty_buckets_empty(cache):
    assert cache.get_dirty_buckets() == []


# ---------------------------------------------------------------------------
# store()
# ---------------------------------------------------------------------------


def test_store_single_bucket(cache, tensor):
    t = tensor([1.0, 2.0])
    cache.store("weight.A", shard_id=0, tensor=t)
    assert cache.size() == 1


def test_store_marks_dirty_by_default(cache, tensor):
    cache.store("weight.A", shard_id=0, tensor=tensor([1.0]))
    dirty = cache.get_dirty_buckets()
    assert len(dirty) == 1
    assert dirty[0].param_name == "weight.A"
    assert dirty[0].shard_id == 0
    assert dirty[0].dirty is True


def test_store_multiple_shards(cache, tensor):
    """PP gather: multiple shard_ids for the same param_name are stored independently."""
    cache.store("layer.weight", shard_id=0, tensor=tensor([1.0]))
    cache.store("layer.weight", shard_id=1, tensor=tensor([2.0]))
    cache.store("layer.weight", shard_id=2, tensor=tensor([3.0]))
    assert cache.size() == 3
    dirty = cache.get_dirty_buckets()
    assert len(dirty) == 3


def test_store_overwrites_existing(cache, tensor):
    t1 = tensor([1.0])
    t2 = tensor([99.0])
    cache.store("w", shard_id=0, tensor=t1)
    cache.store("w", shard_id=0, tensor=t2)
    # Size unchanged (overwrite, not append)
    assert cache.size() == 1
    b = cache.get_all_buckets()[("w", 0)]
    assert b.tensor == t2


def test_store_clones_tensor(cache, tensor, mod):
    """Stored tensor must be a CPU clone independent of the original."""
    t = tensor([5.0, 6.0])
    cache.store("w", shard_id=0, tensor=t)
    b = cache.get_all_buckets()[("w", 0)]
    # The stored tensor must be a distinct object.
    assert b.tensor is not t


def test_store_different_params(cache, tensor):
    cache.store("a.weight", shard_id=0, tensor=tensor([1.0]))
    cache.store("b.weight", shard_id=0, tensor=tensor([2.0]))
    assert cache.size() == 2
    keys = set(cache.get_all_buckets().keys())
    assert keys == {("a.weight", 0), ("b.weight", 0)}


# ---------------------------------------------------------------------------
# mark_synced()
# ---------------------------------------------------------------------------


def test_mark_synced_clears_dirty(cache, tensor):
    cache.store("w", shard_id=0, tensor=tensor([1.0]))
    cache.mark_synced([("w", 0)])
    assert cache.get_dirty_buckets() == []


def test_mark_synced_partial(cache, tensor):
    """mark_synced on a subset leaves other buckets dirty."""
    cache.store("a", shard_id=0, tensor=tensor([1.0]))
    cache.store("b", shard_id=0, tensor=tensor([2.0]))
    cache.mark_synced([("a", 0)])
    dirty = cache.get_dirty_buckets()
    assert len(dirty) == 1
    assert dirty[0].param_name == "b"


def test_mark_synced_missing_key_is_noop(cache, tensor):
    """Calling mark_synced with a key not in cache must not raise."""
    cache.store("w", shard_id=0, tensor=tensor([1.0]))
    cache.mark_synced([("nonexistent", 99)])  # must not raise
    assert len(cache.get_dirty_buckets()) == 1


def test_store_after_sync_marks_dirty_again(cache, tensor):
    cache.store("w", shard_id=0, tensor=tensor([1.0]))
    cache.mark_synced([("w", 0)])
    cache.store("w", shard_id=0, tensor=tensor([2.0]))
    dirty = cache.get_dirty_buckets()
    assert len(dirty) == 1
    assert dirty[0].dirty is True


# ---------------------------------------------------------------------------
# mark_all_dirty() / mark_all_synced()
# ---------------------------------------------------------------------------


def test_mark_all_dirty_resets_clean_buckets(cache, tensor):
    cache.store("a", shard_id=0, tensor=tensor([1.0]))
    cache.store("b", shard_id=0, tensor=tensor([2.0]))
    cache.mark_synced([("a", 0), ("b", 0)])
    assert cache.get_dirty_buckets() == []
    cache.mark_all_dirty()
    assert len(cache.get_dirty_buckets()) == 2


def test_mark_all_synced_clears_all(cache, tensor):
    cache.store("a", shard_id=0, tensor=tensor([1.0]))
    cache.store("b", shard_id=0, tensor=tensor([2.0]))
    cache.mark_all_synced()
    assert cache.get_dirty_buckets() == []


# ---------------------------------------------------------------------------
# evict()
# ---------------------------------------------------------------------------


def test_evict_removes_bucket(cache, tensor):
    cache.store("w", shard_id=0, tensor=tensor([1.0]))
    cache.evict("w", shard_id=0)
    assert cache.size() == 0
    assert ("w", 0) not in cache.get_all_buckets()


def test_evict_missing_key_is_noop(cache):
    cache.evict("nonexistent", shard_id=0)  # must not raise


def test_evict_param_removes_all_shards(cache, tensor):
    """evict_param() removes every shard of a given param_name."""
    for i in range(4):
        cache.store("layer.w", shard_id=i, tensor=tensor([float(i)]))
    assert cache.size() == 4
    cache.evict_param("layer.w")
    assert cache.size() == 0


# ---------------------------------------------------------------------------
# clear()
# ---------------------------------------------------------------------------


def test_clear_empties_cache(cache, tensor):
    cache.store("w", shard_id=0, tensor=tensor([1.0]))
    cache.store("x", shard_id=0, tensor=tensor([2.0]))
    cache.clear()
    assert cache.size() == 0
    assert cache.get_all_buckets() == {}
    assert cache.get_dirty_buckets() == []


# ---------------------------------------------------------------------------
# Thread-safety
# ---------------------------------------------------------------------------


def test_concurrent_stores_are_safe(cache, tensor):
    """Multiple threads writing distinct keys must not corrupt the cache."""
    n_threads = 8
    n_params_per_thread = 50
    errors: list[Exception] = []

    def _writer(thread_id: int):
        try:
            for i in range(n_params_per_thread):
                cache.store(f"thread{thread_id}.w{i}", shard_id=0, tensor=tensor([float(i)]))
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=_writer, args=(t,)) for t in range(n_threads)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    assert errors == [], f"Thread errors: {errors}"
    assert cache.size() == n_threads * n_params_per_thread


def test_concurrent_store_and_mark_synced(cache, tensor):
    """Store + mark_synced concurrently must not raise or lose data."""
    cache.store("w", shard_id=0, tensor=tensor([1.0]))
    errors: list[Exception] = []

    def _syncer():
        try:
            for _ in range(100):
                cache.mark_synced([("w", 0)])
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    def _storer():
        try:
            for i in range(100):
                cache.store("w", shard_id=0, tensor=tensor([float(i)]))
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    t1 = threading.Thread(target=_syncer)
    t2 = threading.Thread(target=_storer)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert errors == []


# ---------------------------------------------------------------------------
# Bucket dataclass properties
# ---------------------------------------------------------------------------


def test_bucket_repr_is_informative(cache, tensor):
    cache.store("layer.0.weight", shard_id=2, tensor=tensor([1.0]))
    b = cache.get_all_buckets()[("layer.0.weight", 2)]
    r = repr(b)
    assert "layer.0.weight" in r
    assert "2" in r
