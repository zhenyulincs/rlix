"""Unit tests for the bucket receiver API on vLLM infer workers.

Tests cover:
- apply_bucket_update(): apply a list of Bucket objects to a model state dict
- merge_buckets(): reassemble PP-sharded buckets into a full parameter tensor
- BucketUpdateRequest / BucketUpdateResult dataclasses
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]


def _make_torch_stub() -> types.ModuleType:
    torch_stub = types.ModuleType("torch")

    class _Tensor:
        def __init__(self, data):
            self._data = list(data)

        def cpu(self):
            return self

        def clone(self):
            return _Tensor(self._data[:])

        def copy_(self, other: "_Tensor") -> "_Tensor":
            self._data = other._data[:]
            return self

        def __eq__(self, other):  # type: ignore[override]
            if isinstance(other, _Tensor):
                return self._data == other._data
            return NotImplemented

        def __repr__(self):
            return f"_Tensor({self._data})"

    torch_stub.Tensor = _Tensor  # type: ignore[attr-defined]
    torch_stub.tensor = lambda data: _Tensor(data)  # type: ignore[attr-defined]

    def _cat(tensors, dim=0):
        combined = []
        for t in tensors:
            combined.extend(t._data)
        return _Tensor(combined)

    torch_stub.cat = _cat  # type: ignore[attr-defined]
    return torch_stub


def _make_bucket_stub(torch_mod) -> types.ModuleType:
    """Return a minimal stub for rlix.pipeline.bucket_cache."""
    stub = types.ModuleType("rlix.pipeline.bucket_cache")

    from dataclasses import dataclass

    @dataclass
    class Bucket:
        param_name: str
        shard_id: int
        tensor: object
        dirty: bool = True

    stub.Bucket = Bucket  # type: ignore[attr-defined]
    stub.BucketKey = object  # type: ignore[attr-defined]
    return stub


def _load_receiver(monkeypatch: pytest.MonkeyPatch):
    import importlib

    for key in list(sys.modules):
        if key.startswith("rlix"):
            monkeypatch.delitem(sys.modules, key, raising=False)

    torch_mod = _make_torch_stub()
    monkeypatch.setitem(sys.modules, "torch", torch_mod)

    bucket_stub = _make_bucket_stub(torch_mod)
    monkeypatch.setitem(sys.modules, "rlix.pipeline.bucket_cache", bucket_stub)

    for pkg in ("rlix", "rlix.pipeline"):
        mod = types.ModuleType(pkg)
        path_suffix = pkg.replace("rlix.", "").replace(".", "/") if pkg != "rlix" else ""
        mod.__path__ = [str(REPO_ROOT / "rlix" / path_suffix)]  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, pkg, mod)

    sys.path.insert(0, str(REPO_ROOT))
    return importlib.import_module("rlix.pipeline.bucket_receiver")


@pytest.fixture()
def mod(monkeypatch):
    return _load_receiver(monkeypatch)


@pytest.fixture()
def Bucket(mod):
    # Use the real Bucket from bucket_cache if available, else get from stub
    import sys as _sys
    return _sys.modules["rlix.pipeline.bucket_cache"].Bucket


@pytest.fixture()
def tensor():
    torch = sys.modules["torch"]

    def _make(data):
        return torch.tensor(data)

    return _make


# ---------------------------------------------------------------------------
# BucketUpdateRequest / BucketUpdateResult
# ---------------------------------------------------------------------------


def test_request_dataclass(mod, Bucket, tensor):
    req = mod.BucketUpdateRequest(
        sync_id="sync_001",
        buckets=[Bucket("w", 0, tensor([1.0]))],
    )
    assert req.sync_id == "sync_001"
    assert len(req.buckets) == 1


def test_result_dataclass_ok(mod):
    res = mod.BucketUpdateResult(sync_id="sync_001", applied=3, failed=0, errors=[])
    assert res.ok is True


def test_result_dataclass_partial_failure(mod):
    res = mod.BucketUpdateResult(
        sync_id="sync_001", applied=2, failed=1, errors=["param X not found"]
    )
    assert res.ok is False


# ---------------------------------------------------------------------------
# merge_pp_shards()
# ---------------------------------------------------------------------------


def test_merge_single_shard(mod, Bucket, tensor):
    """Single shard (non-PP) is returned as-is."""
    b = Bucket("w", 0, tensor([1.0, 2.0, 3.0]))
    result = mod.merge_pp_shards([b])
    assert result == tensor([1.0, 2.0, 3.0])


def test_merge_requires_contiguous_shards(mod, Bucket, tensor):
    """merge_pp_shards must raise if shard_ids are not 0..N-1."""
    buckets = [
        Bucket("w", 0, tensor([1.0])),
        Bucket("w", 2, tensor([3.0])),  # gap: shard_id 1 missing
    ]
    with pytest.raises(ValueError, match="shard_id"):
        mod.merge_pp_shards(buckets)


def test_merge_empty_raises(mod):
    with pytest.raises(ValueError, match="empty"):
        mod.merge_pp_shards([])


# ---------------------------------------------------------------------------
# apply_bucket_update() — happy path
# ---------------------------------------------------------------------------


def test_apply_updates_existing_param(mod, Bucket, tensor):
    state_dict = {"weight": tensor([0.0, 0.0, 0.0])}
    buckets = [Bucket("weight", 0, tensor([1.0, 2.0, 3.0]))]
    req = mod.BucketUpdateRequest(sync_id="s1", buckets=buckets)
    result = mod.apply_bucket_update(state_dict, req)
    assert result.applied == 1
    assert result.failed == 0
    assert state_dict["weight"] == tensor([1.0, 2.0, 3.0])


def test_apply_missing_param_is_skipped(mod, Bucket, tensor):
    state_dict = {"weight": tensor([1.0])}
    buckets = [Bucket("nonexistent", 0, tensor([9.0]))]
    req = mod.BucketUpdateRequest(sync_id="s1", buckets=buckets)
    result = mod.apply_bucket_update(state_dict, req)
    assert result.failed == 1
    assert len(result.errors) == 1
    assert result.ok is False


def test_apply_multiple_buckets(mod, Bucket, tensor):
    state_dict = {
        "a": tensor([0.0]),
        "b": tensor([0.0]),
        "c": tensor([0.0]),
    }
    buckets = [
        Bucket("a", 0, tensor([1.0])),
        Bucket("b", 0, tensor([2.0])),
        Bucket("c", 0, tensor([3.0])),
    ]
    req = mod.BucketUpdateRequest(sync_id="s1", buckets=buckets)
    result = mod.apply_bucket_update(state_dict, req)
    assert result.applied == 3
    assert result.failed == 0
    assert result.ok is True


def test_apply_partial_success(mod, Bucket, tensor):
    state_dict = {"a": tensor([0.0])}
    buckets = [
        Bucket("a", 0, tensor([1.0])),
        Bucket("missing", 0, tensor([2.0])),
    ]
    req = mod.BucketUpdateRequest(sync_id="s1", buckets=buckets)
    result = mod.apply_bucket_update(state_dict, req)
    assert result.applied == 1
    assert result.failed == 1
    assert result.ok is False


def test_apply_empty_buckets(mod, tensor):
    state_dict = {"w": tensor([1.0])}
    req = mod.BucketUpdateRequest(sync_id="s1", buckets=[])
    result = mod.apply_bucket_update(state_dict, req)
    assert result.applied == 0
    assert result.failed == 0
    assert result.ok is True


# ---------------------------------------------------------------------------
# apply_bucket_update() — PP shards (multi-shard reassembly)
# ---------------------------------------------------------------------------


def test_apply_pp_shards_reassembled(mod, Bucket, tensor):
    """Multiple shards for the same param_name are merged before apply."""
    # Simulate a PP model where "weight" is split across 2 PP ranks.
    # After merge, weight = [1.0, 2.0] (shard_0) + [3.0, 4.0] (shard_1).
    state_dict = {"weight": tensor([0.0, 0.0, 0.0, 0.0])}
    buckets = [
        Bucket("weight", 0, tensor([1.0, 2.0])),
        Bucket("weight", 1, tensor([3.0, 4.0])),
    ]
    req = mod.BucketUpdateRequest(sync_id="s1", buckets=buckets)
    result = mod.apply_bucket_update(state_dict, req)
    assert result.applied == 1  # 1 logical param (merged from 2 shards)
    assert result.failed == 0
