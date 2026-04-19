"""CPU-resident bucket cache for PP collective gather and selective weight sync.

Each "bucket" is a single named parameter shard (``param_name``, ``shard_id``).
``shard_id`` corresponds to a Pipeline-Parallel (PP) rank so that all PP ranks
can push their layer slices into the single cache owner before a broadcast sync.

Thread-safety:
    All public methods acquire ``_lock`` before mutating state.  The lock is a
    plain ``threading.Lock``; Ray actor re-entrancy is not assumed.

Typical lifecycle::

    cache = CPUBucketCache()

    # --- PP gather phase (all PP workers push to pp_rank==0 owner) ---
    for pp_rank, (name, tensor) in enumerate(model_state):
        cache.store(name, shard_id=pp_rank, tensor=tensor)

    # --- Selective sync: only push dirty buckets to infer workers ---
    dirty = cache.get_dirty_buckets()
    send(dirty)                        # transport layer
    cache.mark_synced([(b.param_name, b.shard_id) for b in dirty])

    # --- On next checkpoint, mark everything dirty again ---
    cache.mark_all_dirty()
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

try:
    import torch
    _Tensor = torch.Tensor
except ImportError:  # pragma: no cover — allow import without torch installed
    import types as _types
    _torch_stub = _types.ModuleType("torch")

    class _Tensor:  # type: ignore[no-redef]
        pass

    _torch_stub.Tensor = _Tensor  # type: ignore[attr-defined]
    torch = _torch_stub  # type: ignore[assignment]


# Public key type: (param_name, shard_id)
BucketKey = Tuple[str, int]


@dataclass
class Bucket:
    """Single cached weight shard.

    Attributes:
        param_name: Full dotted parameter name (e.g. ``"model.layers.0.weight"``).
        shard_id:   PP-rank index that owns this slice (0 for non-PP models).
        tensor:     CPU clone of the weight tensor at the time of the last
                    ``store()`` call.
        dirty:      ``True`` if this bucket has been written since the last
                    successful sync.  Reset to ``False`` by ``mark_synced()``.
    """

    param_name: str
    shard_id: int
    tensor: _Tensor
    dirty: bool = True

    def __repr__(self) -> str:  # pragma: no cover
        shape = getattr(self.tensor, "shape", "?")
        return (
            f"Bucket(param_name={self.param_name!r}, shard_id={self.shard_id}, "
            f"shape={shape}, dirty={self.dirty})"
        )


class CPUBucketCache:
    """Thread-safe CPU-memory cache for model weight buckets.

    The cache is keyed by ``(param_name, shard_id)``.  Tensors are stored as
    CPU clones so the training GPU remains free for the next forward/backward
    pass while the sync is in flight.

    Args:
        bucket_size_bytes: Reserved for future chunked-bucket support.  Currently
            unused; each ``store()`` call maps one parameter shard to one bucket.
    """

    def __init__(self, *, bucket_size_bytes: int = 256 * 1024 * 1024) -> None:
        self._bucket_size_bytes = bucket_size_bytes
        self._buckets: Dict[BucketKey, Bucket] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def store(self, param_name: str, *, shard_id: int, tensor: _Tensor) -> None:
        """Insert or overwrite the bucket for ``(param_name, shard_id)``.

        The tensor is cloned to CPU memory so the caller may immediately
        reuse or free the source buffer.  The resulting bucket is always
        marked ``dirty=True``.

        Args:
            param_name: Dotted parameter name, e.g. ``"transformer.h.0.weight"``.
            shard_id:   PP rank index (use ``0`` for non-PP models).
            tensor:     Source tensor (any device).  A CPU clone is stored.
        """
        cpu_tensor = tensor.cpu().clone()
        key: BucketKey = (param_name, shard_id)
        with self._lock:
            self._buckets[key] = Bucket(
                param_name=param_name,
                shard_id=shard_id,
                tensor=cpu_tensor,
                dirty=True,
            )

    def mark_synced(self, keys: List[BucketKey]) -> None:
        """Mark the given buckets as clean (successfully synced to infer workers).

        Keys that are not present in the cache are silently ignored.

        Args:
            keys: Sequence of ``(param_name, shard_id)`` tuples to clear.
        """
        with self._lock:
            for key in keys:
                bucket = self._buckets.get(key)
                if bucket is not None:
                    bucket.dirty = False

    def mark_all_dirty(self) -> None:
        """Mark every bucket dirty (e.g. after a new training checkpoint is loaded)."""
        with self._lock:
            for bucket in self._buckets.values():
                bucket.dirty = True

    def mark_all_synced(self) -> None:
        """Mark every bucket clean (bulk sync completed)."""
        with self._lock:
            for bucket in self._buckets.values():
                bucket.dirty = False

    def evict(self, param_name: str, *, shard_id: int) -> None:
        """Remove a single bucket.  No-op if the key is not present."""
        key: BucketKey = (param_name, shard_id)
        with self._lock:
            self._buckets.pop(key, None)

    def evict_param(self, param_name: str) -> None:
        """Remove all shards of *param_name* from the cache."""
        with self._lock:
            keys_to_remove = [k for k in self._buckets if k[0] == param_name]
            for k in keys_to_remove:
                del self._buckets[k]

    def clear(self) -> None:
        """Remove all buckets from the cache."""
        with self._lock:
            self._buckets.clear()

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get_dirty_buckets(self) -> List[Bucket]:
        """Return a snapshot list of all dirty buckets.

        The returned list is a snapshot; subsequent ``store()`` or
        ``mark_synced()`` calls do not affect already-returned ``Bucket``
        objects.
        """
        with self._lock:
            return [b for b in self._buckets.values() if b.dirty]

    def get_all_buckets(self) -> Dict[BucketKey, Bucket]:
        """Return a shallow copy of the full bucket map (dirty and clean)."""
        with self._lock:
            return dict(self._buckets)

    def size(self) -> int:
        """Return the total number of buckets currently held."""
        with self._lock:
            return len(self._buckets)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        with self._lock:
            dirty = sum(1 for b in self._buckets.values() if b.dirty)
            return f"CPUBucketCache(total={len(self._buckets)}, dirty={dirty})"
