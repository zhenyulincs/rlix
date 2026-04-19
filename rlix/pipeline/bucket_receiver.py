"""Receiver-side API for applying bucketed weight updates to a vLLM infer worker.

This module implements the F6-transport receiver interface:
- ``BucketUpdateRequest``: carries a batch of ``Bucket`` objects to apply.
- ``BucketUpdateResult``: reports how many buckets were applied vs. failed.
- ``merge_pp_shards()``: reassembles PP-sharded buckets into a single tensor.
- ``apply_bucket_update()``: applies a ``BucketUpdateRequest`` to a model state dict.

The functions in this module are **pure** (no Ray, no CUDA) so they can be
called from a vLLM InferWorker Ray actor or tested in isolation.

Typical usage inside a vLLM worker::

    from rlix.pipeline.bucket_receiver import apply_bucket_update, BucketUpdateRequest

    def receive_weight_update(self, request: BucketUpdateRequest) -> BucketUpdateResult:
        state_dict = self.llm_engine.model_executor.driver_worker.model_runner.model.state_dict()
        result = apply_bucket_update(state_dict, request)
        if not result.ok:
            logger.warning("Partial weight update: %s", result.errors)
        return result
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import groupby
from typing import Any, Dict, List

try:
    import torch
    _Tensor = torch.Tensor
except ImportError:  # pragma: no cover
    import types as _types

    class _Tensor:  # type: ignore[no-redef]
        pass

from rlix.pipeline.bucket_cache import Bucket


# ---------------------------------------------------------------------------
# Request / result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class BucketUpdateRequest:
    """Payload sent from the training side to a vLLM infer worker.

    Attributes:
        sync_id: Unique identifier for this sync operation (for logging / idempotency).
        buckets: Ordered list of weight buckets to apply.  Buckets for the same
            ``param_name`` with different ``shard_id`` values will be merged by
            ``apply_bucket_update()`` before writing to the state dict.
    """

    sync_id: str
    buckets: List[Bucket]


@dataclass
class BucketUpdateResult:
    """Result returned after applying a ``BucketUpdateRequest``.

    Attributes:
        sync_id: Echo of the request ``sync_id``.
        applied: Number of logical parameters successfully written (after PP merge).
        failed: Number of logical parameters that could not be applied.
        errors: Human-readable error messages for each failure.
    """

    sync_id: str
    applied: int
    failed: int
    errors: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True if every bucket was applied without error."""
        return self.failed == 0


# ---------------------------------------------------------------------------
# PP shard merge
# ---------------------------------------------------------------------------


def merge_pp_shards(buckets: List[Bucket]) -> Any:
    """Concatenate PP-sharded tensors in shard_id order.

    All buckets must share the same ``param_name``.  ``shard_id`` values must
    form a contiguous range ``0, 1, ..., N-1`` (no gaps, no duplicates).

    Args:
        buckets: One or more ``Bucket`` objects for a single parameter.

    Returns:
        A single tensor formed by concatenating the shard tensors along dim 0.

    Raises:
        ValueError: If *buckets* is empty or shard_ids are non-contiguous.
    """
    if not buckets:
        raise ValueError("merge_pp_shards: buckets must not be empty")

    sorted_buckets = sorted(buckets, key=lambda b: b.shard_id)
    expected_ids = list(range(len(sorted_buckets)))
    actual_ids = [b.shard_id for b in sorted_buckets]
    if actual_ids != expected_ids:
        raise ValueError(
            f"merge_pp_shards: shard_id values must be contiguous 0..N-1, "
            f"got {actual_ids} for param_name={sorted_buckets[0].param_name!r}"
        )

    if len(sorted_buckets) == 1:
        return sorted_buckets[0].tensor

    try:
        import torch as _torch

        return _torch.cat([b.tensor for b in sorted_buckets], dim=0)
    except Exception as exc:
        raise RuntimeError(
            f"merge_pp_shards: torch.cat failed for param_name={sorted_buckets[0].param_name!r}: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# apply_bucket_update
# ---------------------------------------------------------------------------


def apply_bucket_update(
    state_dict: Dict[str, Any],
    request: BucketUpdateRequest,
) -> BucketUpdateResult:
    """Apply a batch of weight buckets to *state_dict* in-place.

    Groups buckets by ``param_name``, merges multi-shard PP groups with
    ``merge_pp_shards()``, then copies the merged tensor into the
    corresponding entry in *state_dict*.

    Missing parameters are logged as failures but do not abort the remaining
    updates (fail-partial semantics).

    Args:
        state_dict: Mutable model state dict, e.g. from ``model.state_dict()``.
            Values must support ``.copy_()`` (standard PyTorch tensors do).
        request: The update payload to apply.

    Returns:
        A ``BucketUpdateResult`` summarising applied/failed counts.
    """
    applied = 0
    failed = 0
    errors: List[str] = []

    # Group by param_name (preserving insertion order within each group).
    groups = groupby(sorted(request.buckets, key=lambda b: b.param_name), key=lambda b: b.param_name)

    for param_name, bucket_iter in groups:
        bucket_list = list(bucket_iter)
        try:
            merged = merge_pp_shards(bucket_list)
        except Exception as exc:
            failed += 1
            errors.append(f"{param_name}: shard merge failed — {exc}")
            continue

        if param_name not in state_dict:
            failed += 1
            errors.append(f"{param_name}: not found in state_dict")
            continue

        try:
            state_dict[param_name].copy_(merged.cpu())
            applied += 1
        except Exception as exc:
            failed += 1
            errors.append(f"{param_name}: copy_ failed — {exc}")

    return BucketUpdateResult(
        sync_id=request.sync_id,
        applied=applied,
        failed=failed,
        errors=errors,
    )
