"""Cache-aware ModelUpdateService that uses CPUBucketCache for PP gather + selective sync.

This module extends the base ``ModelUpdateService`` pattern with a CPU-resident
bucket cache layer.  Instead of directly invoking NCCL/IPC for every sync, the
service:

1. **Gathers** PP-sharded weights from all training workers into a CPU bucket
   cache owned by the ``pp_rank==0 / dp_rank==0 / tp_rank==0`` worker.
2. **Selectively syncs** only the dirty (changed) buckets to the inference workers.
3. **Marks** buckets clean after a successful sync.  The next sync round will
   only push buckets that have been modified since the last sync.

Relationship to the base ``ModelUpdateService``:
    This class is a higher-level orchestrator that owns a ``CPUBucketCache`` and
    adds ``populate_cache_from_workers()`` and ``sync_from_cache()`` on top.  The
    lower-level NCCL/IPC transport (``_build_comm_plan_for_sender``, etc.) lives
    in the base class and is unchanged.

Architecture overview::

    Training cluster workers (all PP ranks)
        └─ populate_cache_from_workers()
              ├─ worker.get_pp_weight_shards()  [per PP rank]
              └─ cache.store(param_name, shard_id=pp_rank, tensor)

    CPUBucketCache (owner: pp/dp/tp rank 0)
        └─ get_dirty_buckets()  ──►  sync_from_cache()
                                          └─ tgt_worker.receive_weight_update(request)
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from rlix.pipeline.bucket_cache import Bucket, CPUBucketCache
from rlix.pipeline.bucket_receiver import BucketUpdateRequest

try:
    from roll.utils.logging import get_logger
    logger = get_logger()
except Exception:  # pragma: no cover
    import logging as _logging
    logger = _logging.getLogger(__name__)  # type: ignore[assignment]


class ModelUpdateServiceCached:
    """Cache-aware model weight sync service for a single pipeline.

    Owns a :class:`CPUBucketCache` that holds the latest weights gathered from
    all PP ranks.  Provides two high-level operations:

    - :meth:`populate_cache_from_workers`: pull weight tensors from every
      training worker into the cache (PP gather step).
    - :meth:`sync_from_cache`: push dirty cache buckets to the specified
      inference workers (selective sync step).

    Args:
        pipeline_id: Unique identifier for the owning pipeline.
        src_cluster: ROLL ``Cluster`` for the training workers.
        tgt_cluster: ROLL ``Cluster`` for the inference workers.
        bucket_size_bytes: Passed through to :class:`CPUBucketCache`.
    """

    def __init__(
        self,
        *,
        pipeline_id: str,
        src_cluster: Any,
        tgt_cluster: Any,
        bucket_size_bytes: int = 256 * 1024 * 1024,
    ) -> None:
        if not isinstance(pipeline_id, str) or pipeline_id == "":
            raise ValueError("pipeline_id must be a non-empty string")
        self.pipeline_id = pipeline_id
        self.src_cluster = src_cluster
        self.tgt_cluster = tgt_cluster
        self.cache = CPUBucketCache(bucket_size_bytes=bucket_size_bytes)

    # ------------------------------------------------------------------
    # PP gather
    # ------------------------------------------------------------------

    def populate_cache_from_workers(self) -> None:
        """Pull weight shards from all training workers into the CPU cache.

        Each worker is called with ``get_pp_weight_shards()`` which returns a
        ``{param_name: tensor}`` dict for that worker's PP layer slice.  The
        worker's ``pp_rank`` is used as the ``shard_id`` so that buckets from
        different PP ranks can be merged later by :func:`merge_pp_shards`.

        The cache is **not** cleared before populate; existing buckets are
        overwritten by the new tensors and re-marked dirty.  This means a
        partial populate (e.g. only one PP rank changed) correctly marks only
        the affected buckets dirty.
        """
        for rank, worker in enumerate(self.src_cluster.workers):
            pp_rank = int(self.src_cluster.worker_rank_info[rank].pp_rank)
            shards: Dict[str, Any] = worker.get_pp_weight_shards()
            for param_name, tensor in shards.items():
                self.cache.store(param_name, shard_id=pp_rank, tensor=tensor)

        logger.info(
            f"[ModelUpdateServiceCached] populated cache pipeline_id={self.pipeline_id} "
            f"total_buckets={self.cache.size()} "
            f"dirty={len(self.cache.get_dirty_buckets())}"
        )

    # ------------------------------------------------------------------
    # Selective sync
    # ------------------------------------------------------------------

    def sync_from_cache(self, tgt_dp_ranks: List[int]) -> None:
        """Push dirty cache buckets to the specified inference workers.

        Only buckets that are currently marked dirty will be sent.  After a
        successful dispatch to all target workers, the sent buckets are marked
        clean.

        If there are no dirty buckets, the method returns immediately without
        making any remote calls.

        Args:
            tgt_dp_ranks: Data-parallel ranks in the inference cluster to update.
        """
        dirty_buckets: List[Bucket] = self.cache.get_dirty_buckets()
        if not dirty_buckets:
            logger.info(
                f"[ModelUpdateServiceCached] sync_from_cache skipped (no dirty buckets) "
                f"pipeline_id={self.pipeline_id}"
            )
            return

        sync_id = f"cache_sync/{self.pipeline_id}/{uuid.uuid4().hex[:8]}"
        request = BucketUpdateRequest(sync_id=sync_id, buckets=dirty_buckets)

        logger.info(
            f"[ModelUpdateServiceCached] sync_from_cache_start pipeline_id={self.pipeline_id} "
            f"sync_id={sync_id} dirty_buckets={len(dirty_buckets)} tgt_dp_ranks={tgt_dp_ranks}"
        )

        for dp_rank in tgt_dp_ranks:
            tgt_worker = self.tgt_cluster.rank2worker[int(dp_rank)]
            result = tgt_worker.receive_weight_update(request)
            if not result.ok:
                logger.warning(
                    f"[ModelUpdateServiceCached] partial sync pipeline_id={self.pipeline_id} "
                    f"sync_id={sync_id} dp_rank={dp_rank} "
                    f"applied={result.applied} failed={result.failed} errors={result.errors}"
                )

        # Mark sent buckets clean after all workers confirmed receipt.
        synced_keys = [(b.param_name, b.shard_id) for b in dirty_buckets]
        self.cache.mark_synced(synced_keys)

        logger.info(
            f"[ModelUpdateServiceCached] sync_from_cache_done pipeline_id={self.pipeline_id} "
            f"sync_id={sync_id}"
        )
