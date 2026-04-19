"""Gate 2.5 — Part 2: Selective sync via dynamic NCCL group.

Validates the CPU-cache → dynamic-NCCL-group → target-rank weight transfer
that ``ModelUpdateServiceCached`` uses during expand.

Scenario (2 GPUs, tp=1 for both training and inference):
  - rank 0 = training worker (cache owner, sender)
  - rank 1 = inference worker (receiver)

Steps:
  1. rank 0 has "trained" weights in a CPUBucketCache.
  2. rank 1 has a zeroed "inference" state dict.
  3. A dynamic NCCL group is created between rank 0 and rank 1.
  4. rank 0 broadcasts each bucket CPU → GPU staging → NCCL broadcast.
  5. rank 1 receives and applies each bucket to its state dict.
  6. Assert: every weight on rank 1 equals rank 0's original weights.
  7. Dynamic group is destroyed.
  8. Repeat 3 times to verify group create/destroy stability.

Run with:
    torchrun --nproc-per-node=2 tests/integration/test_gate2_5_selective_sync.py

Expected: all checks print PASS and script exits 0.
"""
from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path
from typing import Dict

import torch
import torch.distributed as dist

N_SYNC_CYCLES = 3          # how many times to create/use/destroy the group
TENSOR_ELEMENTS = 1024 * 1024  # 1M elements per "param" (~2 MB at bfloat16)
N_PARAMS = 8               # number of fake parameters
RTOL = 0.0                 # must be bit-for-bit identical

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import importlib.util as _ilu
from pathlib import Path as _Path

def _load_mod(name, file):
    spec = _ilu.spec_from_file_location(name, file)
    mod = _ilu.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_pipeline_dir = REPO_ROOT / "rlix" / "pipeline"
_bc = _load_mod("rlix.pipeline.bucket_cache", _pipeline_dir / "bucket_cache.py")
_br = _load_mod("rlix.pipeline.bucket_receiver", _pipeline_dir / "bucket_receiver.py")

CPUBucketCache = _bc.CPUBucketCache
BucketUpdateRequest = _br.BucketUpdateRequest
apply_bucket_update = _br.apply_bucket_update


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def r() -> int:
    return dist.get_rank()

def log(msg: str) -> None:
    if r() == 0:
        print(f"[rank0] {msg}", flush=True)

def fail(msg: str) -> None:
    print(f"[rank{r()}] FAIL: {msg}", flush=True)
    dist.barrier()
    sys.exit(1)

def check(condition: bool, msg: str) -> None:
    # gather pass/fail from all ranks
    t = torch.tensor([1 if condition else 0], device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MIN)
    passed = t.item() == 1
    if not passed:
        fail(msg)
    else:
        log(f"PASS  {msg}")

def gpu_allocated_mb() -> float:
    return torch.cuda.memory_allocated() / (1024 ** 2)


# ---------------------------------------------------------------------------
# Dynamic NCCL group helpers
# (simplified version of what ModelUpdateServiceCached will do)
# ---------------------------------------------------------------------------

def create_dynamic_group(group_name: str, ranks: list[int]) -> dist.ProcessGroup:
    """Create a new NCCL process group for the given ranks."""
    new_group = dist.new_group(ranks=ranks, backend="nccl")
    return new_group

def destroy_dynamic_group(group: dist.ProcessGroup) -> None:
    """Destroy a dynamically created NCCL process group."""
    dist.destroy_process_group(group)


# ---------------------------------------------------------------------------
# Build fake "trained" weights on sender (rank 0)
# ---------------------------------------------------------------------------

def make_trained_weights() -> Dict[str, torch.Tensor]:
    """Deterministic non-zero weights that differ per parameter."""
    torch.manual_seed(42)
    return {
        f"layer_{i}.weight": torch.randn(TENSOR_ELEMENTS, dtype=torch.bfloat16)
        for i in range(N_PARAMS)
    }


def make_zero_infer_weights(reference: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {
        name: torch.zeros_like(tensor, device="cuda")
        for name, tensor in reference.items()
    }


# ---------------------------------------------------------------------------
# One full selective sync cycle
# ---------------------------------------------------------------------------

def run_selective_sync_cycle(
    cycle: int,
    trained: Dict[str, torch.Tensor],
    infer_sd: Dict[str, torch.Tensor],
) -> None:
    """
    rank 0: sender — has trained weights in CPU cache.
    rank 1: receiver — has zeroed inference state dict on GPU.
    """
    world_size = dist.get_world_size()
    group_name = f"selective_sync_cycle_{cycle}_{uuid.uuid4().hex[:6]}"
    log(f"  [{cycle}] creating dynamic group '{group_name}'")

    sender_rank = 0
    receiver_rank = 1
    group_ranks = [sender_rank, receiver_rank]

    # All ranks in world participate in new_group() call
    dynamic_group = create_dynamic_group(group_name, group_ranks)

    if r() == sender_rank:
        # Build cache from trained weights
        cache = CPUBucketCache()
        for name, tensor in trained.items():
            cache.store(name, shard_id=0, tensor=tensor.contiguous())

        buckets = cache.get_dirty_buckets()
        log(f"  [{cycle}] sender: broadcasting {len(buckets)} buckets")

        # Broadcast bucket count so receiver knows how many to expect
        count_t = torch.tensor([len(buckets)], device="cuda")
        dist.broadcast(count_t, src=sender_rank, group=dynamic_group)

        for bucket in buckets:
            # Stage CPU → GPU
            gpu_tensor = bucket.tensor.to("cuda", non_blocking=False)

            # Broadcast shape metadata: [ndim, *shape]
            shape = list(gpu_tensor.shape)
            meta = torch.tensor([len(shape)] + shape, dtype=torch.int64, device="cuda")
            dist.broadcast(meta, src=sender_rank, group=dynamic_group)

            # Broadcast dtype as int
            dtype_id = torch.tensor([gpu_tensor.dtype == torch.bfloat16], device="cuda")
            dist.broadcast(dtype_id, src=sender_rank, group=dynamic_group)

            # Broadcast actual data
            dist.broadcast(gpu_tensor, src=sender_rank, group=dynamic_group)

    elif r() == receiver_rank:
        # Receive bucket count
        count_t = torch.zeros(1, dtype=torch.int64, device="cuda")
        dist.broadcast(count_t, src=sender_rank, group=dynamic_group)
        n_buckets = count_t.item()
        log(f"  [{cycle}] receiver: expecting {n_buckets} buckets")

        received: Dict[str, torch.Tensor] = {}
        param_names = list(trained.keys())  # receiver knows the param names in order

        for i, name in enumerate(param_names[:n_buckets]):
            # Receive shape
            # max ndim = 4, so meta has at most 5 elements; we receive [ndim, *shape]
            meta = torch.zeros(5, dtype=torch.int64, device="cuda")
            dist.broadcast(meta, src=sender_rank, group=dynamic_group)
            ndim = meta[0].item()
            shape = tuple(meta[1:1+ndim].tolist())

            # Receive dtype flag
            dtype_id = torch.zeros(1, device="cuda")
            dist.broadcast(dtype_id, src=sender_rank, group=dynamic_group)
            dtype = torch.bfloat16 if dtype_id.item() else torch.float32

            # Receive tensor
            buf = torch.zeros(shape, dtype=dtype, device="cuda")
            dist.broadcast(buf, src=sender_rank, group=dynamic_group)
            received[name] = buf

        # Apply to inference state dict
        for name, tensor in received.items():
            if name in infer_sd:
                infer_sd[name].copy_(tensor)

    else:
        # Other ranks (world_size > 2): not in dynamic group, skip
        pass

    # Destroy dynamic group
    if r() in group_ranks:
        destroy_dynamic_group(dynamic_group)

    dist.barrier()
    log(f"  [{cycle}] dynamic group destroyed")


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_weights(
    trained: Dict[str, torch.Tensor],
    infer_sd: Dict[str, torch.Tensor],
    cycle: int,
) -> None:
    """rank 1 verifies its infer_sd matches rank 0's trained weights."""
    if r() != 1:
        return

    mismatches: list[str] = []
    for name, original in trained.items():
        received = infer_sd[name].cpu()
        if received.shape != original.shape:
            mismatches.append(f"{name}: shape {received.shape} != {original.shape}")
        elif received.dtype != original.dtype:
            mismatches.append(f"{name}: dtype {received.dtype} != {original.dtype}")
        elif not torch.equal(received, original):
            max_diff = (received.float() - original.float()).abs().max().item()
            mismatches.append(f"{name}: max_diff={max_diff:.6f}")

    if mismatches:
        print(f"[rank1] FAIL cycle {cycle}: {len(mismatches)} weight mismatches:")
        for m in mismatches[:5]:
            print(f"  {m}")
        sys.exit(1)
    else:
        print(f"[rank1] PASS cycle {cycle}: all {len(trained)} weights correct", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    world_size = dist.get_world_size()
    log(f"world_size={world_size}, GPU={torch.cuda.get_device_name(local_rank)}")

    if world_size < 2:
        log("SKIP: Gate 2.5 part 2 requires at least 2 GPUs")
        dist.destroy_process_group()
        return

    # Create trained weights on rank 0 (CPU); broadcast names to rank 1
    trained: Dict[str, torch.Tensor] = {}
    if r() == 0:
        trained = make_trained_weights()

    # Broadcast param names so rank 1 knows expected keys
    param_names_encoded: list[str] = list(trained.keys()) if r() == 0 else []
    obj = [param_names_encoded]
    dist.broadcast_object_list(obj, src=0)
    param_names = obj[0]

    if r() == 1:
        # Reconstruct trained dict structure on receiver for verification
        torch.manual_seed(42)
        trained = {
            f"layer_{i}.weight": torch.randn(TENSOR_ELEMENTS, dtype=torch.bfloat16)
            for i in range(N_PARAMS)
        }

    # Inference state dict lives on GPU (rank 1 only)
    infer_sd = make_zero_infer_weights(trained) if r() == 1 else {}

    before_mb = gpu_allocated_mb()
    log(f"GPU before sync cycles: {before_mb:.1f} MB")

    for cycle in range(1, N_SYNC_CYCLES + 1):
        log(f"=== Sync cycle {cycle}/{N_SYNC_CYCLES} ===")
        run_selective_sync_cycle(cycle, trained, infer_sd)
        verify_weights(trained, infer_sd, cycle)
        dist.barrier()

        # Reset infer_sd to zeros for next cycle (re-test idempotency)
        if r() == 1:
            for t in infer_sd.values():
                t.zero_()

    after_mb = gpu_allocated_mb()
    log(f"GPU after sync cycles: {after_mb:.1f} MB")

    # VRAM must not have grown significantly from repeated group create/destroy
    vram_leak_mb = after_mb - before_mb
    if r() == 0:
        if vram_leak_mb > 200:
            print(f"[rank0] FAIL: VRAM grew {vram_leak_mb:.1f} MB across {N_SYNC_CYCLES} cycles (leak?)")
            sys.exit(1)
        else:
            print(f"[rank0] PASS  VRAM stable: grew {vram_leak_mb:.1f} MB across {N_SYNC_CYCLES} cycles")

    dist.barrier()
    log(f"ALL GATE 2.5 PART 2 CHECKS PASSED ({N_SYNC_CYCLES} cycles)")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
