# ENG-123 Multi-Pipeline Extraction: Unified Master Issue Tracker

> **Updated**: 2026-02-10  
> **Scope**: Functional parity with fork via SchedRL + patched upstream ROLL  
> **Triage rules**:  
> 1. No dead code — skip fork features disabled/unused by smoke test  
> 2. Prefer upstream APIs — adopt upstream's newer versions when possible  
> 3. Only flag issues where upstream API is NOT functionally equivalent  
> 4. refer working examples:
  1. `examples/multi_pipeline/pipeline1_sokoban_grpo.yaml` (Clean reference implementation)
  2. `examples/multi_pipeline/pipeline2_sokoban_grpo.yaml` (Companion pipeline)
  3. `examples/multi_pipeline/start_multi_pipeline_test.py` (Main entry point for multi-pipeline testing)  as the reference config for smoke test parity 
---

## Legend

| Tag | Meaning |
|-----|---------|
| 🔴 P0 | Blocks implementation; must fix before coding starts |
| 🔴 P1 | Blocks correctness; must fix in the relevant phase |
| 🟡 P2 | Affects robustness; fix before production, okay to defer for ENG-123 |
| 🟢 P3 | Plan quality / cosmetic; document and move on |
| ✅ RESOLVED | Non-issue after analysis; documented for audit trail |

---

## Category A: Concurrency & Locking Model

### A1 ✅ DESIGN RESOLVED — Two-lock design: `routing_lock` + `swapping_lock`
_Merges: Review #22, Review #14, Review #12, Review #19, SERIOUS-9, SERIOUS-11_
_Previously A1 (P0) + A2 (P1) — merged after lock-safety analysis on 2026-02-10_
_Updated 2026-02-10: upgraded from single-lock to two-lock for defense-in-depth_

**Decision: Two locks with distinct responsibilities. Equivalent to fork's `request_lock` + `swapping_lock`, but simpler (no `already_locked` flag threading).**

#### Lock definitions

```python
self.routing_lock = asyncio.Lock()    # Brief: protects routing metadata
self.swapping_lock = asyncio.Lock()   # Full-duration: serializes shrink/expand
```

| Lock | Protects | Hold duration | Acquired by `generate_one_request()`? |
|------|----------|---------------|--------------------------------------|
| `routing_lock` | `active_dp_ranks`, `src_rank2_dp_rank`, `need_suspend` | Microseconds | ✅ Yes (briefly) |
| `swapping_lock` | Worker physical state (prevents concurrent offload+load on same workers) | Seconds (offload/load) | ❌ Never |

**Key property:** `generate_one_request()` never touches `swapping_lock`, so request routing is never blocked by offload/load operations. Only other lifecycle ops (concurrent shrink/expand) are serialized.

#### Comparison with fork's two-level locking

| Aspect | Fork | Ours |
|--------|------|------|
| Outer lock | `swapping_lock` | `swapping_lock` |
| Inner lock | `request_lock` | `routing_lock` |
| `suspend(already_locked=...)` | 6+ methods with `already_locked` / `request_locked` boolean flags | Not needed — `suspend`/`resume` are always called inside `swapping_lock` |
| `generate_one_request()` | Only acquires `request_lock` | Only acquires `routing_lock` |

#### Why `swapping_lock` is needed

Without it, concurrent shrink + expand can race on the **same worker's physical GPU state**:

```
Shrink({0,1,2,3}):  [routing_lock: clear ranks] [offload rank 0...]
Expand({0,1}):                                  [load rank 0...]  💥 RACE
```

`routing_lock` can't prevent this because offload/load happen outside it (by design, to avoid stalling request routing). `swapping_lock` serializes the entire operation so load waits for offload to complete.

#### Concrete implementation

```python
async def shrink_workers(self, target_gpus):
    async with self.swapping_lock:                      # Serialize vs other lifecycle ops
        offload_ranks = self._calculate_offload_ranks(target_gpus)
        will_shrink_to_zero = (self.active_dp_ranks - set(offload_ranks)) == set()

        # Phase A: Brief metadata update under routing_lock
        async with self.routing_lock:
            if will_shrink_to_zero:
                self.need_suspend = True
                self.suspend_notifier.clear()
            self.active_dp_ranks -= set(offload_ranks)
            src_ranks_to_remap = {s for s, d in self.src_rank2_dp_rank.items()
                                  if d in set(offload_ranks)}
            self._clear_src_rank_mappings(src_ranks_to_remap)
            abort_targets = {dp: list(self.running_requests[dp]) for dp in offload_ranks}
        # routing_lock released — active workers route immediately

        # Phase B: Abort + drain (no routing_lock, still under swapping_lock)
        abort_futs = [w.abort_requests.remote(ids) for dp, ids in abort_targets.items() if ids]
        if abort_futs:
            await asyncio.gather(*abort_futs)
        while sum(len(self.running_requests[dp]) for dp in offload_ranks) > 0:
            await asyncio.sleep(0.5)

        # Phase C: Offload (still under swapping_lock, prevents concurrent expand)
        offload_refs = self.infer_cluster.offload_states_partial(offload_ranks, blocking=False)
        await asyncio.gather(*[asyncio.wrap_future(r.future()) for r in offload_refs])
    # swapping_lock released

async def expand_workers(self, target_gpus):
    async with self.swapping_lock:                      # Waits for any in-progress shrink
        load_ranks = self._calculate_load_ranks(target_gpus)

        # Phase A: Load states (under swapping_lock, prevents concurrent shrink)
        load_refs = self.infer_cluster.load_states_partial(load_ranks, blocking=False)
        await asyncio.gather(*[asyncio.wrap_future(r.future()) for r in load_refs])

        # Phase B: Brief metadata update under routing_lock
        async with self.routing_lock:
            self.active_dp_ranks |= set(load_ranks)
            self.need_suspend = False
            self.suspend_notifier.set()
    # swapping_lock released

async def generate_one_request(self, data):
    # Does NOT acquire swapping_lock — never blocked by lifecycle ops
    while True:
        await self._check_suspend()
        async with self.routing_lock:                    # Brief hold only
            if self.need_suspend:
                continue  # Shrink happened while we waited; re-check suspend
            dp_rank = self._route(data)
            break
    return await self.infer_cluster.workers[dp_rank].generate_request.remote(data)
```

#### 3 upstream fixes required

**Fix 1: Split `_rebalance_on_shrink` into Phase A (under `routing_lock`) + Phase B (no lock)**
- Current upstream holds `routing_lock` during the entire drain loop (`while True: await asyncio.sleep(3)`)
- This blocks ALL request routing for seconds → unacceptable

**Fix 2: Remove `ValueError("Cannot shrink to zero active ranks")` guard (line 1508)**
- Multi-pipeline requires shrink-to-zero
- When `active_dp_ranks` is empty, `need_suspend=True` blocks new requests at `_check_suspend()`

**Fix 3: Add suspend re-check in `generate_one_request()` after acquiring `routing_lock`**
- Race window: request passes `_check_suspend()`, blocks on `routing_lock`, shrink-to-zero runs, lock released → caller finds empty `active_dp_ranks`
- Fix: `while True` loop re-checks `need_suspend` after acquiring lock

#### Performance summary

| Scenario | `routing_lock` hold time | Request routing during lifecycle op |
|----------|-------------------------|-------------------------------------|
| Partial shrink | ~μs (metadata update) | ✅ Active workers keep serving |
| Shrink-to-zero | ~μs (metadata update) | N/A (no active workers) |
| Expand | ~μs (metadata update) | ✅ Active workers keep serving |
| Concurrent shrink+expand | Serialized by `swapping_lock` | ✅ Active workers keep serving |

**Owner:** Phase 3 §4 — upstream patch to `RequestScheduler` (add `swapping_lock`, refactor `_rebalance_on_shrink`, fix `generate_one_request`).

---

### A3 🟡 P2 — `notify_ready_to_release` idempotency underspecified
_Merges: Review #25, SERIOUS-12_

**Fact:** Fork's implementation is 107 lines with `offload_lock`, `offload_notified` flag, `offload_event`, step-level deduplication, first-caller vs. subsequent-caller serialization. Called from 4 different sites.

**Resolution:** Port fork's test-and-set idempotency pattern. Minimum contract:
- Atomic `try_set_offload_notified()` — returns True for first caller only
- Subsequent callers wait on `offload_event` 
- Step-level dedup: skip if `global_step <= last_completed_release_step`

**Owner:** Phase 3 — implement in SchedRL coordinator or adapter.

---

### A4 🟡 P2 — Abort retry has no backoff
_Merges: Review #24_

**Fact:** After abort, `TrajEnvManager.make_decision()` immediately retries → `generate_one_request()` → blocked at suspend gate. No backoff between retries.

**Resolution:** Accept for ENG-123; document risk. Add backoff only if observed flooding during testing.

**Owner:** Deferred.

---

## Category B: Worker Lifecycle (load/offload/start/stop)

### B1 ✅ RESOLVED — `load_states_partial` signature mismatch blocks expand path
_Merges: Review #21, SERIOUS-8_

**Fact:**
| | Upstream | Fork |
|-|----------|------|
| Signature | `load_states_partial(target_dp_ranks)` | `load_states_partial(active_dp_ranks, start_server_thread, server_meta_info, target_dp_ranks)` |
| Lifecycle target | `target_dp_ranks` | mixed `active_dp_ranks` + `target_dp_ranks` |
| Server/KV behavior | `strategy.load_states()` (vLLM wake-up path) | Explicit `start_server_thread` + `server_meta_info` |
| Idempotency | `assert` on already-loaded → crashes | `warning + return` → safe retry |

**Problem:** Port plan was still carrying fork-only arguments and mixed target concepts. For ENG-123 we standardize on `dp_ranks` and the upstream 1-arg API.

**Resolution (updated):** 
- Use upstream `load_states_partial(dp_ranks)` directly in adapter expand flow (no extra thread/meta args).
- Treat lifecycle RPC targets as `dp_ranks` everywhere (`expand_workers(dp_ranks)`, `shrink_workers(dp_ranks)`).
- Keep checkpoint/version signaling decoupled (`promote_active_checkpoint(...)` + `ModelUpdateService.sync_selected_workers(...)`).
- Keep upstream fail-fast precondition assertions unchanged (`assert is_loaded is False` for load, `assert is_loaded is True` for offload).

**Owner:** Phase 3 §4 — plan cleanup (completed).

---

### B2 ✅ DESIGN RESOLVED — `offload_states_partial` lacks server thread cleanup
_Merges: Review #23, SERIOUS-7_

**Fact:**
| | Upstream | Fork |
|-|----------|------|
| Offload path | `strategy.offload_states()` | Explicit stop/join of fork-only server thread + state offload |

**Problem (actual in upstream-aligned port):** GPU memory may not be freed on shrink because strategy offload is gated by `pipeline_config.is_actor_infer_colocated` (e.g., vLLM and SGLang strategies). In multi-pipeline time-sharing, shrink/offload must be triggered regardless of colocation flags.

**Resolution (updated):**
- Do not port fork-only server-thread lifecycle (`stop_server`/`join`) into adapter.
- Keep upstream `offload_states_partial(dp_ranks)` calling `strategy.offload_states()`.
- Patch framework strategies to support scheduler-mandated offload bypassing `is_actor_infer_colocated` (e.g., `force=True` argument or equivalent hook), so shrink/offload always releases GPU memory when SchedRL issues shrink/stop.
- Keep fail-fast behavior: if offload does not release memory, crash (existing post-offload checks apply).

**Owner:** Phase 3 §4 — adapter implementation.

---

### B3 ✅ DESIGN RESOLVED — `teardown_collective_groups` missing upstream
_Merges: SERIOUS-13_

**Fact:** Fork already has dynamic group teardown for selective model update (`SelectiveModelUpdateGroup.teardown()` calling `teardown_collective_groups(...)` on all participant workers). Upstream is missing the equivalent teardown surface for selective-update groups.

**Resolution:** Port fork semantics into upstream `third_party/ROLL`:
- Add `teardown_collective_groups(model_update_name, group_names)` to the upstream Strategy/Worker surface, implemented by calling `roll.utils.collective.collective.destroy_collective_group(group_name)` and removing any per-update comm-plan bookkeeping.
- Ensure teardown is invoked for every created group on **every participant** rank (sender + all receivers), ideally via a `finally:` block in selective update orchestration so timeouts/exceptions do not leak groups.
- Fix upstream `GroupManager.destroy_collective_group()` to call `dist.destroy_process_group(g)` and maintain correct bookkeeping (not dict-only deletion) to prevent NCCL resource leaks in long-running jobs.

**Owner:** Phase 4.

---

## Category C: GPU Scheduler & Orchestration (SchedRL core)

### C1 ✅ DESIGN RESOLVED — `ConcurrentAgenticPipeline` (2,110 lines) decomposed
_Merges: Review #18_
_Resolved 2026-02-10 via method-level analysis of all 44 items in the class_

**Fact:** Single bullet point in plan for the largest porting task. 44 methods, ~2,110 lines covering init, run loop (17 phases), GPU helpers, timeline tracing, validation.

**Resolution:** Full method-level decomposition into 4 destination modules.

#### Destination 1: SchedRL Core — GPU Scheduler Client (`schedrl/gpu_client.py`, ~212 lines)

These methods talk to `CentralizedGPUScheduler` and know nothing about ROLL clusters/workers:

| Method | Lines | Size | Notes |
|--------|-------|------|-------|
| `_request_gpu_and_wait()` | 941-993 | 53 | Blocking `request_gpus.remote()` with timeout |
| `_release_and_request_gpu_blocking()` | 995-1053 | 59 | Atomic release+request, calls `_offload_cluster_all()` before release |
| `_validate_gpu_allocation()` | 1056-1136 | 81 | Validates allocation vs device_mapping at DP worker boundaries |
| `_release_gpu()` | 1138-1156 | 19 | Release + offload + `release_gpus.remote()` |

**Porting notes:**
- `_release_and_request_gpu_blocking()` and `_release_gpu()` call `_offload_cluster_all()` before releasing. This creates a dependency on the cluster helpers (Destination 2). Either accept the dependency or extract offload into a callback.
- `_validate_gpu_allocation()` knows about TP size via `cluster_config` — pass as parameter, don't import ROLL types.

#### Destination 2: Adapter — Pipeline Orchestration (`schedrl/adapter/pipeline.py`, ~1,230 lines)

The bulk of the class. Calls both SchedRL GPU client and upstream ROLL cluster APIs:

**Init & wiring (309 lines):**

| Method | Lines | Size | Notes |
|--------|-------|------|-------|
| `__init__()` | 158-466 | 309 | Creates actor_train/actor_infer/critic clusters, RolloutSchedulers, ModelUpdateService, requests init GPUs, looks up RequestScheduler handle |

**Cluster state helpers (78 lines):**

| Method | Lines | Size | Notes |
|--------|-------|------|-------|
| `logger` (property) | 468-473 | 6 | On-demand logger |
| `_get_cluster_strategy_load_states()` | 475-485 | 11 | Check per-worker GPU load state |
| `_ensure_cluster_offloaded()` | 487-512 | 26 | Verify offloaded + force if needed |
| `_get_actor_infer_sleep_level()` | 514-520 | 7 | Read sleep_level from config |
| `_offload_cluster_all()` | 522-536 | 15 | Offload with `stop_server` for actor_infer |
| `_get_cluster_for_cluster_id()` | 538-545 | 8 | Lookup `self.actor_train/actor_infer/critic` by cluster_id string |

**Model update (26 lines):**

| Method | Lines | Size | Notes |
|--------|-------|------|-------|
| `model_update()` | 547-572 | 26 | Thin wrapper over `super().model_update()` with timeout guard |

**Run loop (416 lines):**

| Method | Lines | Size | Notes |
|--------|-------|------|-------|
| `run()` | 1159-1575 | 416 | 17-phase orchestration: suspend → model_update → request GPUs → expand → rollout → batch process → value compute → log probs → advantage → critic train → actor train → metrics. Uses GPU client for request/release, upstream ROLL for compute. |

**Eval & batch processing (92 lines):**

| Method | Lines | Size | Notes |
|--------|-------|------|-------|
| `val()` | 1577-1608 | 32 | Validation evaluation loop |
| `adjust_batch()` | 1610-1669 | 60 | Multi-turn batch adjustment (copy/truncate modes) |

**Config validation (271 lines):**

| Method | Lines | Size | Notes |
|--------|-------|------|-------|
| `_validate_partial_gpu_config()` | 1671-1772 | 102 | Master validation: device_mapping overlap, capacity, DP size |
| `_validate_minimum_dp_size()` | 1774-1788 | 15 | DP size ≥ min_required |
| `_validate_critic_uses_freed_gpus()` | 1790-1817 | 28 | Critic on freed GPUs, disjoint from train |
| `_validate_reference_colocation()` | 1819-1837 | 19 | Reference colocates with actor_train |
| `_validate_freed_gpu_capacity()` | 1839-1857 | 19 | Sufficient freed GPU capacity |
| `_validate_parallelism_compatibility()` | 1859-1891 | 33 | TP/PP/EP match device_mapping |
| `_validate_minimum_active_ranks()` | 1893-1942 | 50 | At least 1 DP rank remains after shrink |

**Cleanup (38 lines):**

| Method | Lines | Size | Notes |
|--------|-------|------|-------|
| `cleanup()` | 1943-1980 | 38 | Terminate clusters, unregister from GPU scheduler |

#### Destination 3: Observability (defer-able, ~365 lines)

Nice-to-have tracing. Can defer for ENG-123 or port as-is:

| Method | Lines | Size | Notes |
|--------|-------|------|-------|
| `get_worker_metadata()` | 574-713 | 140 | Collects worker IDs, DP ranks, GPU IDs for timeline annotation |
| `_annotate_events()` | 715-812 | 98 | Annotate timeline events with worker metadata |
| `_save_and_annotate_timeline()` | 814-861 | 48 | Save final Ray timeline trace |
| `_save_timeline_snapshot()` | 863-939 | 77 | Rolling per-step snapshot (throttled) |

**Porting notes:** These only depend on `self.worker_metadata` (set in `__init__`) and Ray profiling APIs. Fully self-contained. Port as a mixin or defer entirely.

#### Destination 4: Module-level Utilities (~226 lines)

Standalone functions, not class methods:

| Function | Lines | Size | Port to | Notes |
|----------|-------|------|---------|-------|
| `_get_env_timeout_s()` | 56-79 | 24 | SchedRL utils | Env var → timeout with validation |
| `_get_env_timeout_optional_s()` | 82-110 | 29 | SchedRL utils | Same but returns None if disabled |
| `timeout_context()` | 113-133 | 21 | SchedRL utils | signal.alarm context manager |
| `_get_named_actor_with_timeout()` | 136-154 | 19 | SchedRL utils | Ray actor lookup with polling |
| `get_episode_scores()` | 1982-1988 | 7 | Adapter utils | Data extraction |
| `get_traj_rollout_time()` | 1990-1996 | 7 | Adapter utils | Data extraction |
| `get_traj_env_time()` | 1998-2004 | 7 | Adapter utils | Data extraction |
| `compute_data_metrics()` | 2006-2096 | 91 | Adapter utils | Metrics computation (scores, times, lengths) |
| `GroupFilter` class | 2098-2109 | 12 | Adapter | User-defined group filter interface |

#### Summary by destination

| Destination | Lines | % of total | Phase |
|-------------|-------|------------|-------|
| **SchedRL Core** (GPU client) | ~212 | 10% | Phase 2 |
| **Adapter** (pipeline orchestration) | ~1,230 | 58% | Phase 3 |
| **Observability** (timeline tracing) | ~365 | 17% | Defer / Phase 5 |
| **Module-level utilities** | ~226 | 11% | Phase 2-3 |
| **Constant/import preamble** | ~77 | 4% | – |
| **Total** | **2,110** | **100%** | |

#### Key dependencies between destinations

```
SchedRL Core (GPU client)
    ↓ calls _offload_cluster_all() before release
Adapter (pipeline orchestration)
    ↓ calls GPU client for request/release
    ↓ calls upstream ROLL Cluster APIs for compute
    ↓ optionally calls Observability for tracing
Observability (timeline)
    ↓ reads self.worker_metadata (set in __init__)
    ↓ uses Ray profiling APIs
Module-level utilities
    ← used by all of the above
```

**Owner:** This decomposition is now the Phase 3 task list. Each destination is an implementation chunk.

---

### C2 ✅ RESOLVED — `active_checkpoint_version` phantom dependency
_Merges: Review #26, SERIOUS-6_

**Fact:** Older plan text referenced `self.coordinator.active_checkpoint_version`, but no such concrete field exists in current code paths.

**Resolution (final):**
- Keep adapter lifecycle APIs version-free: `shrink_workers(worker_indices)` / `expand_workers(worker_indices)`.
- Infer shrink-to-zero inside callee from active-rank state.
- Coordinator owns promotion timing via explicit `promote_active_checkpoint(...)` signaling.
- Sender strategy cache state owns `active_rollout_checkpoint_version` and cache metadata.
- `ModelUpdateService` is orchestration-only for selective sync.
- `global_step` remains timeline/progress lineage only.

**Owner:** Plan cleanup (completed).

---

### C3 ✅ DESIGN RESOLVED — `_validate_calculated_ranks` broken for expand mode
_Merges: Review #16, SERIOUS-4_

**Fact:** Upstream ROLL `RequestScheduler._validate_calculated_ranks(ranks, mode)` ignores `mode` and unconditionally enforces `dp_rank in self.active_dp_ranks`. This is correct for shrink but wrong for expand.
- Buggy upstream code: `third_party/ROLL/roll/distributed/scheduler/generate_scheduler.py` (`_validate_calculated_ranks`, around the state-consistency check).
- Fork behavior is correct (mode-aware): `third_party/ROLL_multi_pipeline/roll/distributed/scheduler/generate_scheduler.py`.

Plan uses `skip_load=True` which bypasses validation, so this doesn't block ENG-123 directly. But it's a trap for future callers.

**Resolution (Option A):** Patch upstream `_validate_calculated_ranks` to be mode-aware:
- `mode="shrink"`: all ranks must already be active.
- `mode="expand"`: all ranks must NOT already be active.

**Owner:** Phase 3 §4 — upstream patch.

---

### C4 🟡 P2 — `_rebalance_on_expand` infinite loop risk
_Merges: SERIOUS-1_

**Fact:** `_rebalance_on_expand` computes `src_ranks_to_abort` from `len(src_rank2_dp_rank)` (all mappings), but selects abort candidates only from `old_active_dp_ranks`. If `src_rank2_dp_rank` contains "zombie" entries pointing to inactive ranks, the abort target can become larger than the selectable pool. The current `cycle(dp_rank_to_src_ranks.keys())` loop can then spin forever once all per-worker lists are empty (no `await` inside the loop).

**Resolution:** Make rebalancing best-effort and guaranteed-terminating:
- Filter-first: compute the abortable pool using only mappings whose `dp_rank in old_active_dp_ranks`.
- Compute `planned_to_abort` using the abortable pool size (or clamp `planned_to_abort = min(planned_to_abort, available)`).
- If clamping occurs, `logger.warning(...)` and continue (rebalancing accuracy is not correctness-critical for ENG-123).
- Selection policy: abort from the most-loaded old worker each iteration (greedy balance), not round-robin.
  - Implementation: each iteration, pick `dp_rank = argmax len(dp_rank_to_src_ranks[dp_rank])` over the remaining non-empty workers, pop one `src_rank`, and repeat.
  - Termination condition: if `max_load == 0` (no `src_ranks` left to steal), stop selection even if `remaining_to_abort > 0` and `logger.warning(...)` (best-effort).

**Owner:** Phase 3 §4 — upstream patch (`RequestScheduler._rebalance_on_expand`).

---

## Category D: Model Update & Selective Sync

### D1 ✅ DESIGN RESOLVED — Bucket caching hook violates Option A boundary
_Merges: Review #1_

**Fact (clarified):**
- Selective model update for TP>1 needs **sender-side bucket caching/staging**; rebuilding bucket metadata on every update is too slow.
- Fork behavior: bucket caching is triggered by `ModelUpdateService.refresh_sender_cache(...)` calling `Worker.build_model_update_bucket_cache(...)` on sender ranks (not by `ActorWorker.initialize`).
- Upstream port needs a minimal way to wire this capability (strategy/worker surface) without pulling scheduler policy into core ROLL.

**Resolution (Option B; validated):**
- Explicitly bless minimal upstream hook(s) as a boundary exception to enable sender-strategy-owned bucket caching for ENG-123 selective sync.
- Keep the patch narrowly scoped to **wiring** only (no scheduler or pipeline orchestration logic in core ROLL):
  - Add a small hook right after strategy creation in `ActorWorker.initialize` to wrap/augment the strategy for bucket-cache wiring (Megatron-only in ENG-123; others fail fast).
  - The actual cache build remains service-driven (`ModelUpdateService.refresh_sender_cache(...)`), matching the fork control flow.
- Document exactly which upstream files are patched and why (ENG-123 TP>1 selective sync perf requirement).

**Owner:** Phase 4 — boundary exception documentation.

---

### D2 ✅ DESIGN RESOLVED — `update_parameter_in_bucket` signature mismatch
_Merges: Review #2_

**Fact:**
- vLLM `RollWorker`: expects `serialized_named_tensors[self.rank]` (list indexed by rank)
- Base `InferenceStrategy`: completely different 6-argument signature
- Plan claims dict `{worker.rank: bucket_data}` works — it doesn't for list-indexed path

**Resolution (updated):**
- Do not use a raw `(meta_infos, buffer, ranks_in_worker)` call shape against upstream vLLM `RollWorker.update_parameter_in_bucket(...)`.
- Always call the existing vLLM worker contract: `update_parameter_in_bucket(serialized_named_tensors, is_lora=False)` where each TP/PP worker reads `serialized_named_tensors[self.rank]`.
- If the payload is identical for all TP/PP ranks, use an indexable wrapper (`__getitem__`) that returns the same serialized bytes for any rank to avoid building a full list.
- Subset/selective DP updates are controlled by selecting which DP-rank engine(s) you invoke (the vLLM `collective_rpc(_async)` fanout is scoped to TP/PP workers inside that engine, not all DP ranks). Keep the fork receiver-side allowlist (`set_model_update_allowed_dp_ranks(tgt_dp_ranks)`) as a guardrail so non-target DP ranks can early-return even if a future path fans out wider.

**Owner:** Phase 4.

---

### D3 ✅ DESIGN RESOLVED — `send_recv_utils.py` port massively underscoped
_Merges: Review #17_

**Fact:** Fork adds ~237 new lines (TpShardSpec, `_TpShardAssembly`, `process_bucket_tp_sharded()`, `meta_to_dict()`/`dict_to_meta()`). These are consumed by 5+ vLLM version-specific worker files.

Plan says "port sharding logic" but doesn't scope the consumer-side changes.

**Resolution (updated; reuse upstream; port only what is needed):**
- Do not port the fork TP-shard receiver assembly (`TpShardSpec` / `_TpShardAssembly` / `RecvBucketManager.process_bucket_tp_sharded`) for ENG-123.
- Rationale: validated selective-update configs set `ROLL_SELECTIVE_MODEL_UPDATE_RECEIVER_DISABLE_CPU_STAGING=1`, which bypasses the fork’s CPU staging + TP shard-aware unpack path entirely.
- Use upstream `roll/utils/send_recv_utils.py` bucket format (`serialize_named_weights(...)` + `named_tensors_from_bucket(...)`) and upstream vLLM `RollWorker.update_parameter_in_bucket(serialized_named_tensors, ...)` contract (rank-indexed list).
- Limitation (explicit): ENG-123 selective update does not include the fork’s memory-optimized TP shard-aware unpack; if we later need lower receiver memory peak for TP>1, revisit and port the fork sharding path then.

**Owner:** Phase 4.

---

## Category E: Request Routing & Scheduling

### E1 ✅ DESIGN RESOLVED — `request_id` dual-write conflict
_Merges: Review #11_

**Fact:** 
- Plan says: `lm_input.meta_info["request_id"] = build_request_id(...)` in `make_decision()`
- Upstream `generate_one_request()` overwrites: `data.meta_info["request_id"] = f"{uuid}_{counter}"`
- Canonical `request_id` is lost

Plan introduced `schedrl_request_id` as the solution but left the stale `build_request_id` instruction.

**Resolution:** 
- Remove stale `build_request_id` instruction from Phase 3
- Use `schedrl_request_id` consistently for canonical ID
- Upstream `request_id` remains for internal routing (no upstream change)

**Owner:** Plan cleanup.

---

### E2 ✅ DESIGN RESOLVED — `RolloutScheduler` missing `pipeline_id` parameter
_Merges: SERIOUS-18_

**Fact:** Fork adds `pipeline_id=None` to `RolloutScheduler.__init__()` for unique per-pipeline actor naming. Upstream doesn't have it.

**Resolution:** Add `pipeline_id: Optional[str] = None` to upstream `RolloutScheduler.__init__()`. Use it to prefix both `GroupQueueManager` and `RequestScheduler` Ray actor names created via `.options(name=...)` to avoid multi-pipeline name collisions. Minimal, upstreamable patch (no child-actor signature changes required).

**Owner:** Phase 3 §4 — upstream patch.

---

### E3 🟢 P3 — `DynamicSamplingScheduler` not addressed
_Merges: Review #15_

**Fact:** Upstream has two scheduler types. Plan only covers `RequestScheduler` (agentic). ENG-123 is agentic-only.

**Resolution:** Document "ENG-123 agentic-only" restriction. Non-agentic support is out of scope.

**Owner:** Plan documentation.

---

## Category F: API Functional Equivalence (Verified This Session)

### F1 ✅ RESOLVED — `compute_advantage` is identical
Both fork and upstream `compute_advantage` in `functionals.py` are byte-for-byte identical. Smoke test uses `"grpo"` which doesn't hit the `"agentic_reinforce"` branch only present in upstream's `agentic_compute_advantage`.

### F2 ✅ RESOLVED — `agg_loss` difference is internal to workers
Fork uses `masked_mean()` for `token-mean`; upstream uses `.sum() / batch_num_tokens`. But `ConcurrentAgenticPipeline.run()` never calls `agg_loss` directly — it's called internally by `actor_train.train_step()`. Since we use upstream workers, upstream behavior is what we get. Acceptable.

### F3 ✅ RESOLVED — `use_kl_loss` exists in both codebases
Both `PPOConfig` classes have `use_kl_loss: bool = field(default=False)`. Fork's `assert not self.pipeline_config.use_kl_loss` works against upstream config.

### F4 ✅ RESOLVED — `set_max_steps` minor diff (validation vs clamp)
Fork raises `ValueError` on invalid config; upstream uses `max(1, ...)`. Upstream is safer. Use upstream.

### F5 ✅ RESOLVED — `adv_estimator` upstream is a superset
Upstream adds `"agentic_reinforce"`. Smoke test uses `"grpo"` → no impact.

### F6 ✅ RESOLVED — Fork missing upstream features is a NON-ISSUE
Features like reward cluster, dynamic batching, `train_infer_correction` exist in upstream and are available when building on upstream ROLL. The ported `ConcurrentAgenticPipeline` benefits from them automatically.

### F7 ✅ RESOLVED — `position_ids` and `output_queue.put` — use upstream
Fork's 1-based `position_ids` and missing `env_id` in `output_queue.put` are fork drift. Use upstream's correct versions.

### F8 ✅ RESOLVED — Random sleep in `TrajEnvManager.step()` is dead code
Fork debug cruft. Don't port. Use upstream.

### F9 ✅ RESOLVED — `SharedStorage` logging instrumentation is nice-to-have
Minor observability enhancement. Can upstream as clean patch. Not blocking.

---

## Category G: Plan Quality Issues

### G1 ✅ RESOLVED — `close_admission` shrink-to-zero missing `routing_lock` mention
_Review #3_

**Fix:** Added explicit "Execute under `routing_lock`" language to the shrink-to-zero `close_admission(...)` path in the extraction plan's Admission Error Handling section.

### G2 🟢 P3 — Duplicate section numbering (multiple #### 3) and #### 4) sections)
_Review #4, #20_

**Fix:** Renumber Phase 3 subsections correctly.

### G3 🟢 P3 — Expand failure policy phase label inconsistency
_Review #5_

**Fix:** Note Phase 4 dependency in expand failure policy description.

### G4 🟡 P2 — SharedStorage detached cleanup
_Review #13, SERIOUS-19_

Detached actor persists across unregister cycles; port reservation keys accumulate. 

**Fix:** Document as service-mode follow-up. Add `delete_prefix` on unregister/teardown. Not blocking ENG-123.

---

## Category H: Critical Edge Cases & Implementation Gaps (Verified 2026-02-11)

### H1 ✅ DESIGN RESOLVED — Dead assertions in `_update_active_allocations`
**File:** `third_party/ROLL_multi_pipeline/roll/distributed/scheduler/centralized_gpu_scheduler.py`, lines 3545-3546

**Fact:** Code uses bare tuple expressions `len(...) > 0, "msg"` instead of `assert` statements. Python evaluates these as tuples and discards them, so the intended validation is a no-op.

**Impact:** If a buggy plan ever passes `gpus_to_allocate=[]` or `dp_ranks_to_add=[]`, the scheduler will proceed to construct an invalid expand-from-zero allocation state instead of failing fast.

**Fix:** Replace tuple expressions with real `assert` (or explicit `if ...: raise ValueError`) invariants.

### H2 ✅ DESIGN RESOLVED — `pipeline_id` parsing breaks on multi-word cluster names
**File:** `third_party/ROLL_multi_pipeline/roll/distributed/scheduler/centralized_gpu_scheduler.py`, lines 811, 952

**Fact:** `notify_completion` and `notify_cluster_released` use `cluster_id.rsplit("_", 1)[0]` to parse pipeline_id. This fails for clusters like `pipeline1_actor_infer` -> `pipeline1_actor` (wrong pipeline_id).
**Impact:** `PendingCompletionRequest` created with incorrect pipeline_id. Future registry lookups using this ID will fail.
**Fix:** Use `self._parse_cluster_id(cluster_id)` (already implemented in the same file) which handles known suffixes correctly.

### H3 ✅ DESIGN RESOLVED — `notify_completion` TOCTOU race (idempotency check outside lock)
**File:** `third_party/ROLL_multi_pipeline/roll/distributed/scheduler/centralized_gpu_scheduler.py`, lines 940-967

**Fact:** The idempotency check (`if cluster_id in self.pending_completion_requests`) and the dict insertion happen **without holding `self.lock`**.

**Impact:** Concurrent calls can race and overwrite each other, so “idempotent duplicate suppression” is not deterministic and earlier request state can be lost.

**Fix:** Wrap the full check-and-insert (and wakeup) block in `async with self.lock:`.

### H4 🟡 P2 — End-of-cycle guard logic broken (unconditional break)
**File:** `centralized_gpu_scheduler.py`, lines 1269-1278

**Fact:** The loop checking for active GENERATION workers breaks unconditionally on the first iteration (`break` not indented under `if`).
**Impact:** Stall detection logic (warning if no active workers) is non-functional.
**Fix:** Remove the misplaced break; implement proper `any()` check.

### H5 🟡 P2 — `_execute_expansions` partial failure orphans pending requests
**File:** `centralized_gpu_scheduler.py`, lines 3367-3380

**Fact:** If `asyncio.gather` raises (e.g., one worker OOMs), the loop exits. Pending requests (flagged `has_pending_request=True`) waiting for this batch are never signaled.
**Impact:** Callers waiting for GPUs hang forever.
**Fix:** Ensure signaling happens in a `finally` block or catch-and-signal pattern.

### H6 🟡 P2 — `unregister_pipeline` leaks state, causing crashes
**File:** `centralized_gpu_scheduler.py`, lines 982-993

**Fact:** Only removes the registry entry. Does not clean up `active_allocations`, `pending_requests`, or `pending_completion_requests`.
**Impact:** Scheduler crashes on next cycle when it tries to process these orphaned items and fails registry lookups.
**Fix:** Implement full state cleanup in `unregister_pipeline`.

### H7 🟡 P2 — Silent state inconsistency in shrink update
**File:** `centralized_gpu_scheduler.py`, lines 3509-3511

**Fact:** Step 2 of `_update_active_allocations` uses `if dp_rank in active_dp_ranks:` to skip missing ranks silently.
**Impact:** Masks bugs where scheduler plans to shrink a worker that is already inactive.
**Fix:** Change to `assert` to fail fast on state inconsistency.

### H8 🟡 P2 — Phase 1 result is discarded (Dead Code)
**File:** `centralized_gpu_scheduler.py`, line 1095

**Fact:** `_fifo_sorted_pending_and_active_cluster()` is called but result is ignored. Phase 3 recalculates its own snapshot.
**Impact:** Wasted CPU cycles, maintenance confusion. ~180 lines of dead code (`_dedup_and_query_timestamps` etc).
**Fix:** Remove the dead call and the unused methods.

### H9 🟡 P2 — Incorrect invariant assertion (Dead Code)
**File:** `centralized_gpu_scheduler.py`, line 1595

**Fact:** Dead method `_dedup_and_query_timestamps` asserts that `len(union) == len(A) + len(B)`, implying intersection must be empty. This is false (a cluster can be active AND have pending expansion).
**Fix:** Remove dead code.

### H10 🟢 P3 — `rebalance_on_expand` lock precondition not enforced
**File:** `generate_scheduler.py`

**Fact:** Docstring says "PRE-CONDITION: request_lock MUST be held". Private method `_rebalance_on_expand` assumes it. Public wrapper `rebalance_on_expand` calls it **without** acquiring the lock.
**Fix:** Add locking to the public wrapper `rebalance_on_expand`.

### H11 🟢 P3 — `_try_activate_one` commits shrinks before validation
**File:** `centralized_gpu_scheduler.py`, lines 2979-2988

**Fact:** Validation logic appends shrink ops and modifies state *before* the final "is this valid?" check. If it defaults to False (bail out), side effects remain.
**Fix:** Check eligibility before committing donor shrinks.

---

## Summary Statistics

| Severity | Count | Status |
|----------|-------|--------|
| 🔴 P0 (blocks start) | 0 | (none) |
| 🔴 P1 (blocks phase) | 0 | (none) |
| 🟡 P2 (defer-able) | 10 | A3, A4, C4, G4, H4, H5, H6, H7, H8, H9 |
| 🟢 P3 (cosmetic) | 5 | E3, G2, G3, H10, H11 |
| ✅ Resolved / Design Resolved | 25 | A1, B1-B3, C1-C3, D1-D3, E1-E2, F1-F9, G1, H1-H3 |
| **Total** | **40** | **0 blocking, 15 defer-able/cosmetic, 25 resolved** |

---

## Cross-Reference: Original Issue → Master ID

| Original ID | Master ID | Notes |
|-------------|-----------|-------|
| SERIOUS-1 | C4 | Infinite loop risk |
| SERIOUS-2 | F7 ✅ | Use upstream |
| SERIOUS-3 | F7 ✅ | Use upstream |
| SERIOUS-4 | C3 | Mode-aware validation |
| SERIOUS-5 | D2 | Signature mismatch |
| SERIOUS-6 | C2 | Phantom dependency |
| SERIOUS-7 | B2 | Server thread cleanup |
| SERIOUS-8 | B1 | Load partial signature + idempotency |
| SERIOUS-9 | A1 | Lock model + ordering (merged A1+A2) |
| SERIOUS-10 | F8 ✅ | Dead code |
| SERIOUS-11 | A1 | Public wrapper no lock (merged A1+A2) |
| SERIOUS-12 | A3 | notify_ready_to_release |
| SERIOUS-13 | B3 | teardown_collective_groups |
| SERIOUS-14/15 | F2 ✅ | agg_loss internal to workers |
| SERIOUS-16 | F1 ✅ | compute_advantage identical |
| SERIOUS-17 | F6 ✅ | Non-issue |
| SERIOUS-18 | E2 | pipeline_id parameter |
| SERIOUS-19 | F9 ✅ / G4 | Logging + cleanup |
| SERIOUS-20 | F3 ✅ | use_kl_loss exists in both |
| Review #1 | D1 | Bucket caching boundary |
| Review #2 | D2 | update_parameter_in_bucket |
| Review #3 | G1 | Plan: routing_lock mention |
| Review #4 | G2 | Plan: numbering |
| Review #5 | G3 | Plan: phase label |
| Review #11 | E1 | request_id dual-write |
| Review #12 | A1 | Shrink ordering (merged A1+A2) |
| Review #13 | G4 | SharedStorage cleanup |
| Review #14 | A1 | Public wrapper no lock (merged A1+A2) |
| Review #15 | E3 | DynamicSamplingScheduler |
| Review #16 | C3 | Validate ranks expand |
| Review #17 | D3 | send_recv_utils scope |
| Review #18 | C1 | ConcurrentAgenticPipeline decomp |
| Review #19 | A1 | Lock held during drain (merged A1+A2) |
| Review #20 | G2 | Duplicate numbering |
| Review #21 | B1 | load_states_partial sig |
| Review #22 | A1 | Locking doesn't exist (merged A1+A2) |
| Review #23 | B2 | offload thread cleanup |
| Review #24 | A4 | Abort retry backoff |
| Review #25 | A3 | notify_ready_to_release |
| Review #26 | C2 | active_checkpoint_version |
| NEW-1 | H1 | Dead assertions |
| NEW-2 | H2 | pipeline_id parsing |
| NEW-3 | H3 | notify_completion race |
| NEW-4 | H4 | End-of-cycle guard |
| NEW-5 | H5 | Expansion failure orphans |
| NEW-6 | H6 | Unregister leaks |
| NEW-7 | H7 | Silent shrink skip |
| NEW-8 | H8 | Phase 1 dead code |
| NEW-9 | H9 | Invariant dead code |
| NEW-10 | H10 | Lock precondition |
| NEW-11 | H11 | Try-activate side effects |
