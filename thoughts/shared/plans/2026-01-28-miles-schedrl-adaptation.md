# Miles SchedRL Adaptation Implementation Plan

**Status (2026-02-04)**: Deferred/archived. Current integration focus is **ROLL + SkyRL-train**. Kept for reference only.

## Overview

Adapt **Miles** (Phase 4 target) to the shared SchedRL protocol (`design_doc/multi-pipeline-adaptation-plan.md`) by adding: (1) DP-subset lifecycle control for rollout workers, (2) safe shrink/expand with **admission-close + abort + backend-confirmed ACK + retry**, (3) selective sync-on-resume (subset-scoped weight sync + staged SGLang memory resume), and (4) standardized `report_progress(...)` heartbeats (trajectory units, 2% bands).

**Stage 1 validation target: Retool only.** All SchedRL features in this plan must be implemented and validated using the Retool example first.

**Stage 2: SWE-agent integration.** After Stage 1 is complete, integrate the SWE-agent / Mini-SWE tool-loop as a separate stage.

**One-step-off only.** This plan targets the **one-step-off async** training driver (`third_party/miles/train_async.py`) and explicitly does **not**
support Miles’ “persistent background worker” fully-async rollout mode (`third_party/miles/examples/fully_async/`).

This plan assumes the centralized scheduler exists (or will exist) and focuses on **Miles-side** changes: MilesRouter + RolloutManager + SGLang rollout loop + minimal coordinator hooks.

**Important: weight activation cadence (Miles async).**
- `third_party/miles/train_async.py` updates rollout weights only every `--update-weights-interval` rollouts (the broadcast happens when `(rollout_id + 1) % update_weights_interval == 0`).
- Because `train_async.py` prefetches one “next rollout” early, it drains that prefetched rollout *before* calling `actor_model.update_weights()`; the prefetched rollout completes under the *previous* weights by design.
- **SchedRL requirement**: treat Miles’ `active_checkpoint_version` (rollout-side “which weights are active”) as advancing only on these interval boundaries. Do not assume “one-step-off” implies “weights change every step”.

## Current State Analysis

### What already exists (reusable)
- **Rollout engine lifecycle primitives (cluster-wide only)**:
  - `RolloutManager.offload()` calls `release_memory_occupation` on engines (`third_party/miles/miles/ray/rollout.py:176`).
  - `RolloutManager.onload(tags=...)` calls `resume_memory_occupation(tags=...)` (`third_party/miles/miles/ray/rollout.py:182`).
  - Staged resume helpers exist: `onload_weights()` / `onload_kv()` (`third_party/miles/miles/ray/rollout.py:191`, `third_party/miles/miles/ray/rollout.py:194`).
- **Global request dispatcher exists (MilesRouter)**, but only supports add/list workers:
  - Routes: `/add_worker`, `/list_workers`, catch-all proxy (`third_party/miles/miles/router/router.py:67`).
  - Least-connections routing via `worker_request_counts` (`third_party/miles/miles/router/router.py:227`).
- **Abort endpoint exists in SGLang**:
  - `/abort_request` accepts `rid` (targeted) and `abort_all` (engine-wide) (`third_party/sglang/python/sglang/srt/entrypoints/http_server.py:1136`).
  - Generation requests already support providing `rid` as part of `GenerateReqInput` (`third_party/sglang/python/sglang/srt/managers/io_struct.py:166`).
- **Retry buffer exists for aborted work**:
  - `RolloutDataSourceWithBuffer` stores groups for retry/oversampling (`third_party/miles/miles/rollout/data_source.py:157`).
- **Training→rollout weight sync plumbing exists**:
  - Trainer fetches `rollout_engines` + a lock from `RolloutManager.get_rollout_engines_and_lock()` (`third_party/miles/miles/backends/megatron_utils/actor.py:458`).
  - Distributed weight sync uses a rollout-engine lock to prevent NCCL deadlock (`third_party/miles/miles/backends/megatron_utils/update_weight/update_weight_from_distributed.py:225`).

### Gaps that block SchedRL
- **No subset lifecycle**: `RolloutManager.onload/offload` operate on all engines (`third_party/miles/miles/ray/rollout.py:176`).
- **No safe admission-close**:
  - MilesRouter has no disable/remove API, and it may keep routing to a worker while the engine is trying to drain/flush (`third_party/miles/miles/router/router.py:227`, `third_party/miles/miles/backends/sglang_utils/sglang_engine.py:275`).
- **SGLangEngine.shutdown() calls a router endpoint MilesRouter does not implement**:
  - It calls `/remove_worker` when `args.use_miles_router` (`third_party/miles/miles/backends/sglang_utils/sglang_engine.py:298`).
- **Abort+retry is not implemented for multi-turn examples**:
  - Base Miles SGLang rollout abort path is global (`abort_all=True` to all workers) (`third_party/miles/miles/rollout/sglang_rollout.py:282`).
  - Examples generally treat abort as terminal (e.g. Retool and others).
- **No SchedRL-standard progress heartbeats**:
  - No `report_progress(queued_trajectories, inflight_trajectories, percent_completed, oldest_unfinished_creation_ts, ...)`.

## Desired End State

After this plan:
1) **Subset lifecycle** exists for rollout DP workers:
   - `onload(worker_indices=...)`, `offload(worker_indices=...)`, `onload_weights(worker_indices=...)`, `onload_kv(worker_indices=...)`.
2) **Shrink/expand is safe** for stateless rollouts:
   - Shrink ordering: `close_admission(indices)` → `abort_inflight(indices)` → **wait for ACK** → `offload(indices)`.
   - Retry semantics: aborted work is re-queued and retried (bounded engine-error retries; unbounded preemption retries).
3) **Selective sync-on-resume**:
   - Resume ordering per subset: `onload_weights(indices)` → `trainer.update_weights(indices)` → `onload_kv(indices)` → resume generation.
4) **Progress heartbeats** are emitted:
   - At batch start and every time `percent_completed` crosses a 2% band (trajectory units).
5) **Stage 2 tool-loop baseline**:
   - After Stage 1, integrate SWE-agent with a conservative shrink policy (drain-only by default) unless/until explicit idempotency is implemented.

### SchedRL-facing contract (Miles adapter)
Expose an Adapter surface that matches the Final Plan:
- `close_admission(worker_indices, action_id, activation_epoch) -> ActionResponse`
- `open_admission(worker_indices, action_id, activation_epoch) -> ActionResponse`
- `shrink_workers(worker_indices, action_id, activation_epoch) -> ActionResponse`
- `expand_workers(worker_indices, base_version, action_id, activation_epoch) -> ActionResponse`

Mapping notes:
- `close_admission/open_admission` are implemented as router admission control for the subset (disable/enable routing).
- `shrink_workers` performs: close admission → abort → wait ACK → offload subset, and MUST fully release GPU memory.
 - `expand_workers` performs: onload subset → sync to `base_version` (if needed) → return; scheduler calls `open_admission(...)` separately.

Registration invariant (State Reset on Registration):
- On (re)registration, assume `S_actual={}` and release/kill any leftover engines from a prior session.

## What We're NOT Doing

- No Ray placement-group resizing or true GPU “reclaim”; placement is fixed per run.
- No “resume inside a turn” migration; only **turn/request-level** abort+retry.
- No idempotency keys / dedupe for side-effectful tools (explicit backlog item).
- No partial-rollout continuation when using MilesRouter (explicitly blocked by Miles middleware assertions).
- No Miles “fully async” persistent background rollout worker (`third_party/miles/examples/fully_async/`).

## Implementation Approach

Use Miles’ existing structure:
- Keep the core **one-step-off** training loop structure; add **thin hooks** for scheduler request/release at phase boundaries (`third_party/miles/train_async.py:32`).
- Add a **proxy/adapter** layer that:
  - exposes subset ops to the centralized scheduler,
  - emits release ACKs,
  - does not make scheduling decisions itself.
- Make MilesRouter the authoritative place for **admission control** and (optionally) **inflight RID accounting**.

The plan prefers enabling `args.use_miles_router=True` for SchedRL runs, because we can extend MilesRouter to:
- safely disable routing without deleting worker state while in-flight requests exist, and
- track per-worker in-flight request IDs for ACK waiting.

One-step-off constraint (why this is feasible):
- `train_async.py` keeps at most one “next rollout” in flight and **drains it before weight update** (`third_party/miles/train_async.py:63-68`), giving
  the coordinator a clean stop point for SchedRL-controlled time sharing.

---

## Phase 1: MilesRouter Admission Control + ACK Plumbing

### Overview
Make admission-close implementable and safe (no dict deletion while requests are in flight), and provide a robust ACK signal for “requests drained/aborted” on a per-worker basis.

### Changes Required

#### 1) Add worker state: enabled/disabled vs dead
**File**: `third_party/miles/miles/router/router.py`
**Changes**:
- Add `disabled_workers: set[str]` distinct from `dead_workers`.
- Update `_use_url()` to exclude `disabled_workers` (and `dead_workers`) from routing pool, while keeping their counters intact for `_finish_url`.
- Extend `list_workers()` to return structured status per worker (enabled/disabled/dead + active_count).

#### 2) Add control-plane endpoints
**File**: `third_party/miles/miles/router/router.py`
**Changes**:
- Implement:
  - `POST /disable_worker?url=...` (idempotent; moves url into `disabled_workers`)
  - `POST /enable_worker?url=...` (idempotent; removes url from `disabled_workers` and `dead_workers` only if explicitly desired)
  - `POST /remove_worker?url=...` (only succeeds when `worker_request_counts[url]==0`; otherwise returns 409/400)
- Maintain backwards compatibility with the shutdown path in `SGLangEngine.shutdown()` (`third_party/miles/miles/backends/sglang_utils/sglang_engine.py:298`).
  - Concretely: support the existing call form `POST /remove_worker?url=http://host:port` (query param), not a JSON body.

#### 3) (Recommended) Track inflight request IDs by worker for ACK waiting
**File**: `third_party/miles/miles/router/router.py`
**Changes**:
- When proxying requests:
  - Parse JSON request bodies when present.
  - If the payload contains `rid`, maintain:
    - `inflight_rids_by_worker[url]: set[rid]`
    - `worker_by_rid[rid]: url`
  - On response completion (in `finally`), remove rid from inflight sets.
- Add `GET /inflight_rids?url=...` and/or include inflight rid counts in `list_workers`.

**Why**: This makes “wait for abort ACK” implementable without reaching into internal asyncio task state: ACK is “no inflight rid remains on that disabled worker” within timeout.

### Success Criteria

#### Automated Verification
- [ ] `cd third_party/miles && pytest -q` (baseline; no new tests added in this phase)

#### Manual Verification
- [ ] Start MilesRouter and register workers; disable one and confirm it receives no new requests but can still finish in-flight requests without assertion failures.
- [ ] `SGLangEngine.shutdown()` no longer fails due to missing `/remove_worker`.

---

## Phase 2: RolloutManager Subset Lifecycle + URL Mapping

### Overview
Add subset-scoped control of rollout engines and a stable mapping between “worker index” and router worker URL.

### Changes Required

#### 1) Add subset parameters to lifecycle methods
**File**: `third_party/miles/miles/ray/rollout.py`
**Changes**:
- Extend:
  - `offload(worker_indices: list[int] | None = None)`
  - `onload(tags: list[str] | None = None, worker_indices: list[int] | None = None)`
  - `onload_weights(worker_indices: list[int] | None = None)`
  - `onload_kv(worker_indices: list[int] | None = None)`
- Define `worker_indices` as indices into the logical engine list `rollout_engines` (`third_party/miles/miles/ray/rollout.py:128`).
- Confirmed from source: subset filtering for rollout lifecycle should live in `RolloutManager` only (no `ActorGroup` changes needed for Phase 2), because `RolloutManager.offload/onload` iterate `self.rollout_engines` directly (`third_party/miles/miles/ray/rollout.py:127-131`, `third_party/miles/miles/ray/rollout.py:176-189`) and `create_rollout_manager()` instantiates `RolloutManager` directly (`third_party/miles/miles/ray/placement_group.py:169-174`).

#### 2) Record stable engine URLs
**File**: `third_party/miles/miles/ray/rollout.py`
**Changes**:
- During engine init (where host/port are allocated), store `engine_urls` aligned with `rollout_engines`.
- Expose a method to query `engine_urls` (for the proxy/scheduler to disable/enable specific workers).

#### 3) Add router admission helpers at the RolloutManager level
**File**: `third_party/miles/miles/ray/rollout.py`
**Changes**:
- Add helper methods that call router control-plane endpoints (from Phase 1) for a given subset of indices:
  - `close_admission(worker_indices, action_id, activation_epoch)`
  - `open_admission(worker_indices, action_id, activation_epoch)`
  - (optional) `wait_inflight_drained(worker_indices, timeout_s)`

### Success Criteria

#### Automated Verification
- [ ] `cd third_party/miles && pytest -q`

#### Manual Verification
- [ ] In a small local run, disable a subset and confirm routing avoids it; then offload only that subset and confirm other workers stay responsive.

---

## Phase 3: Deterministic `rid` + Abort/ACK/Retry Semantics (Stateless Rollouts)

### Overview
Implement SchedRL’s required migration policy (`REQUEST_RETRY`) for mid-flight shrink on **stateless** rollouts (no external side effects), using:
- admission-close (router disable),
- abort (SGLang `/abort_request`),
- ACK wait (router inflight rid tracking),
- retry (requeue aborted work).

### Changes Required

#### 1) Plumb deterministic `rid` into SGLang generate calls
**File**: `third_party/miles/miles/rollout/sglang_rollout.py`
**Changes**:
- When posting to `/generate`, include a deterministic `rid` in the payload.
- Define `rid` as `"{trajectory_id}:{turn_id}:{attempt}"`.
  - `trajectory_id`: stable per sample/trajectory (e.g., `Sample.index` or an explicit UUID stored in `Sample.metadata`).
  - `turn_id`: stable per multi-turn step.
  - `attempt`: increment only on engine error retries (not on preemption retries).

**Notes**:
- SGLang accepts `rid` for generate requests via `GenerateReqInput.rid` (`third_party/sglang/python/sglang/srt/managers/io_struct.py:166`).

#### 2) Add retry-on-abort loop at the rollout layer (not training)
**File**: `third_party/miles/miles/rollout/sglang_rollout.py`
**Changes**:
- In `generate_rollout_async(...)`, when a group finishes and any sample has `finish_reason.type == "abort"`:
  - treat it as **preempted**, not terminal,
  - re-queue the entire group back into `RolloutDataSourceWithBuffer` (`third_party/miles/miles/rollout/data_source.py:187`),
  - do not count it toward `data` (the training batch),
  - continue until the per-step target is satisfied.
- Add a configurable cap for **engine error** retries per `(trajectory_id, turn_id)` (default 3).
  - Preemption retries are not capped.

#### 3) Implement shrink flow hooks for Miles
**Files**:
- `third_party/miles/miles/ray/rollout.py`
- `third_party/miles/miles/router/router.py`
**Changes**:
- Define the shrink procedure used by the proxy:
  1) `close_admission(indices)`
  2) `POST {engine_url}/abort_request {"abort_all": true}` for those indices (or targeted `rid` abort if available)
  3) Wait for ACK:
     - `wait_inflight_drained(indices, timeout_s)` via MilesRouter inflight tracking
     - if timeout: **fail fast** (crash pipeline; do not proceed to offload)
  4) `offload(indices)`

### Success Criteria

#### Automated Verification
- [ ] `cd third_party/miles && pytest -q`

#### Manual Verification
- [ ] Force a shrink during an in-progress rollout batch and confirm:
  - affected requests return `finish_reason.type == "abort"`,
  - aborted groups are retried and eventually complete,
  - shrink only proceeds after ACK (no in-flight requests on the disabled workers).

---

## Phase 4: Selective Sync-on-Resume (Subset-Scoped Weight Sync)

### Overview
Make resuming a subset cheap and correct: only the active subset loads weights + receives the latest weights broadcast, then loads KV/cache.

### Changes Required

#### 1) Make RolloutManager expose subset engine list for weight sync
**File**: `third_party/miles/miles/ray/rollout.py`
**Changes**:
- Extend `get_rollout_engines_and_lock(worker_indices=None)` to return only the subset engine handles.

#### 2) Add optional subset parameter to trainer-side `update_weights`
**Files**:
- `third_party/miles/miles/ray/actor_group.py`
- `third_party/miles/miles/backends/megatron_utils/actor.py`
**Changes**:
- Allow `RayTrainGroup.update_weights(worker_indices=None)` to pass a subset down to the underlying trainer actors.
- In `MegatronTrainRayActor.update_weights()` (or equivalent), call `rollout_manager.get_rollout_engines_and_lock(worker_indices=...)` instead of always syncing all engines (`third_party/miles/miles/backends/megatron_utils/actor.py:458`).

#### 3) Define resume ordering
**Files**:
- `third_party/miles/train.py`
- `third_party/miles/train_async.py`
**Changes**:
- For SchedRL-controlled resume of generation subset `A`:
  - `rollout_manager.onload_weights(A)`
  - `actor_model.update_weights(A)`
  - `rollout_manager.onload_kv(A)`
  - enable admission for `A`

### Success Criteria

#### Automated Verification
- [ ] `cd third_party/miles && pytest -q`

#### Manual Verification
- [ ] After shrinking to subset `P` and later expanding to `A`, confirm only `A` does staged resume + weight sync, and generation succeeds with the latest weights.

---

## Phase 5: `report_progress(...)` Heartbeats (2% Bands, Trajectory Units)

### Overview
Expose standardized rollout progress to the centralized scheduler so it can make fair, non-thrashy decisions.

### Changes Required

#### 1) Add creation timestamp tracking for oldest unfinished
**File**: `third_party/miles/miles/rollout/sglang_rollout.py`
**Changes**:
- When a group is first created/submitted, attach `creation_ts` into each sample’s metadata and preserve it across retries.
- Compute `oldest_unfinished_creation_ts` as the min creation_ts among unfinished groups (queued via buffer + in-flight tasks).

#### 2) Emit progress at batch start + 2% bands
**File**: `third_party/miles/miles/rollout/sglang_rollout.py`
**Changes**:
- Define:
  - `target_trajectories = args.rollout_batch_size * args.n_samples_per_prompt`
  - `completed_trajectories = completed_groups * args.n_samples_per_prompt`
  - `inflight_trajectories = inflight_groups * args.n_samples_per_prompt` (where `inflight_groups = len(state.pendings)` in group units)
  - `queued_trajectories = max(target_trajectories - completed_trajectories - inflight_trajectories, 0)`
  - `percent_completed = completed_trajectories / max(target_trajectories, 1)`
- Call `scheduler.report_progress(...)`:
  - once at batch start,
  - then whenever `percent_completed` crosses an additional 2% boundary,
  - immediately when the denominator window changes (e.g., when drops/retries change the target accounting).

### Success Criteria

#### Automated Verification
- [ ] `cd third_party/miles && pytest -q`

#### Manual Verification
- [ ] During a rollout batch, confirm the scheduler receives progress events with monotonic `percent_completed` and sensible queued/inflight values.

---

## Phase 6: Stage 1 Validation Run (Retool)

### Overview
Validate **all** Stage 1 SchedRL features end-to-end on Retool:
- subset lifecycle (onload/offload),
- admission control + abort + ACK + retry migration,
- selective sync-on-resume,
- `report_progress(...)` heartbeats.

### Validation Target
- Retool: `third_party/miles/examples/retool/generate_with_retool.py`

### Success Criteria

#### Automated Verification
- [ ] `cd third_party/miles && pytest -q`

#### Manual Verification
- [ ] Retool baseline sanity: run once with SchedRL disabled and confirm the example completes (no router/rollout regressions).
- [ ] Forced shrink mid-rollout causes abort+ACK+retry and eventually completes the batch.
- [ ] Expand/resume performs subset-scoped staged resume + weight sync and generation succeeds.
- [ ] Scheduler receives `report_progress(...)` events at batch start + 2% bands (trajectory units).
- [ ] Stage 1 run uses the one-step-off driver (`third_party/miles/train_async.py`), not the fully-async background worker example.
- [ ] Stage 1 run explicitly documents/sets `--update-weights-interval` and treats it as the rollout-weight activation cadence (the proxy/scheduler must not assume per-step activation).

---

## Phase 7: Stage 2 Integration (SWE-agent / Mini-SWE)

### Overview
After Stage 1 is stable on Retool, integrate the SWE-agent tool-loop example and validate that SchedRL controls do not cause duplicated side effects.

### Integration Target
- SWE-agent: `third_party/miles/examples/experimental/swe-agent/generate_with_swe_agent.py`

### Baseline Shrink Policy (Stage 2)
- Default to **drain-only** shrink (close admission, wait for in-flight to finish, then offload).
- Do **not** enable abort+retry for tool-loop rollouts unless idempotency / resume guarantees are implemented.

### Success Criteria

#### Automated Verification
- [ ] `cd third_party/miles && pytest -q`

#### Manual Verification
- [ ] Shrink closes admission and waits for completion; no repeated tool side effects are observed.
- [ ] Stage 2 run uses the one-step-off driver (`third_party/miles/train_async.py`), not the fully-async background worker example.

---

## Testing Strategy

### Unit / Component
- Prefer minimal direct tests under Miles’ existing pytest setup (no new test suite introduced).

### Integration
- Run one short training loop with SchedRL hooks enabled and a tiny batch size; force shrink/expand events.

### Manual Testing Steps
1) Start a Miles run with `use_miles_router=true`.
2) Force-disable 1 worker and trigger shrink (router disable + abort + ACK + offload).
3) Expand a different subset, run selective resume (weights → sync → kv).
4) Confirm progress events and release ACK ordering.

## Performance Considerations
- Avoid thrash by using admission-close before abort/offload.
- Prefer engine-wide abort (`abort_all=true`) for shrink-to-zero / hard preemption; use targeted `rid` abort for rebalance or partial shrink where available.
- Keep weight sync subset-scoped on resume to reduce broadcast overhead.

## Migration Notes
- Existing Miles runs without SchedRL should remain unchanged unless `schedrl.enabled=true` is set.
- SchedRL-enabled runs should require `use_miles_router=true` to guarantee admission control and ACK semantics.

## References
- Design: `design_doc/archive/adaptation_miles.md`
- Protocol: `design_doc/multi-pipeline-adaptation-plan.md`
- Research: `thoughts/shared/research/2026-01-28-schedrl-framework-mechanisms.md`
- Research: `thoughts/shared/research/2026-01-28-schedrl-adaptation-research.md`
- Rollout manager: `third_party/miles/miles/ray/rollout.py`
- MilesRouter: `third_party/miles/miles/router/router.py`
- SGLang abort: `third_party/sglang/python/sglang/srt/entrypoints/http_server.py`
- Weight sync (Megatron): `third_party/miles/miles/backends/megatron_utils/actor.py`
