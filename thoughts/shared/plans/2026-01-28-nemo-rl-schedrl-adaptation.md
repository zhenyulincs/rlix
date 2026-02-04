# NeMo-RL (GRPO + vLLM) SchedRL Adaptation Implementation Plan

**Status (2026-02-04)**: Deferred/archived. Current integration focus is **ROLL + SkyRL-train**. Kept for reference only.

## Overview

Adapt **NeMo-RL’s GRPO pipeline** to the shared SchedRL protocol (`design_doc/multi-pipeline-adaptation-plan.md`) by adding: (1) DP-subset lifecycle control (shrink/expand), (2) deterministic per-turn vLLM request IDs + safe abort/retry migration (**targeted abort + backend-confirmed ACK from day one**), (3) **selective/cheap sync + faster wake/sleep behaviors from day one**, and (4) standardized `report_progress(...)` heartbeats.

Default protocol settings for NeMo-RL Phase 2:
- `update_policy = INFLIGHT` when async training is enabled (`vllm_cfg.async_engine && grpo.async_grpo.enabled`), else `QUIESCE-by-drain`
- Runtime requirement (fail fast): when async training is enabled, assert `grpo.async_grpo.in_flight_weight_updates == true` at startup (do not allow async training with `in_flight_weight_updates=false`).
- Note on naming: NeMo-RL “async-1off” configs mean `max_trajectory_age_steps=1` (bounded staleness), not a strict “one-step-off pipelining” guarantee.
- `migration_policy = REQUEST_RETRY` (abort the current turn/request, retry elsewhere; two-phase commit invariant required)
- `expand_rebalance_policy = REBALANCE_QUEUED` (queued-only day one; in-flight rebalance requires cooperative cancellation, Phase 2+)

Primary design references:
- `design_doc/archive/adaptation_nemo_rl.md`
- `design_doc/multi-pipeline-adaptation-plan.md`
- `thoughts/shared/research/2026-01-28-schedrl-framework-mechanisms.md`
- `thoughts/shared/research/2026-01-28-schedrl-adaptation-research.md`

---

## Current State Analysis

### Coordinator and lifecycle today

- Sync GRPO (`grpo_train(...)`): cluster-wide generation boundaries use `policy_generation.prepare_for_generation()` / `finish_generation()` and weight sync via `refit_policy_generation(...)`. (`third_party/nemo-rl/nemo_rl/algorithms/grpo.py:917`, `third_party/nemo-rl/nemo_rl/models/generation/vllm/vllm_generation.py:691`)
- Async GRPO: rollout collection is performed by `AsyncTrajectoryCollector` with admission gating via `_refit_pause_cleared`. (`third_party/nemo-rl/nemo_rl/algorithms/async_utils.py:238`, `third_party/nemo-rl/nemo_rl/algorithms/grpo.py:2454`)

### Subset lifecycle (missing)

- `VllmGeneration.prepare_for_generation()` / `finish_generation()` call `RayWorkerGroup.run_all_workers_single_data(...)` (all workers). (`third_party/nemo-rl/nemo_rl/models/generation/vllm/vllm_generation.py:691`, `third_party/nemo-rl/nemo_rl/models/generation/vllm/vllm_generation.py:715`)
- `RayWorkerGroup` already has single-worker execution (`run_single_worker_single_data(...)`), but no subset filtering in `run_all_workers_single_data(...)`. (`third_party/nemo-rl/nemo_rl/distributed/worker_groups.py:631`, `third_party/nemo-rl/nemo_rl/distributed/worker_groups.py:755`)

### Request IDs + targeted abort (missing)

- Async vLLM worker generates `request_id = uuid4()` internally, so the coordinator cannot do deterministic per-turn IDs required by SchedRL. (`third_party/nemo-rl/nemo_rl/models/generation/vllm/vllm_worker_async.py:747`, `third_party/nemo-rl/nemo_rl/models/generation/vllm/vllm_worker_async.py:920`)
- No exposed “abort these request_ids on this subset” surface in NeMo-RL’s vLLM generation layer.

### Dispatching (needs extension for rebalance)

- Async vLLM dispatch round-robins DP leaders (`current_generate_dp_shard_idx`). (`third_party/nemo-rl/nemo_rl/models/generation/vllm/vllm_generation.py:549`)
- Outputs are tagged with `gen_leader_worker_idx` which can be used for local load accounting. (`third_party/nemo-rl/nemo_rl/models/generation/vllm/vllm_generation.py:578`)

### Progress heartbeats (missing)

- NeMo-RL does not report SchedRL-standard `report_progress(...)`. `AsyncTrajectoryCollector` has enough internal state to compute it but lacks:
  - explicit queued-vs-inflight bookkeeping and
  - enqueue timestamps for `oldest_unfinished_creation_ts`. (`design_doc/archive/adaptation_nemo_rl.md:122`, `design_doc/multi-pipeline-adaptation-plan.md:214`)

---

## Desired End State

SchedRL-critical properties:
- **DP-subset lifecycle**: `expand_workers(worker_indices)` / `shrink_workers(worker_indices)` to wake/sleep only a subset of rollout DP shards.
- **Shrink migration**: implement `REQUEST_RETRY` with strict ordering:
  - Close admission → Abort(P) → Wait abort ACK → Sleep/Offload(P) → Return.
- **Deterministic request IDs**: coordinator-provided `request_id` per turn so targeted abort is possible.
- **Progress reporting**: `report_progress(queued_trajectories, inflight_trajectories, percent_completed, oldest_unfinished_creation_ts, ...)` at batch start + 2% bands, trajectory units.
- **Day-one performance features**:
  - Smarter expand rebalance: queued-first (`REBALANCE_QUEUED`).
  - More selective/cheap sync + faster wake/sleep: sync-on-expand only for newly activated workers (where possible), and use tags/levels to minimize work.

### SchedRL-facing contract (NeMo-RL adapter)
Expose an Adapter surface that matches the Final Plan:
- `close_admission(worker_indices, action_id, activation_epoch) -> ActionResponse`
- `open_admission(worker_indices, action_id, activation_epoch) -> ActionResponse`
- `shrink_workers(worker_indices, action_id, activation_epoch) -> ActionResponse`
- `expand_workers(worker_indices, base_version, action_id, activation_epoch) -> ActionResponse`

Mapping notes:
- `close_admission`: block new prompt-group starts for `worker_indices` (do not enqueue new work onto those shards).
- `open_admission`: resume prompt-group starts for `worker_indices`.
- `shrink_workers`: abort in-flight for `worker_indices`, wait for abort ACK, then sleep/offload at **Level 2 (weights+KV)** so GPU memory is fully released.
 - `expand_workers`: wake subset, sync to `base_version` if needed, then let the scheduler call `open_admission(...)`.

Registration invariant (State Reset on Registration):
- On (re)registration, assume `S_actual={}` and release/kill any leftover servers from a prior session.

---

## What We’re NOT Doing

- No Ray placement group resizing at runtime (fixed placement).
- No `PARTIAL_ROLLOUT_MIGRATE` (resume tokens / partial completions).
- No new external libraries or packages.
- No new test suites from scratch; use existing NeMo-RL tests/examples and add the smallest possible validation harness only if absolutely necessary.

---

## Implementation Approach

Principles:
- Smallest change that meets the shared protocol.
- Fail fast: if abort ACK does not arrive within timeout, crash the pipeline; do not proceed with shrink/offload.
- Reuse NeMo-RL’s native execution units:
  - async collector schedules **prompt-group threads**; we migrate by retrying the **current turn/request**.
- Day-one performance goals are achieved primarily by:
  - minimizing unnecessary wake/sleep/sync calls (subset-scoped where possible),
  - smarter routing/rebalance decisions based on local load,
  - avoiding “sync work that will be immediately preempted” via merged shrink/expand sequencing.

---

## Scheduler “Proxy” Boundary (Conceptual, Thin)

Keep a clear boundary between “NeMo-RL internals” and “SchedRL protocol wiring”, but do not introduce a thick proxy that owns scheduling decisions.

Design:
- The coordinator continues to call the scheduler client directly for `request_gpus/release_gpus`, and implements `expand_workers/shrink_workers` semantics.
- A thin wrapper/adapter layer (optional) is allowed to centralize:
  - `notify_cluster_released(...)` after `sleep(worker_indices=...)` completes,
  - strict shrink ordering enforcement (close admission → abort+ACK → sleep),
  - the `report_progress(...)` emission surface (collector → scheduler client).
- The wrapper must be unidirectional: forwards to NeMo-RL components, adds ACK/telemetry, and must not choose versions, policies, or preemption targets.

Implementation note:
- Prefer implementing this boundary as a small helper object inside an existing module (or a few functions) rather than adding a new `schedrl/` package up front.

---

## Phase 0: Runnable Reference Workload (Async + Multi-turn)

Goal: a reproducible job that exercises async multi-turn rollouts, refit boundaries, and resume. This is required to validate shrink/expand safely.

Base:
- Script: `third_party/nemo-rl/examples/run_grpo_sliding_puzzle.py`
- Config: `third_party/nemo-rl/examples/configs/grpo_sliding_puzzle.yaml`
- Async reference: `third_party/nemo-rl/docs/guides/async-grpo.md`

Changes:
- Add one new config (or a small run script wrapper) enabling:
  - async GRPO rollout collection, and
  - async vLLM engine.

Success Criteria:
- Manual: run the example end-to-end and observe it reaches: generation → training → refit → generation again.
- Automated (sanity): `cd third_party/nemo-rl && uv run --group test pytest -q`

---

## Phase 1: DP-Subset Lifecycle (Wake/Sleep Only Selected Workers)

### 1.1 Add subset filtering to worker-group helpers

File: `third_party/nemo-rl/nemo_rl/distributed/worker_groups.py`
- Extend `run_all_workers_single_data(...)` to accept `worker_indices: list[int] | None = None`.
- Behavior:
  - `worker_indices is None`: current behavior (all workers).
  - else: iterate only the specified indices and still apply `run_rank_0_only_axes` filtering.

### 1.2 Add subset lifecycle surface on vLLM generation

File: `third_party/nemo-rl/nemo_rl/models/generation/vllm/vllm_generation.py`
- Add:
  - `wake(worker_indices: list[int], *, tags: list[str] | None = None) -> bool`
  - `sleep(worker_indices: list[int], *, level: int = 1) -> bool`
- Implement by calling `run_all_workers_single_data(... worker_indices=...)` with:
  - `wake_up(_async)` and pass-through `tags` (reusing existing vLLM wake tags support)
  - `sleep(_async)` and pass-through `level` (extend worker method signatures if needed)

Day-one performance requirement (faster wake/sleep):
- Expose sleep `level` and wake `tags` end-to-end so the scheduler can choose:
  - “cheap sleep” (drop KV only) vs “deep sleep” (drop weights too), and
  - weights-only wake vs KV-only wake.

Success Criteria:
- Manual: sleeping a subset does not stop other DP shards; waking subset restores readiness.

---

## Phase 2: REQUEST_RETRY Migration + Smarter Rebalance (Day One)

### 2.1 Coordinator-provided deterministic `request_id` (required)

Files:
- `third_party/nemo-rl/nemo_rl/experience/rollouts.py`
- `third_party/nemo-rl/nemo_rl/experience/rollouts.py` (async turn generation helpers)
- `third_party/nemo-rl/nemo_rl/models/generation/vllm/vllm_worker_async.py`

Decisions:
- `trajectory_id`: stable id for the sample (per rollout item), derived from the sample’s original idx plus a per-run prefix (must remain stable across retries).
- `turn_id`: the multi-turn index inside `run_sample_multi_turn_rollout(...)`.
- `request_id = f\"{trajectory_id}:{turn_id}:{attempt}\"` where:
  - `attempt` increments only for engine-error retries,
  - preemption retries reuse the same `attempt` semantics but do not cap retries (SchedRL rule).

Implementation:
- Pass `request_id` through the async generation call chain via `BatchedDataDict` (e.g., `request_ids: list[str]` aligned with batch rows).
- In `vllm_worker_async`, use the provided `request_id` instead of generating `uuid4()`.

### 2.2 Targeted abort + backend-confirmed ACK (required)

File: `third_party/nemo-rl/nemo_rl/models/generation/vllm/vllm_worker_async.py`
- Track in-flight request IDs per worker (async-safe).
- Add:
  - `abort_requests(request_ids: list[str]) -> bool` (or returns structured info)
  - `abort_all_inflight() -> bool`
- ACK semantics (SchedRL): return only when the worker confirms those request_ids are no longer in-flight.
- Timeout fail-fast: if abort does not converge within timeout, raise an exception (crash pipeline).

File: `third_party/nemo-rl/nemo_rl/models/generation/vllm/vllm_generation.py`
- Add coordinator-facing method:
  - `abort_requests(worker_indices: list[int], request_ids: list[str]) -> None`
  - This targets only the shrinking/overloaded subset.

### 2.3 Retry queue at NeMo-RL’s native unit (prompt-group)

File: `third_party/nemo-rl/nemo_rl/algorithms/async_utils.py`

Add explicit tracking so we can implement both shrink and expand rebalance correctly:
- A prompt-group “work item” structure containing:
  - `repeated_batch` (or a light reference needed to regenerate),
  - `trajectory_ids`/`request_ids` for the current turn(s),
  - `enqueue_ts` (stable across retries),
  - retry counters for engine errors.

Implement:
- `queued_prompt_groups` as “not yet started” work items.
- `inflight_prompt_groups` as those currently executing in `_inflight_threads`.
- On abort/cancel for shrink or expand-rebalance:
  - re-enqueue the work item (preserve `enqueue_ts`), and retry on a non-preempted shard.

### 2.4 Smarter rebalance on expand (Phase 2+; queued-only day one)

Expand behavior:
- Day one: rebalance queued/not-started work onto newly expanded workers first (`REBALANCE_QUEUED`).
- Phase 2+: optional in-flight migration requires cooperative cancellation in the rollout loop (see Final Plan review finding).

Stop condition (from shared protocol):
- Let `load[dp] = queued_trajectories_by_worker[dp] + inflight_trajectories_by_worker[dp]`.
- Stop when `(max(load) - min(load)) / step_target_trajectories <= 0.05`.

Implementation notes:
- Use deterministic request IDs so abort+retry does not double-apply env/tool side effects.
- Prefer aborting turns with the smallest “remaining work” estimate first (proxy: smallest generated token count so far, or newest turns).

Success Criteria:
- Expand causes new work to land on expanded subset quickly.
- Optional in-flight rebalance reduces tail latency without breaking safety (abort ACK + retry semantics).

---

## Phase 3: Selective/Cheap Sync + Sync-on-Expand (Day One)

Goal: minimize sync work and wake/sleep overhead on day one, while staying correct.

### 3.1 “Selective” at the coordinator level (always enabled)

File: `third_party/nemo-rl/nemo_rl/algorithms/grpo.py`
- Ensure the scheduler/coordinator sequencing avoids waste:
  - Do not sync a worker that is about to be shrunk.
  - When an expand is scheduled, sync only after the expand decision is final (scheduler re-validates `active_checkpoint_version` immediately before expand).
- Coalesce multiple pending version intents (“highest version wins”; never abort a running sync), per shared protocol.

### 3.2 Cheap wake/sleep + cheap sync primitives (subset-scoped where possible)

Files:
- `third_party/nemo-rl/nemo_rl/models/generation/vllm/vllm_generation.py`
- `third_party/nemo-rl/nemo_rl/models/generation/vllm/vllm_worker.py`
- `third_party/nemo-rl/nemo_rl/models/generation/vllm/vllm_worker_async.py`

Implement day-one knobs:
- Wake tags: `tags=["weights"]`, `tags=["kv_cache"]` (already supported by vLLM v1 APIs; make it scheduler-selectable).
- Sleep levels:
  - `shrink_workers(...)` MUST use `level=2` (weights+KV) so GPU memory is fully released for time-sharing.
  - `level=1` is only allowed for non-time-sharing internal pauses (not used for SchedRL shrink).

Sync-on-expand:
- For expanded subset `A`:
  - `wake(A, tags=["weights"])`
  - perform refit/sync for the current `active_checkpoint_version`
  - `wake(A, tags=["kv_cache"])` (optional; can be deferred until just before admission opens if safe)

Note on non-colocated inference:
- Day one still uses the existing collective sync path for correctness; “cheap” here means coalescing + avoiding unnecessary syncs and avoiding syncing workers that will be shrunk immediately.

Success Criteria:
- Expand does not trigger unnecessary full-cluster wake/sleep; newly activated workers are made ready with minimal work.

---

## Phase 4: Progress Heartbeats (`report_progress`) + Timestamps

File: `third_party/nemo-rl/nemo_rl/algorithms/async_utils.py`

### 4.1 Correct units (trajectory counts)

Per shared mapping:
- `queued_trajectories = queued_prompt_groups * num_generations_per_prompt`
- `inflight_trajectories = inflight_prompt_groups * num_generations_per_prompt`
- `step_target_trajectories = train_global_batch_size` (NeMo-RL’s per-step rollout target)

### 4.2 `oldest_unfinished_creation_ts` (enqueue timestamps; day-one correctness)

Requirements:
- Track enqueue time at submission time, not completion time.
- Preserve enqueue time across retries (`REQUEST_RETRY` rule).

### 4.3 Cadence

Emit:
- at batch start, and
- on every 2% progress band crossing, and
- when `percent_completed >= 1.0` (batch ready signal).

---

## Testing / Validation Checklist

Automated (smallest relevant):
- `cd third_party/nemo-rl && uv run --group test pytest -q`

Manual (must do on a real run):
1. Run the async sliding puzzle baseline (Phase 0).
2. During rollout, trigger shrink of a subset:
   - observe admission closes, targeted aborts happen, ACK is observed, subset sleeps, and work is retried elsewhere.
3. Trigger expand:
   - observe new workers wake with tags/levels, sync-on-expand runs, queued-first rebalance happens, and optional in-flight rebalance can be enabled.
4. Confirm `report_progress` logs show reasonable counts and stable `oldest_unfinished_creation_ts` across retries.

---

## Performance Considerations (Day One)

- Heartbeat cadence: keep batch-start + 2% bands; avoid additional high-frequency polling.
- Retry overhead: abort+retry adds wasted tokens; ensure it triggers only for shrink/expand rebalance decisions or hard failures, and always requires backend ACK before proceeding.
- Coalescing: never abort an in-flight sync; coalesce to “highest version wins” to avoid repeated weight transfers.

## Migration Notes

- This should be an additive change: existing configs without SchedRL enabled behave as before.

## References

- NeMo-RL adaptation notes: `design_doc/archive/adaptation_nemo_rl.md`
- Shared protocol: `design_doc/multi-pipeline-adaptation-plan.md`
- Mechanisms research: `thoughts/shared/research/2026-01-28-schedrl-framework-mechanisms.md`
- Cross-framework research: `thoughts/shared/research/2026-01-28-schedrl-adaptation-research.md`
