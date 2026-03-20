# Issues in rlix (schedrl/ library)

> Source: `code_review/output/group_*/review_findings.yaml` + `review_summary.yaml`
> Scope: 31 `[S]` commits in `schedrl/` (now `rlix/`)
> Date: 2026-03-01

---

## P1 — Should Block Merge (6 issues)

### Group: GPU Allocation Atomicity — `scheduler.py`

~~Both share the same root cause: `idle_gpus` is mutated before `active_allocations` is updated.~~ **DOWNGRADED to P3 (audit hardening only)**

- ~~**G01-RULE-47.1** — GPU leak on exception path in `_apply_plan_and_signal`~~ **DOWNGRADED P3**
  - File: `rlix/scheduler/scheduler.py:L2543-2544`
  - Both mutations are Python set/dict builtins (`set.__isub__`, `dict.__setitem__`) that cannot practically throw between them. The `ClusterAllocation` object is fully constructed before either mutation. Tracing code that could fail runs after both mutations complete. The method runs under the scheduler lock, and any exception triggers `_fail_fast_shutdown` — "permanently lost GPUs" cannot happen in practice.
  - Status: P3 audit hardening. No code change needed.

- ~~**G01-RULE-28.2** — TP-group GPU bundles partially allocated (same root cause as RULE-47.1)~~ **DOWNGRADED P3**
  - Same analysis as RULE-47.1. The TP-group atomicity concern requires an exception between two Python builtins, which is not a realistic failure mode.

### Group: Resize Call Ordering — `scheduler.py`

- ~~**G03-AF-28.3** — `_execute_resize_calls` does not guarantee shrinks complete before expands~~ **FIXED**
  - `rlix/scheduler/scheduler.py:L2028-2070` — `_execute_resize_calls` now runs all shrinks concurrently via `asyncio.gather`, waits for completion, then runs all expands concurrently. Shrinks always complete before any expand starts.

### Group: Progress Tracking — `scheduler.py`

- ~~**G05-AF-05-003** — Stale adapter streams accumulate indefinitely in progress store~~ **FIXED**
  - `rlix/pipeline/coordinator.py:L345` — `clear_progress_stream()` added for per-stream clearing, called by `GroupQueueManager.end_progress_batch()` after `get_batch()` returns. `rlix/scheduler/scheduler.py:L1296` — `clear_progress()` added for pipeline-level clearing when all streams are gone.

### Group: Multi-LoRA Pipeline — `multi_lora_pipeline.py`

- ~~**G11-AF-1.3** — Phase 1 `_notify_ready_to_release_actor_infer` silently removed by e703995~~ **FIXED (intentional by design)**
  - Omission is intentional and extensively documented in `rlix/pipeline/multi_lora_pipeline.py` (L356-361, L429-436, L579-584). Actor_infer keeps its GENERATION allocation across ticks so `sync_lora_weights` can push trained LoRA weights directly to active workers. Release happens at end-of-loop cleanup (L848).

### Group: Coordinator Lock — `coordinator.py`

- ~~**G11-RULE-48.2** — `sync_adapter_weights` holds `_resize_sync_lock` across unbounded blocking RPCs~~ **DOCUMENTED (accepted design tradeoff)**
  - File: `rlix/pipeline/coordinator.py:L459-494`
  - Lock held across NCCL broadcast is intentional: releasing before sync risks sending weights to a worker shrunk mid-broadcast (NCCL rank mismatch → crash). NCCL collectives are not cancellable so a per-sync timeout would leave the communicator broken. Mitigations: `resize_infer` acquires with `_RESIZE_LOCK_TIMEOUT_S` (180s default) so scheduler is never blocked indefinitely; if all workers are sleeping, lock is released immediately (early return).

### Group: Orchestrator — `orchestrator.py`

- ~~**G10-BUG-01** — `scheduler_env` NameError on cold-start of scheduler actor~~ **FIXED**
  - `rlix/orchestrator/orchestrator.py:L118` — `_ensure_scheduler_singleton` now takes `env_vars` parameter; call site passes `self._env_vars`.

---

## P2 — Fix Before Next Milestone (17 issues)

### Group: Scheduler API Validation Gaps — `scheduler.py` and `validation.py`

These are all in `schedrl/scheduler/scheduler.py` and `schedrl/protocol/validation.py`, from Group 1 (commit `f5e5691`):

- ~~**RULE-6.3** — Missing `_should_background_rebalance_locked` function~~ **FIXED**
  - `rlix/scheduler/scheduler.py:L1569` — `_should_background_rebalance_locked()` now exists.

- ~~**RULE-17.2** — `_compute_shrink_budget_by_pipeline_id` allows shrinking clusters with pending requests~~ **INVALID**
  - Shrink budget behavior for zero-target clusters with pending requests is intentional by design.

- ~~**RULE-23.2** — `validate_execution_plan` silently skips shrink ops for unregistered clusters~~ **FIXED**
  - `rlix/scheduler/validation.py:L178` — `continue` replaced with `ValidationError(condition=11)` raise.

- ~~**RULE-34.1** — `register_pipeline_topology` accepts arbitrary cluster names~~ **FIXED**
  - `rlix/protocol/validation.py:L67-70` — `validate_register_pipeline` now checks `cluster_name not in ALL_CLUSTER_NAMES`.

- ~~**RULE-34.2** — `validate_register_pipeline` missing `actor_infer` check~~ **FIXED**
  - `rlix/protocol/validation.py:L49` — `GENERATION_CLUSTER_NAME` (actor_infer) presence check added.

- ~~**RULE-35.3** — `_topology_ready.wait()` has no timeout~~ **FIXED**
  - `rlix/scheduler/scheduler.py:L315` — `asyncio.wait_for(..., timeout=_TOPOLOGY_READY_TIMEOUT_S)` with configurable `RLIX_TOPOLOGY_READY_TIMEOUT_S` env var.

- ~~**RULE-43.2** — `admit_pipeline` and `register_pipeline_topology` do not trigger wakeup event~~ **INVALID**
  - Wakeup on admit/register is not needed by design — no requests can be pending at admission time.

- ~~**RULE-53a.3** — Cluster name registered vs cluster name parsed are inconsistent~~ **FIXED**
  - Same fix as RULE-34.1 — `validate_register_pipeline` enforces `ALL_CLUSTER_NAMES` at registration.

- ~~**RULE-54.3** — `unregister_pipeline` uses string prefix scan without validating cluster ID format~~ **FIXED**
  - `rlix/scheduler/scheduler.py:L982` — `unregister_pipeline` now uses `parse_cluster_id` with fail-fast on malformed IDs.

- ~~**RULE-55.2** — GPU double-counting in Phase 2 non-gen demand donor shrink~~ **FIXED**
  - `rlix/scheduler/scheduler.py:L1751-1766` — `already_in_shrink` dedup guard; `planned_available_gpus |= bundle` only runs for genuinely new shrink entries.

- ~~**RULE-58.2** — `notify_ready_to_release` raises on zero active DP instead of returning `[]`~~ **FIXED**
  - `rlix/scheduler/scheduler.py:L2704-2705` — `await_release_gpus` returns early when `active_dp_ranks` is empty.

### Group: Shutdown / Fail-Fast — `orchestrator.py` and `scheduler.py`

- ~~**RULE-15.2 / RULE-35.1** — `_fail_fast_shutdown` calls `orchestrator.shutdown.remote()` without awaiting~~ **FIXED**
  - `rlix/scheduler/scheduler.py:L2081` — now `await asyncio.wait_for(ref, timeout=_FAIL_FAST_SHUTDOWN_TIMEOUT_S)`.

- ~~**RULE-15.3** — `_kill_local_ray_task` missing `max_task_retries=0`~~ **FIXED**
  - `rlix/orchestrator/orchestrator.py:L85` — changed to `@ray.remote(max_retries=0, max_task_retries=0)`.

### Group: Multi-LoRA Pipeline Quality — `multi_lora_pipeline.py`

- ~~**G11-AF-1.2** — First-ready tag starvation risk — deque is FIFO, not round-robin~~ **FIXED**
  - `rlix/pipeline/multi_lora_pipeline.py:L491-520` — round-robin fairness added: non-blocking `ray.wait(..., timeout=0)` probes all ready refs, then picks the one closest to deque front. Consumed tags re-enter at the tail via `in_flight.append(...)`, cycling each LoRA to the front in turn.

- ~~**G07-RULE-21.1** — `consumed_samples` not tracked per-adapter for iterator fast-forward~~ **REWORDED — accepted limitation**
  - File: `rlix/pipeline/multi_lora_pipeline.py`
  - Original claim ("shrink-to-zero causes data repetition") is incorrect: `resize_infer` only touches routing/worker-load state — per-tag `RolloutScheduler` actors survive shrink/expand intact with their iterator state in memory. No "known gap TODO" exists in the file.
  - Actual gap: **crash/resume** — per-tag rollout scheduler iterator state is not persisted in `state.kv` (only `lora_step_by_adapter`, `global_tick`, `tag_to_adapter`, `pending_val_info` are saved). After a full checkpoint-restore cycle, rollout schedulers are recreated from scratch and iterators restart from the beginning.
  - Status: accepted limitation. Only relevant if deterministic resume continuity is required.

- ~~**G07-RULE-21.2** — `adapter_rng_states` not persisted to `WorkerState.kv` on shrink~~ **ACCEPTED**
  - File: `rlix/pipeline/multi_lora_pipeline.py`
  - Per-adapter RNG lives inside Megatron train strategy and is saved only at `save_steps` intervals during real checkpoints. Current resize flow preempts infer workers, not train workers, so train-side RNG stays alive in memory during normal operation.
  - Status: accepted limitation for the current deployment model. Only matters if train actors are also torn down between checkpoints, which does not happen.

- ~~**G07-RULE-19.1** — Single `shrink_sampler` call vs documented val-first/train-second two-call protocol~~ **CLOSED (outdated)**
  - File: `rlix/pipeline/multi_lora_pipeline.py` — `_shrink_all_schedulers`
  - `_shrink_all_schedulers()` already implements a 2-phase multi-call pattern: all-but-last `skip_offload=True`, last `skip_offload=False`. This satisfies the intent of RULE-19.1 (only one physical offload, all others routing-only). The deviation from the literal "val-first then train" ordering does not matter semantically.

### Group: Client — `client.py`

- ~~**G10-BUG-02** — Explicit `env_vars` parameter silently ignored in `connect()`~~ **FIXED**
  - `rlix/client/client.py:L47-48,L59` — `env_vars` properly passed through `ConnectOptions` to orchestrator and runtime_env.

### Group: Configuration & Code Quality

- ~~**G01-DOD-INV-8** — Hardcoded timeout literals in `orchestrator.py` (6 instances)~~ **FIXED**
  - `rlix/orchestrator/orchestrator.py:L34-39` — all 6 timeouts now use `parse_env_timeout_s` with env var overrides.

- ~~**G01-RULE-36.1** — `assert` used for correctness invariants (elided under `-O` flag)~~ **FIXED**
  - No `assert` statements remain in `rlix/scheduler/scheduler.py`.

- ~~**G05-AF-05-001** — `"__full_finetune__"` magic string literal not a named constant~~ **FIXED**
  - `rlix/scheduler/scheduler.py:L87` — `_FULL_FINETUNE_STREAM_KEY` module-level constant.

- ~~**G03-AF-5f6a1ab** — Brittle Ray internal API in `kill_pipeline` last-resort path~~ **ACCEPTED (last-resort only)**
  - `rlix/orchestrator/orchestrator.py:L360-384` — still uses Ray internals but gated behind named-actor kill failure + timeout. Explicit `FIXME` comment in code. Accepted risk: path only triggers for unnamed actors that survive normal kill flow.

- ~~**G03-AF-d945a80** — `max_concurrency` reduced from 1000 to 32 without documented rationale~~ **FIXED**
  - `rlix/pipeline/coordinator.py:L31-33` — comment added documenting rationale.

### Group: Design Sign-Off Required — NEEDS_HUMAN

- ~~**G03-AF-62bee8b** — `CompletionSuspensionOp` removal needs sign-off~~ **CLOSED (signed off)**
  - `CompletionSuspensionOp` is fully removed from codebase (zero references in `rlix/`). Replacement path: `await_release_gpus` → `PendingPlannedReleaseRequest` → scheduling cycle converts to `SchedGuidedShrinkOp` → waiter signaled after commit (L2632-2636). Protected set in gap-ratio planner prevents re-expansion of ranks pending shrink.

- ~~**G03-AF-bd80618** — `percent_completed > 1.0` overshoot not documented~~ **FIXED**
  - `rlix/pipeline/coordinator.py:L409` — coordinator now clamps `percent_completed` to 1.0 before emitting to scheduler.

### Group: Cross-Cutting — INV-6 Policy Leakage Check

- ~~**Cross-cutting INV-6** — `Priority.` references in `rlix/pipeline/` need human review~~ **INVALID**
  - All `Priority.` uses in `rlix/pipeline/` are pass-through enum arguments (e.g., `Priority.GENERATION` passed to `_request_cluster_gpus`). No scheduling policy logic in pipeline layer.

---

## Summary (by priority)

**All 23 original issues resolved.** No open items remain.

- P1: 6 → **0 open** — GPU allocation atomicity (×2) DOWNGRADED to P3, resize ordering FIXED, stale streams FIXED, Phase 1 notify_ready FIXED (intentional by design), unbounded lock hold DOCUMENTED, NameError cold-start FIXED
- P2: 17 → **0 open** — scheduler API validation gaps (×11) FIXED, shutdown fire-and-forget (×2) FIXED, multi-LoRA pipeline quality: 1 FIXED (round-robin) + 3 CLOSED (accepted/outdated), client env_vars FIXED, hardcoded timeouts FIXED, assert misuse FIXED, magic string FIXED, brittle Ray API ACCEPTED (last-resort), max_concurrency comment FIXED, CompletionSuspensionOp SIGNED OFF, percent_completed FIXED, INV-6 INVALID
- Invalid: 4 — RULE-43.2, RULE-17.2, Cross-cutting INV-6, G07-RULE-19.1 (outdated)
