# SchedRL Adaptation Plan: SkyRL-train (one-step-off + fully-async, vLLM-first)

## Overview

This plan adapts **SkyRL-train** to the SchedRL protocol so it can be scheduled alongside other pipelines for elastic
time-sharing. We support two existing SkyRL-train async modes:

- **One-step-off async** (pipelined) → SchedRL `update_policy=BATCH`
- **Fully-async bounded staleness** (streaming) → SchedRL `update_policy=QUIESCE-by-abort` using `pause_generation()` →
  weight sync → `resume_generation()`

Initial “async + multi-turn” reference target stays **GSM8K multi-turn** (`third_party/SkyRL/skyrl-train/examples/turn_level_rewards/`)
per `design_doc/adaptation_skyrl.md`. SkyAgent/SkyContent integration is out of scope for this phase.

## Current State Analysis (What exists today)

### Reusable mechanisms

- **Pause/abort/resume boundary exists** and is already used by the fully-async trainer:
  - `pause_generation()` / `resume_generation()` in `third_party/SkyRL/skyrl-train/skyrl_train/inference_engines/inference_engine_client.py:580`
  - Fully-async uses pause→sync→resume every train step: `third_party/SkyRL/skyrl-train/skyrl_train/fully_async_trainer.py:415`
- **Single-request retry-until-non-abort exists** for the token-space `generate()` path:
  - `third_party/SkyRL/skyrl-train/skyrl_train/inference_engines/inference_engine_client.py:212`
- **Engine routing exists** (session-affinity via hash):
  - `_select_engine_idx(session_id)` in `third_party/SkyRL/skyrl-train/skyrl_train/inference_engines/inference_engine_client.py:161`
  - `route_prompts_to_engines(..., session_ids)` in `third_party/SkyRL/skyrl-train/skyrl_train/inference_engines/utils.py:74`
- **Weight sync strategy infrastructure exists** (broadcast + CUDA IPC):
  - `third_party/SkyRL/skyrl-train/skyrl_train/weight_sync/broadcast_strategy.py:72`
  - `third_party/SkyRL/skyrl-train/skyrl_train/weight_sync/cuda_ipc_strategy.py:90`
- **vLLM engine already accepts explicit request_id** at the vLLM API level:
  - `self.llm.generate(..., request_id=request_id, ...)` in `third_party/SkyRL/skyrl-train/skyrl_train/inference_engines/vllm/vllm_engine.py:407`

### Protocol gaps vs SchedRL (must address)

From `design_doc/multi-pipeline-adaptation-plan.md` + `design_doc/adaptation_skyrl.md`:

1) **Subset lifecycle is missing**
   - SkyRL-train assumes a fixed `generator.num_inference_engines` and treats the engine list as static.
   - SchedRL requires subset `shrink_workers(worker_indices=...)` / `expand_workers(worker_indices=...)`.

2) **Deterministic per-turn backend request IDs are missing**
   - vLLM request ids are currently generated inside the engine (`uuid4().hex`) for async `generate()`:
     `third_party/SkyRL/skyrl-train/skyrl_train/inference_engines/vllm/vllm_engine.py:432`
   - SchedRL requires coordinator-provided deterministic ids:
     `request_id = f"{trajectory_id}:{turn_id}:{attempt}"` (so we can do targeted abort+ACK+retry).

3) **Shrink requires subset-targeted abort + abort ACK + retry on remaining**
   - Today, `pause_generation()` aborts **all engines**: `third_party/SkyRL/skyrl-train/skyrl_train/inference_engines/inference_engine_client.py:580`
   - Retry loops currently pin to a single `engine_idx` and do not “reroute after shrink”:
     - token `generate()` retry: `third_party/SkyRL/skyrl-train/skyrl_train/inference_engines/inference_engine_client.py:212`
     - `/chat/completions` retry: `third_party/SkyRL/skyrl-train/skyrl_train/inference_engines/inference_engine_client.py:330`

4) **SchedRL progress heartbeats are missing**
   - No `report_progress(queued_trajectories, inflight_trajectories, percent_completed, oldest_unfinished_creation_ts, ...)`.

## Desired End State (Definition of Done)

SkyRL-train is schedulable by SchedRL with:

1) Supported modes:
   - One-step-off async → `update_policy=BATCH`
   - Fully-async bounded staleness → `update_policy=QUIESCE-by-abort` (pause→sync→resume)

2) Elastic time-sharing (`migration_policy=REQUEST_RETRY`) that works mid-flight:
   - **Close admission** for subset `P`
   - **Abort subset** `P` (not abort-all), wait **abort ACK** (`inflight(P)==0`)
   - **Retry** affected work on `new_S` (do not restart trajectories if commit point not crossed)

3) Deterministic per-turn request IDs (coordinator-provided) used by the backend:
   - vLLM `request_id` and (future) SGLang `rid`

4) `report_progress(...)` emitted at batch start + 2% bands (trajectory units):
   - `queued_trajectories`, `inflight_trajectories`, `percent_completed`, `oldest_unfinished_creation_ts`

5) Adapter RPC surface matches the Final Plan:
   - `close_admission(worker_indices, action_id, activation_epoch) -> ActionResponse`
   - `open_admission(worker_indices, action_id, activation_epoch) -> ActionResponse`
   - `shrink_workers(worker_indices, action_id, activation_epoch) -> ActionResponse`
   - `expand_workers(worker_indices, checkpoint_version, action_id, activation_epoch) -> ActionResponse`

Registration invariant (State Reset on Registration):
- On (re)registration, assume `S_actual={}` and release/kill any leftover engines from a prior scheduler session.

## What We’re NOT Doing (Scope exclusions)

- SkyAgent/SkyContent integration (explicitly excluded in `design_doc/adaptation_skyrl.md`)
- Fully-async on SGLang until abort/pause semantics match vLLM
- Mini-SWE / tool-heavy agentic async until we define commit points / idempotency rules for side effects

## Patching Strategy (Phase 1)

For the first SkyRL integration, we do not assume direct edits to `third_party/SkyRL/skyrl-train/`.

- Implement SkyRL-specific hooks via import-time patching using `sitecustomize.py` shims.
- Ship the shim to the driver and all Ray workers via `PYTHONPATH` (and set `SKYRL_PYTHONPATH_EXPORT=true` so SkyRL
  propagates `PYTHONPATH` into Ray runtime env).
- Scope: SchedRL shrink/expand support is only for `generator.batched=false` (non-batched, `num_prompts==1` per
  `InferenceEngineClient.generate(...)` call). Batched generation is out of scope in this phase.

## Plan Phases

### Phase 0 — Baseline “Supported Pipeline” (no scheduler control)

Goal: establish SkyRL-train as an accepted pipeline in this workspace before adding SchedRL control loops.

- One-step-off entrypoint: `third_party/SkyRL/skyrl-train/examples/async/main_async.py`
- Fully-async entrypoint: `third_party/SkyRL/skyrl-train/examples/fully_async/main_fully_async.py`
- Multi-turn reference: `third_party/SkyRL/skyrl-train/examples/turn_level_rewards/` (GSM8K multi-turn)

Deliverable: “how to run” notes + pinned configs; no shrink/expand.

## Actionable Implementation Notes (mined from the older plan)

These are concrete “first PR” items that keep the plan implementable and directly tie into the SchedRL hard
requirements (deterministic request ids + targeted abort + subset control).

Note: in Phase 1, implement these via `sitecustomize.py` patch shims (monkeypatching SkyRL at import time) rather than
editing `third_party/SkyRL/skyrl-train/` directly.

1) Add `request_ids` to the token-space generate API surface:
   - Extend `InferenceEngineInput` in `third_party/SkyRL/skyrl-train/skyrl_train/inference_engines/base.py` to include:
     - `request_ids: Optional[List[str]]`
     - invariant: if present, `len(request_ids) == len(prompt_token_ids)` for `generate()`.
   - In `third_party/SkyRL/skyrl-train/skyrl_train/inference_engines/inference_engine_client.py`, plumb
     per-prompt request ids through `InferenceEngineClient.generate(...)` down to engine `generate(...)`.
   - In `third_party/SkyRL/skyrl-train/skyrl_train/inference_engines/vllm/vllm_engine.py`, update async
     `generate()` to:
     - use provided `request_ids[i]` instead of `uuid4().hex`, and
     - fall back to the current UUID behavior when `request_ids` is not provided.
   - This enables SchedRL-required deterministic ids:
     `request_id = f"{trajectory_id}:{turn_id}:{attempt}"`.

2) Add targeted abort (by request id) + an abort ACK signal:
   - Extend vLLM `abort_generation(...)` to accept `request_ids: Optional[List[str]]`.
     - If `request_ids` is provided, abort only those; otherwise preserve current “abort everything unfinished” behavior.
   - Return an explicit ACK value from abort (at minimum: count of aborted request ids, ideally the ids).
   - Add an `InferenceEngineClient.abort_requests(request_ids=[...])` method.
     - To route aborts correctly, the client must be able to map `request_id -> engine_idx`.
     - Practical approach: record this mapping at submission time in `InferenceEngineClient.generate()` for any request
       where the request id is supplied (batched and single-prompt).

3) Make retry loops reroute-able after shrink:
   - Today the retry loops pin to a specific `engine_idx` (token `generate()` retry and `/chat/completions` retry).
   - For `migration_policy=REQUEST_RETRY`, after a shrink-triggered abort, the next attempt must be able to reselect a
     new engine from the **active engine set** (otherwise a shrink that removes the pinned engine can deadlock retries).

### Phase 1 — Implement `report_progress(...)` (pipeline heartbeats)

Goal: make SkyRL-train observable to the scheduler with SchedRL-standard units and cadence.

Approach (fully-async is the primary target first):

- Track inflight **generation groups** via `_AsyncStalenessManager._stat.running`:
  - `third_party/SkyRL/skyrl-train/skyrl_train/fully_async_trainer.py:155`
- Track queued **generation groups** via `generation_output_group_buffer.qsize()`:
  - enqueue point at `third_party/SkyRL/skyrl-train/skyrl_train/fully_async_trainer.py:571`
- Convert **groups → trajectories** using `group_size = len(generator_output["response_ids"])`:
  - `third_party/SkyRL/skyrl-train/skyrl_train/fully_async_trainer.py:608`
- Define `step_target_trajectories = train_batch_size * n_samples_per_prompt` (SchedRL-standard).
- Track `oldest_unfinished_creation_ts` keyed by trajectory identity; do not reset on abort/retry.

One-step-off:

- Identify the “rollout-ready boundary” (where the next train step batch is ready) and compute the same fields.

Deliverable: pipeline heartbeats at batch start + 2% bands, consistent with `design_doc/multi-pipeline-adaptation-plan.md`.

### Phase 2 — Subset lifecycle for inference engines (admission + shrink/expand)

Goal: add subset control required by SchedRL’s `expand_workers(...)` / `shrink_workers(...)`.

Required changes:

1) **Active engine set**
   - Add `active_engine_indices` (or an equivalent stable routing table).
   - Routing must use only active engines; do not rely on `hash % len(engines)` changing with shrink.
   - Implementation note (minimal): keep `engines` list stable, and maintain an `active_engine_indices` list plus a
     `paused_engine_indices` set in `InferenceEngineClient`; route only to `active \\ paused`.

2) **Admission control**
   - Close admission for subset `P` before aborting (avoid “submitted but not schedulable” races).
   - Implementation note (minimal): admission-close is “routing-close”: mark the subset as paused/inactive so new work
     cannot be routed to it before abort is issued.

3) **Subset-targeted abort + abort ACK**
   - Implement “abort only engines in `P`” and wait for ACK (timeout → fail fast).
   - For now, “abort ACK” can be defined as: each engine in `P` reports no unfinished requests.
   - Implementation note (minimal): add `pause_generation_subset(engine_indices)` and `resume_generation_subset(engine_indices)`
     on `InferenceEngineClient` so shrink can abort only the targeted engines while keeping the rest live.

4) **Retry must reroute**
   - On shrink-triggered abort, retry loops must be able to select a new active engine (cannot pin to removed engine).

Deliverable: safe shrink/expand that preserves correctness and allows continuing training/generation on `new_S`.

### Phase 3 — Deterministic per-turn backend request IDs (REQUEST_RETRY requirement)

Goal: enable targeted abort+ACK+retry and robust debug via stable request identity.

Design:

- Coordinator constructs: `request_id = f"{trajectory_id}:{turn_id}:{attempt}"` (attempt increments on retry-after-ACK).
- Plumb request_id into the backend for:
  - token-space `generate()`
  - (future) `/chat/completions` and SGLang

Concrete wiring points:

- Multi-turn generator passes a stable `session_id` today:
  - `third_party/SkyRL/skyrl-train/skyrl_train/generators/skyrl_gym_generator.py:291`
- vLLM currently generates random request ids for async generate:
  - `third_party/SkyRL/skyrl-train/skyrl_train/inference_engines/vllm/vllm_engine.py:432`
- vLLM can accept explicit request ids:
  - `third_party/SkyRL/skyrl-train/skyrl_train/inference_engines/vllm/vllm_engine.py:407`

Deliverable: deterministic `request_id` end-to-end + counters/limits:

- cap engine-error retries per `(trajectory_id, turn_id)` (default 3)
- do not cap preemption retries (scheduler-driven aborts)

### Phase 4 — Mini-SWE async (tools) (backlog after shrink safety)

Goal: add async Mini-SWE examples once side-effect safety is specified.

- One-step-off: new entrypoint mirroring `examples/async/main_async.py` but using:
  `third_party/SkyRL/skyrl-train/examples/mini_swe_agent/mini_swe_generator.py`
- Before enabling mid-flight shrink:
  - define commit point(s)
  - add idempotency keys for tool actions

## Success Criteria (Minimum)

- Fully-async GSM8K multi-turn runs with `pause_generation()`/`resume_generation()` active and continues after weight sync.
- Mid-flight shrink (while generation is happening) completes with:
  - admission closed on `P`
  - abort ACK received on `P`
  - continued rollout on `new_S` with retries completing
- `report_progress(...)`:
  - emits at batch start and when 2% bands are crossed
  - uses trajectory units; `percent_completed >= 1.0` implies “next train step batch ready”

## Open Questions (Need your call)

1) Resolved constraint: implement SkyRL integration via `sitecustomize.py` shims (out-of-tree) for now; avoid direct
   edits to `third_party/SkyRL/skyrl-train/`.
2) For “shrink”, we MUST fully release GPU memory (Final Plan invariant). If a “sleep/offload” path does not actually
   free weights+KV on GPU, we must terminate the relevant actors or add a real full-offload primitive before calling
   `shrink_workers(...)` complete. For vLLM engines, `shrink_workers(...)` MUST use deep sleep (`sleep(level=2)`) so both
   model weights and KV cache are released; it is acceptable to keep CUDA runtime / CUDA graph allocations for now.
3) Resolved constraint: SchedRL shrink/expand support is scoped to **non-batched** generation only.
   - SchedRL runs MUST set `generator.batched=false`.
   - Batched generation (`generator.batched=true`) is out of scope for shrink/expand in this phase.

## References

- `design_doc/adaptation_skyrl.md`
- `design_doc/multi-pipeline-adaptation-plan.md`
- Research: `thoughts/shared/research/2026-01-28-schedrl-framework-mechanisms.md`
- Research: `thoughts/shared/research/2026-01-28-schedrl-adaptation-research.md`
