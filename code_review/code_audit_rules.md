# SchedRL Codebase Audit: Logic Rules & Subagent Prompts

Complete set of correctness constraints for auditing the SchedRL codebase.
Rules 1–27 are from the original design spec. Later rules were incrementally added through codebase analysis and now include disambiguated families (`52a/52b`, `53a/53b`).

Rule-ID Disambiguation: `RULE-52a.*` and `RULE-52b.*` are distinct rule families. `RULE-53a.*` and `RULE-53b.*` are also distinct rule families.

---

## 1. Lifecycle: Init, Register, Admit, Unregister, Shutdown

### Rule 1.1: Registration State Integrity

Constraint: `Orchestrator.register_pipeline` must synchronously call `self._scheduler.register_pipeline.remote(...)` and subsequently update its internal `_pipelines[pipeline_id]` dictionary to `PipelineState(..., registered=True, admitted=False)`.

Audit Prompt: "Review the `Orchestrator.register_pipeline` method. Verify that it executes a strict, blocking `ray.get(...)` call to the scheduler's `register_pipeline` method. Then, confirm it idempotently updates the orchestrator's internal `_pipelines` dictionary, setting `registered=True` and maintaining `admitted=False`."

### Rule 1.2: Admission Gate Validation

Constraint: `Orchestrator.admit_pipeline` must enforce that the pipeline state exists and `registered` is `True`. It must block on `_scheduler.admit_pipeline.remote()` and exclusively return the `AdmitResponse` object containing the `self._scheduler` handle.

Audit Prompt: "Examine the `Orchestrator.admit_pipeline` method. Check the short-circuit logic: it must raise a `RuntimeError` if the pipeline is not registered. Ensure the call to `_scheduler.admit_pipeline.remote` is awaited via `ray.get()`, followed immediately by mutating the pipeline state to `admitted=True`."

### Rule 1.3: Total Annihilation on Pipeline Teardown (kill_pipeline)

Constraint: `Orchestrator.kill_pipeline` must perform comprehensive cleanup spanning three systems:
- SharedStorage: Retrieve the `SHARED_STORAGE_ACTOR` strictly from `namespace="global_storage_namespace"` to release coordination metadata.
- Ray Actors: Use `ray.util.state.list_actors` with a filter on `ray_namespace`, and aggressively invoke `ray.kill(handle, no_restart=True)` on ALL alive actors (falling back to `ActorID` for unnamed actors if necessary).
- Placement Groups: List and destroy all placement groups prefixed exactly with `schedrl_pg:{pipeline_id}:`.

Audit Prompt: "Audit the `Orchestrator.kill_pipeline` method. Validate all three tiers of cleanup: (1) Confirm `SHARED_STORAGE_ACTOR` is looked up in the correct `global_storage_namespace`. (2) Ensure both named and unnamed actors within the target `ray_namespace` are forcefully killed with `no_restart=True`. (3) Verify that placement groups matching the prefix `schedrl_pg:{pipeline_id}:` are destroyed via `ray.util.remove_placement_group`."

### Rule 1.4: Fail-Safe GPU Tracing

Constraint: Inside `SchedulerImpl`, all tracing logic must be wrapped in `_safe_trace_call`. If a trace throws an error (other than I/O), it must disable tracing entirely but allow the scheduler to continue running.

Audit Prompt: "Review `SchedulerImpl._safe_trace_call`. Verify the exception handling block. Ensure that any generic `Exception` raised by tracing tools safely logs an error and cleanly toggles `_enable_gpu_tracing = False` without crashing the core scheduling loop."

---

## 2. Generate/Rollout Schedulers & Parameter Offloading

### Rule 2.1: Atomicity via RequestScheduler Delegation

Constraint: `RolloutScheduler.shrink_sampler` and `expand_sampler` MUST NOT manipulate workers directly. They must strictly act as thin wrappers that delegate the entire operation to `self.generate_scheduler.shrink_workers.remote(dp_ranks, skip_offload=...)` / `expand_workers.remote(dp_ranks, skip_load=...)`, which owns the `routing_lock`.

Audit Prompt: "Review the `RolloutScheduler.shrink_sampler` and `expand_sampler` implementation. Confirm they are strictly thin wrappers. Verify the core remote calls target `self.generate_scheduler.shrink_workers.remote(dp_ranks, skip_offload=...)` and `expand_workers.remote(dp_ranks, skip_load=...)`. Flag any logic attempting local state mutation within the `RolloutScheduler`."

### Rule 2.2: SchedRL Concurrent Pipeline Adapter Logic

Constraint: `SchedRLConcurrentPipeline._shrink_workers` handles routing and offloading together by explicitly calling `shrink_sampler` with `skip_offload=False`. Conversely, `_expand_workers` passes `train_skip_load` down to `skip_load=bool(...)`.

Audit Prompt: "Examine `SchedRLConcurrentPipeline._shrink_workers` and `_expand_workers`. Validate that `_shrink_workers` guarantees `skip_offload=False`. Ensure `_expand_workers` faithfully forwards the `train_skip_load` parameter to the `skip_load` argument of the expand sampler."

---

## 3. Selective Model Update (IPC / NCCL Mechanisms)

### Rule 3.1: IPC Fallback over NCCL Colocation (Topology Validation)

Constraint: In `ModelUpdateService._build_comm_plan_for_sender`, duplicate physical GPUs (same node/gpu combination) between source and target must be excluded from the `tgt_ranks_in_group` list. NCCL cannot form a group with duplicate physical GPUs; same-GPU parameters must fall back to the IPC path.

Audit Prompt: "Audit `ModelUpdateService._build_comm_plan_for_sender`. Locate the nested loop matching `tgt_gpu_key` to `src_gpu_keys`. Verify that if a target device maps to the same physical GPU as the source, it is intentionally excluded from `tgt_ranks_in_group` via a `continue` statement to force IPC fallback."

### Rule 3.2: Asynchronous Collective Group Setup

Constraint: `ModelUpdateService.sync_selected_workers` must execute setup calls remotely: calling `setup_collective_group.remote(..., mode="receiver")` on all members of `tgt_ranks_in_group` and `mode="sender"` on the `src_rank`. It must barrier via `_ray_get_with_timeout` before initiating over-the-wire broadcast.

Audit Prompt: "Inspect `ModelUpdateService.sync_selected_workers`. Confirm that collective group setups execute asynchronously for both sender and receiver modes. Guarantee that a strict `_ray_get_with_timeout` barrier awaits the completion of all `setup_collective_group` calls before invoking `selective_sync_active_cache.remote`."

### Rule 3.3: Strict Broadcast Topology Execution

Constraint: The actual transfer (`worker.selective_sync_active_cache.remote`) is only given a `comm_plan` if `is_leader` is `True`. Non-leader ranks must receive `comm_plan=None`.

Audit Prompt: "Review the final `sync_refs.append` block in `ModelUpdateService.sync_selected_workers`. Ensure `comm_plan` is defined cleanly via `comm_plan_by_rank.get(...) if is_leader else None`. Verify the explicit passing of the `is_leader` boolean to the `selective_sync_active_cache` task."

---

## 4. Shrink-to-Zero & Expand-from-Zero

### Rule 4.1: Shrink-to-Zero Admission Gating

Constraint: `RequestScheduler._rebalance_on_shrink` must set `need_suspend=True` and clear `suspend_notifier` BEFORE clearing `active_dp_ranks` when shrinking to zero. This ensures new requests block at `_check_suspend()` rather than hitting an empty worker set.

Audit Prompt: "Review `RequestScheduler._rebalance_on_shrink`. Locate the branch where `keep_ranks` is empty. Verify that `need_suspend=True` is set and `suspend_notifier.clear()` is called BEFORE `active_dp_ranks` is mutated. Flag any code path where `active_dp_ranks` becomes empty while `need_suspend` is `False`."

### Rule 4.2: Expand-from-Zero Resume Sequence

Constraint: `RequestScheduler.expand_workers` must call `resume()` (set `need_suspend=False`, set `suspend_notifier`) AFTER workers are loaded and model update is complete, not before.

Audit Prompt: "Examine `RequestScheduler.expand_workers`. Verify that `resume()` is called only after: (1) `active_dp_ranks` is updated, (2) model weights are loaded (if `skip_load=False`). Ensure no requests can be routed before `resume` completes."

### Rule 4.3: Sleep Level 2 Validation

Constraint: `SchedRLAdapter` must validate `sleep_level=2` at initialization. If `sleep_level` is not 2, raise `RuntimeError` immediately.

Audit Prompt: "Search `SchedRLAdapter.__init__` or `initialize_pipeline`. Confirm there is an explicit check that `sleep_level == 2`. Verify it raises a clear `RuntimeError` with message explaining why level 2 is required for SchedRL time-sharing."

### Rule 4.4: NCCL Buffer Release

Constraint: Offload path must call `offload_states(include=[OffloadStateType.optimizer_states])` with `offload_nccl=True` to reclaim NCCL buffer VRAM.

Audit Prompt: "Trace the offload path in `VllmStrategy.offload_states`. Verify that when `sleep_level=2`, NCCL buffers are explicitly released. Check that `offload_nccl=True` is set in the offload configuration or explicitly passed."

---

## 5. Abort & ACK Semantics

### Rule 5.1: Targeted Abort Scope

Constraint: `RequestScheduler.abort_request(worker_indices)` must only abort requests on the specified dp ranks, not all requests. It must maintain `dp_rank_to_request_ids` inverse index for efficient targeting.

Audit Prompt: "Review `RequestScheduler.abort_request`. Verify the method accepts `worker_indices` parameter and uses an inverse index (`dp_rank_to_request_ids`) to identify targeted request IDs. Flag any implementation that aborts globally without filtering by dp rank."

### Rule 5.2: ACK Definition (In-Flight Removal)

Constraint: ACK is defined as "request leaves `running_requests`", NOT `finish_reason == "abort"`. Normal completion while abort is issued must be treated as valid ACK.

Audit Prompt: "Examine the abort completion logic in `RequestScheduler`. Verify that the await on futures resolves when the request ID is removed from `running_requests`. Confirm that `finish_reason == "abort"` is NOT required — normal completion is acceptable."

### Rule 5.3: Abort Timeout Fail-Fast

Constraint: If targeted requests do not leave `running_requests` within the configured timeout, the pipeline must crash (fail-fast), not proceed with offload.

Audit Prompt: "Locate the timeout handling in abort logic. Verify that on `asyncio.TimeoutError`, the code raises `RuntimeError` or triggers `shutdown(force=True)`. Flag any code that catches timeout silently or proceeds with offload after timeout."

---

## 6. Priority & Scheduling

### Rule 6.1: 7-Tier Priority Enum Completeness

Constraint: `Priority` enum must have exactly 7 tiers in order: `INITIALIZATION=0 < ACTOR_TRAINING=1 < CRITIC_TRAINING=2 < OLD_LOG_PROBS=3 < REF_LOG_PROBS=4 < VALUE_COMPUTE=5 < GENERATION=6`.

Audit Prompt: "Review `schedrl/protocol/types.py` `Priority` enum. Verify all 7 tiers exist with correct integer values. Check that comparison operators work correctly (lower value = higher priority)."

### Rule 6.2: Gap-Ratio Computation

Constraint: `_plan_generation_gap_ratio` must compute `normalized_gap = gap / target_ratio` where `target_ratio = remaining / step_target_trajectories`. Pipelines with higher `normalized_gap` get priority for expansion.

Audit Prompt: "Examine `SchedulerImpl._plan_generation_gap_ratio`. Verify the computation of `percent_remaining`, `target_ratio`, and `normalized_gap`. Confirm that acceptors are sorted by `normalized_gap` descending, with `pipeline_id` as tiebreaker."

### Rule 6.3: Background Rebalance Trigger

Constraint: `_should_background_rebalance_locked` must return `True` when a suspended generation cluster has non-zero remaining demand, even without pending requests.

Audit Prompt: "Review `SchedulerImpl._should_background_rebalance_locked`. Verify the branch that checks for suspended `actor_infer` with `remaining > 0`. Ensure this triggers a wakeup for expand-from-zero scheduling."

### Rule 6.4: FIFO Queue Semantics

Constraint: Pending requests must be processed in FIFO order within each priority bucket. No priority boosting or reordering within a bucket.

Audit Prompt: "Examine `_state.pending_bucket(priority)` and the signal loop. Verify that requests are popped from the front (index 0) of the list. Flag any code that reorders or sorts pending requests within a bucket."

---

## 7. Progress Reporting

### Rule 7.1: 2% Band Emission

Constraint: Progress must be emitted when `percent_completed` crosses a 2% band boundary (0%, 2%, 4%, ..., 100%). Not on every trajectory.

Audit Prompt: "Review `RolloutScheduler._maybe_emit_progress`. Verify the `last_percent_reported` tracking and the `int(percent_completed * 50) != int(last_percent_reported * 50)` comparison. Confirm emission only on band crossing."

### Rule 7.2: Completion Guarantee

Constraint: Progress must always be emitted once when `percent_completed >= 1.0`, regardless of band alignment.

Audit Prompt: "Check `_maybe_emit_progress` for an explicit check: `if percent_completed >= 1.0 and last_percent_reported < 1.0`. Verify this forces a final emission at completion."

### Rule 7.3: Multi-LoRA Aggregation

Constraint: For multi-LoRA, `percent_completed` must be computed as `min(1.0, sum(collected[a]) / sum(target[a]))`. Per-adapter progress must be in `metrics["percent_completed_by_adapter"]`.

Audit Prompt: "Examine multi-LoRA progress aggregation. Verify that total `percent_completed` sums across all adapters. Confirm `metrics["percent_completed_by_adapter"]` dict exists with per-adapter entries."

---

## 8. Multi-LoRA Training

### Rule 8.1: Sequential Adapter Training Loop

Constraint: `train_step_lora` must process adapters sequentially: for each adapter, complete all forward/backward/optimizer_step before moving to next adapter. No interleaving.

Audit Prompt: "Review `MegatronStrategy.train_step_lora`. Verify the loop structure: `for adapter_name in adapters_to_update: restore_rng → forward_backward → optimizer.step → save_rng`. Flag any concurrent or interleaved processing."

### Rule 8.2: Per-Adapter RNG Isolation

Constraint: Each adapter must have isolated RNG state stored in `adapter_rng_states[adapter_name]`. Must restore before training, save after training.

Audit Prompt: "Examine the RNG state handling in `train_step_lora`. Verify `torch.set_rng_state`, `torch.cuda.set_rng_state`, `random.setstate`, `np.random.set_state` are called for each adapter before its training. Confirm state is saved back after training."

### Rule 8.3: DDP Bucket Cache Clear

Constraint: Between adapter optimizer steps, `model.bucket_groups` caches must be cleared to prevent cross-adapter gradient pollution.

Audit Prompt: "Locate the bucket cache clearing in `train_step_lora`. Verify `model.bucket_groups` or equivalent is reset after each `optimizer.step()`. This is critical when `use_distributed_optimizer=False`."

### Rule 8.4: Per-Adapter Optimizer Validation

Constraint: `SchedRLMultiLoraPipeline.__init__` must raise `RuntimeError` if `lora_optimizer_mode != "per_adapter"`.

Audit Prompt: "Review `SchedRLMultiLoraPipeline.__init__`. Find the validation check for `lora_optimizer_mode`. Verify it raises `RuntimeError` with clear message if mode is not `per_adapter`."

### Rule 8.5: LoRA ID Consistency Verification

Constraint: After expand/load, `_verify_lora_model_update` must query all workers and verify `adapter_name → lora_int_id` mapping is consistent across ranks.

Audit Prompt: "Examine `_verify_lora_model_update`. Verify it queries each worker's `lora_adapter_id` mapping. Confirm it raises `RuntimeError` if any worker has inconsistent ID for the same adapter name."

---

## 9. Namespace Isolation

### Rule 9.1: Per-Pipeline Namespace Injection

Constraint: At pipeline admission, `runtime_env.env_vars` must include `ROLL_RAY_NAMESPACE=f"pipeline_{pipeline_id}_NS"` and `PIPELINE_ID={pipeline_id}`.

Audit Prompt: "Review the pipeline actor creation in `SchedRLAdapter.create_coordinator`. Verify `.options(runtime_env={"env_vars": {...}})` includes both `ROLL_RAY_NAMESPACE` and `PIPELINE_ID`. Confirm namespace follows `pipeline_{pipeline_id}_NS` format."

### Rule 9.2: SharedStorage Global Namespace

Constraint: All SharedStorage get/create/lookup must use `namespace="global_storage_namespace"`, NOT `RAY_NAMESPACE`.

Audit Prompt: "Search for all SharedStorage actor references. Verify each uses `.options(namespace="global_storage_namespace")` or equivalent. Flag any usage of `namespace=RAY_NAMESPACE` for SharedStorage."

### Rule 9.3: No Default Namespace Assumption

Constraint: No actor creation may rely on implicit default namespace. All `.options(name=..., namespace=...)` calls must explicitly set `namespace`.

Audit Prompt: "Grep for `.options(name=` patterns. Verify each includes explicit `namespace=` parameter. Flag any actor creation without namespace specification."

---

## 10. SharedStorage & Key Scoping

### Rule 10.1: Pipeline-Scoped Keys

Constraint: All SharedStorage keys for per-pipeline data (ports, rendezvous, metadata) must be prefixed with `{pipeline_id}:`.

Audit Prompt: "Review `Worker.get_free_port` and rendezvous code. Verify port lock keys use format `{pipeline_id}:MASTER_ADDR_PORT:{ip}:{port}`. Verify rendezvous keys use `{pipeline_id}:{cluster_name}`."

### Rule 10.2: Delete Prefix for Teardown

Constraint: `SharedStorage.delete_prefix` must accept `pipeline_id` and delete all keys starting with `{pipeline_id}:`.

Audit Prompt: "Examine `SharedStorage.delete_prefix`. Verify it iterates keys and deletes those with matching prefix. Confirm it's called in `kill_pipeline` with correct `pipeline_id`."

### Rule 10.3: HF Cache Isolation

Constraint: Worker initialization must set `HF_HOME=f"/tmp/schedrl/pipelines/{pipeline_id}/hf_home"` (private) and `HF_HUB_CACHE` (shared) via Ray `runtime_env.env_vars` before any HF imports.

Audit Prompt: "Check pipeline actor creation for HF env vars. Verify `HF_HOME` includes `pipeline_id`. Confirm these are set in `runtime_env.env_vars`, not in `Worker.__init__` (too late due to imports)."

---

## 11. Request ID Validation

### Rule 11.1: Canonical Format

Constraint: Request ID format must be `{pipeline_id}:{traj_id}:{turn_id}:{attempt}`. All components must be non-empty strings (`pipeline_id`, `traj_id`) or non-negative integers (`turn_id`, `attempt`).

Audit Prompt: "Review `schedrl/protocol/request_id.py`. Verify `build_request_id` produces the canonical format. Verify `parse_request_id` extracts all 4 components. Check that `validate_request_id` rejects malformed IDs."

### Rule 11.2: Pipeline ID Delimiter Rejection

Constraint: `pipeline_id` must not contain `:`. `validate_pipeline_id` must raise `ValueError` if `:` is present.

Audit Prompt: "Examine `validate_pipeline_id`. Verify it checks for `:` character and raises `ValueError`. Confirm this is called during registration/admission."

### Rule 11.3: Dual-Write for Upstream Compatibility

Constraint: For ROLL, set both `schedrl_request_id` (canonical) and `request_id` (internal). Abort tracking uses internal `request_id`; cross-pipeline correlation uses `schedrl_request_id`.

Audit Prompt: "Review `TrajEnvManager.make_decision`. Verify `data.meta_info["schedrl_request_id"]` is set with canonical format. Verify `request_id` is preserved for internal scheduler use."

---

## 12. Admission Gating

### Rule 12.1: Suspend Gate in generate_one_request

Constraint: `RequestScheduler.generate_one_request` must call `_check_suspend()` at entry. If `need_suspend=True`, await `suspend_notifier.wait()` before proceeding.

Audit Prompt: "Examine `RequestScheduler.generate_one_request`. Locate `_check_suspend()` call. Verify it awaits `suspend_notifier` when `need_suspend=True`. Confirm this is the FIRST blocking operation."

### Rule 12.2: Empty Active Ranks Error

Constraint: If `active_dp_ranks` is empty AND `need_suspend=False`, raise `RuntimeError("No active workers and not suspended")`. This indicates state desync.

Audit Prompt: "Review the routing logic in `generate_one_request`. Find the check for empty `active_dp_ranks`. Verify it raises `RuntimeError` if `need_suspend=False`. This should be unreachable in correct flow."

### Rule 12.3: Sticky Routing Revalidation

Constraint: When `generate_one_request` checks sticky mapping, it must verify the mapped `dp_rank` is still in `active_dp_ranks`. If not, clear and re-pick.

Audit Prompt: "Trace the sticky routing logic in `generate_one_request`. Verify after `src_rank2_dp_rank[src_rank]` lookup, there's a check `if dp_rank not in active_dp_ranks`. Confirm stale mappings are cleared and re-selected."

---

## 13. Strict Shrink Ordering

### Rule 13.1: Close Admission First

Constraint: Shrink sequence MUST be: (1) close admission (`need_suspend=True`), (2) abort/drain, (3) offload. Never out of order.

Audit Prompt: "Review `RequestScheduler._rebalance_on_shrink`. Verify the ordering: `suspend()` is called before any abort, and abort completes before any offload. Flag any code that offloads before abort ACK."

### Rule 13.2: Abort Before Offload

Constraint: `skip_offload=False` must guarantee abort+ACK before calling `offload_states` on workers.

Audit Prompt: "Examine `shrink_sampler` with `skip_offload=False`. Verify the sequence: `abort_request(worker_indices)` → await ACK → then `worker.offload_states.remote()`. Confirm no offload happens before abort completion."

---

## 14. Placement Group Cleanup

### Rule 14.1: PG Naming Convention

Constraint: Placement groups must be named `schedrl_pg:{pipeline_id}:{cluster_name}`.

Audit Prompt: "Search for placement group creation. Verify naming follows `schedrl_pg:{pipeline_id}:{cluster_name}` format. This enables prefix-based cleanup."

### Rule 14.2: PG Destruction on kill_pipeline

Constraint: `kill_pipeline` must list placement groups, filter by prefix `schedrl_pg:{pipeline_id}:`, and destroy all matches.

Audit Prompt: "Review `Orchestrator.kill_pipeline`. Verify it calls `ray.util.list_placement_groups()` or equivalent, filters by prefix, and calls `ray.util.remove_placement_group()` for each."

---

## 15. Fail-Fast Shutdown

### Rule 15.1: No Silent Exception Catching

Constraint: Critical lifecycle methods must NOT catch exceptions silently. Any exception must propagate to trigger `shutdown(force=True)`.

Audit Prompt: "Review `scheduling_cycle`, `resize_infer`, `expand_workers`, `shrink_workers`. Flag any broad `except Exception: pass` or `except: pass` blocks. Exceptions should be logged and re-raised or trigger shutdown."

### Rule 15.2: Global Shutdown on Critical Failure

Constraint: Timeout on abort ACK, expansion failure, or execution phase failure must call `orchestrator.shutdown(force=True, reason=..., source=...)`.

Audit Prompt: "Search for timeout handling in critical paths. Verify that on `asyncio.TimeoutError`, the code calls `shutdown(force=True)` with descriptive reason. Flag any timeout that silently retries or continues."

### Rule 15.3: Ray Restart Knobs

Constraint: All SchedRL actors must be created with `max_restarts=0`, `max_task_retries=0`. No automatic Ray-level retries.

Audit Prompt: "Search for `@ray.remote` decorators and `.options()` calls on actor creation. Verify `max_restarts=0` and `max_task_retries=0` are set. Flag any positive values."

---

## 16. GPU Tracing Correctness

### Rule 16.1: Trace Context Lifecycle

Constraint: `_start_gpu_trace` must store context in `_gpu_contexts[gpu_id]` ONLY after successful `track.open()`. `_end_gpu_trace` must pop context before `track.close()`.

Audit Prompt: "Review `_start_gpu_trace` and `_end_gpu_trace`. Verify context is stored only on successful slice open. Verify `_end_gpu_trace` pops context before closing track. Flag any code that stores context before confirming slice opened."

### Rule 16.2: Trace End on GPU Reclaim

Constraint: When GPUs are reclaimed (shrink, remove, unregister), `_end_traces_for_gpu_ids()` must be called BEFORE adding GPUs back to `idle_gpus`.

Audit Prompt: "Search for `idle_gpus |=` or `idle_gpus.update()` patterns. Verify `_end_traces_for_gpu_ids()` is called immediately before. Flag any code path that returns GPUs to idle pool without ending traces."

### Rule 16.3: Queue Slice Close Before Pop

Constraint: In `_signal_pending_request`, queue trace slice must close BEFORE popping from bucket. Counter update must happen AFTER pop with correct depth.

Audit Prompt: "Review `_signal_pending_request`. Verify `_trace_queue_slice_close(cluster_id)` is called before `bucket.pop(idx)`. Verify `_trace_queue_counter_update(priority, len(bucket))` is called after pop with updated length."

---

## 17. Scheduler State Consistency

### Rule 17.1: Allocation State Integrity

Constraint: `ClusterAllocation` must maintain consistency: `gpu_ids == sorted(set().union(*dp_rank_to_gpus.values()))` and `active_dp_ranks == set(dp_rank_to_gpus.keys())` for generation clusters.

Audit Prompt: "Review all mutation sites of `ClusterAllocation` in `_apply_plan_and_signal`. Verify `gpu_ids`, `active_dp_ranks`, and `dp_rank_to_gpus` are updated atomically. Flag any partial update that leaves state inconsistent."

### Rule 17.2: Protected Ranks During Planning

Constraint: `_plan_generation_gap_ratio` must never shrink dp ranks that are currently protected (active in generation with pending requests).

Audit Prompt: "Examine `_plan_generation_gap_ratio`. Locate the `_receiver_eligible` or similar predicate. Verify protected/active dp ranks are excluded from shrink budget computation."

### Rule 17.3: Planned Available GPUs Tracking

Constraint: `_state.planned_available_gpus` must be updated atomically with execution plan. Must not leak between scheduling cycles.

Audit Prompt: "Review the execution plan building phase. Verify `planned_available_gpus` is computed fresh each cycle and cleared/reset before next cycle. Flag any stale GPU tracking."

---

## 18. Coordinator → Scheduler Primitives

### Rule 18.1: request_gpus Blocking Behavior

Constraint: `request_gpus` must block until GPUs are allocated and returned. Must not return empty list except on error/shutdown.

Audit Prompt: "Review `SchedulerImpl.request_gpus`. Verify it creates `PendingRequest`, adds to bucket, awaits `pending.event.wait()`, and returns `pending.result`. Flag any early return path that doesn't wait for allocation."

### Rule 18.2: release_gpus Immediate Effect

Constraint: `release_gpus` must immediately remove allocation from `active_allocations` and return GPUs to `idle_gpus`. Must signal wakeup for pending requests.

Audit Prompt: "Examine `SchedulerImpl.release_gpus`. Verify it pops from `active_allocations`, updates `idle_gpus`, ends GPU traces, and sets `_wakeup_event`. Flag any deferred cleanup."

### Rule 18.3: release_and_request_gpus Atomicity

Constraint: `release_and_request_gpus` must execute as single atomic transaction: release first, then request. If request fails, GPUs must already be released.

Audit Prompt: "Review `SchedulerImpl.release_and_request_gpus`. Verify release completes before request is submitted. Check error handling: if request fails, released GPUs must remain in `idle_gpus` (not leaked)."

### Rule 18.4: notify_ready_to_release Blocking Semantics

Constraint: `notify_ready_to_release` must block until scheduler commits a shrink in its next cycle. Must not return immediately.

Audit Prompt: "Examine `SchedulerImpl.notify_ready_to_release`. Verify it creates `PendingPlannedReleaseRequest`, stores in `pending_planned_release_requests`, and awaits `event.wait()`. Confirm the event is set in `_apply_plan_and_signal` after shrink commit."

---

## 19. Dual RolloutScheduler Coordination

### Rule 19.1: Train/Val Shrink Order

Constraint: For shared infer cluster, val scheduler must shrink with `skip_offload=True` (routing only), then train scheduler shrinks with `skip_offload=False` (actual offload).

Audit Prompt: "Review `SchedRLConcurrentPipeline._shrink_workers`. Verify train scheduler is called with `skip_offload=False`. If val scheduler exists, verify it shrinks first with `skip_offload=True`."

### Rule 19.2: Train/Val Expand Order

Constraint: For shared infer cluster, train scheduler must expand with `skip_load=False` (actual load), then val scheduler expands with `skip_load=True` (routing only).

Audit Prompt: "Examine `SchedRLConcurrentPipeline._expand_workers`. Verify train scheduler is called with `skip_load=False`. If val scheduler exists, verify it expands after train with `skip_load=True`."

### Rule 19.3: Shared RequestScheduler Reference

Constraint: Both train and val `RolloutScheduler` must share the same `RequestScheduler` actor for routing consistency.

Audit Prompt: "Review pipeline initialization where `RolloutScheduler` actors are created. Verify both train and val schedulers receive the same `request_scheduler=self.generate_scheduler` reference."

---

## 20. Proportional Rebalancing

### Rule 20.1: Rebalance Termination Guarantee

Constraint: `_rebalance_on_expand` must have explicit termination condition. Must NOT use infinite `cycle()` without bound check.

Audit Prompt: "Review `RequestScheduler._rebalance_on_expand`. Locate the loop that distributes requests. Verify there's an explicit `while remaining > 0` with bound check. Flag any `cycle()` usage without termination guard."

### Rule 20.2: Zombie Mapping Prevention

Constraint: Before proportional rebalance, filter `src_rank2_dp_rank` to only include mappings to `old_active_dp_ranks`. Stale mappings to removed ranks must not inflate abort count.

Audit Prompt: "Examine `_rebalance_on_expand`. Verify `dp_rank_to_src_ranks` is built by filtering to `dp_rank in old_active_dp_ranks`. Flag code that uses raw `src_rank2_dp_rank` without filtering."

### Rule 20.3: Max-Load Selection Policy

Constraint: When selecting `src_ranks` to abort, pick from the most-loaded worker first (`argmax len(dp_rank_to_src_ranks[dp_rank])`), not round-robin.

Audit Prompt: "Review the abort selection loop in `_rebalance_on_expand`. Verify selection uses `argmax` to pick from most-loaded worker. Flag any `cycle()` or round-robin selection."

---

## 21. Worker State Persistence

### Rule 21.1: consumed_samples Tracking

Constraint: `SchedRLMultiLoraPipeline` must track `consumed_samples` per adapter. Must be saved before shrink, restored after expand.

Audit Prompt: "Search for `consumed_samples` in `SchedRLMultiLoraPipeline`. Verify it's saved to `WorkerState` or equivalent before shrink. Verify it's passed to `expand_sampler` for iterator fast-forward."

### Rule 21.2: Adapter RNG State Persistence

Constraint: Per-adapter RNG states must be saved to persistent storage (`WorkerState.kv`) before shrink and restored after expand.

Audit Prompt: "Examine shrink/expand logic in `SchedRLMultiLoraPipeline`. Verify `adapter_rng_states` dict is serialized and persisted before shrink. Verify it's deserialized and restored after expand."

---

## 22. Loop Termination Safety

### Rule 22.1: Gap-Ratio Planning Iteration Limit

Constraint: `_plan_generation_gap_ratio` must have iteration limit to prevent infinite loops. Must raise `RuntimeError("gap_ratio_generation_planning_exceeded_limits")` after threshold.

Audit Prompt: "Review `_plan_generation_gap_ratio`. Locate the iteration counter and threshold check. Verify `iterations > 10,000` raises `RuntimeError`. Flag any loop without iteration limit."

### Rule 22.2: Rebalance Selection Bounds

Constraint: Proportional rebalance loops must clamp planned aborts to available `src_ranks`. Must log warning and break if `max_load` reaches zero.

Audit Prompt: "Examine `_rebalance_on_expand` and `_rebalance_on_shrink`. Verify there's a check: `if max_load == 0: logger.warning(...) and break`. Flag loops that assume infinite supply."

---

## 23. Execution Plan Validation

### Rule 23.1: Plan Validation Before Commit

Constraint: `validate_execution_plan()` must be called before `_apply_plan_and_signal`. Invalid plan must raise, not commit partial state.

Audit Prompt: "Review `scheduling_cycle`. Verify `validate_execution_plan(plan)` is called before `_apply_plan_and_signal(plan)`. Verify exceptions from validation prevent any state mutation."

### Rule 23.2: Cluster ID Consistency

Constraint: All operations in `ExecutionPlan` must reference valid, registered `cluster_id`s. Operations on unregistered clusters must raise `RuntimeError`.

Audit Prompt: "Examine `validate_execution_plan`. Verify it checks each `cluster_id` exists in `pipeline_registry` or `active_allocations`. Flag any operation on unknown cluster."

### Rule 23.3: GPU Set Non-Overlap

Constraint: `gpus_to_allocate` across all operations must not overlap. No GPU can be allocated twice in same plan.

Audit Prompt: "Review plan validation logic. Verify it checks for GPU ID overlap across all `signal_pending_allocation_ops` and `sched_guided_allocation_ops`. Flag any duplicate GPU assignment."

---

## 24. resize_infer Mutual Exclusivity

### Rule 24.1: Exactly One Direction

Constraint: `resize_infer(dp_ranks_to_remove, dp_ranks_to_add)` must have exactly one of the two lists non-empty. Both empty or both non-empty must raise `ValueError` (strict XOR: `if bool(dp_ranks_to_remove) == bool(dp_ranks_to_add): raise ValueError`).

Audit Prompt: "Review `SchedRLAdapter.resize_infer` and `SchedRLConcurrentPipeline.resize_infer`. Verify mutual exclusivity check: `if bool(dp_ranks_to_remove) == bool(dp_ranks_to_add): raise ValueError`. Flag missing validation."

### Rule 24.2: Adapter RPC Lock

Constraint: `resize_infer` must hold `_infer_resize_lock` for the entire operation to prevent concurrent resize calls.

Audit Prompt: "Examine `SchedRLConcurrentPipeline.resize_infer`. Verify it acquires `self._infer_resize_lock` (context manager or explicit acquire/release). Flag any resize without lock."

---

## 25. Checkpoint & Resume Consistency

### Rule 25.1: Per-Adapter Scheduler State

Constraint: `save_checkpoint` for multi-LoRA must save `adapter_schedulers[name].state_dict()` for each adapter, not just global scheduler.

Audit Prompt: "Review checkpoint save logic in `MegatronStrategy`. Verify it saves `{"mode": "per_adapter", "schedulers": {name: sch.state_dict() ...}}`. Flag code that saves only global `self.scheduler.state_dict()`."

### Rule 25.2: Checkpoint Version in Active Model Spec

Constraint: After checkpoint save, `active_base_version` / `checkpoint_version` must be updated so expand uses correct weights.

Audit Prompt: "Examine post-checkpoint logic. Verify `checkpoint_version` or `global_step` is stored and used in subsequent `promote_active_checkpoint` calls."

### Rule 25.3: Resume Weight Consistency

Constraint: After checkpoint resume, model weights must match checkpoint state. Must not resume with stale weights.

Audit Prompt: "Review `load_checkpoint` for multi-LoRA. Verify it restores both model weights and optimizer state per-adapter. Verify `adapter_rng_states` is restored."

---

## 26. Routing Lock Atomicity

### Rule 26.1: routing_lock Scope

Constraint: `routing_lock` must be held for ANY mutation of `active_dp_ranks`, `src_rank2_dp_rank`, or `dp_rank_to_src_ranks`. No mutations outside lock.

Audit Prompt: "Search for all write accesses to `active_dp_ranks`, `src_rank2_dp_rank`. Verify each is inside `async with self.routing_lock:` block. Flag any mutation outside lock."

### Rule 26.2: Brief Lock Hold

Constraint: `routing_lock` must not be held across `await` for I/O operations (RPCs, sleeps). Only hold for in-memory state updates.

Audit Prompt: "Review lock scope in `_rebalance_on_shrink` and `_rebalance_on_expand`. Verify `routing_lock` is released before any `await ray.get()` or `await asyncio.sleep()`. Flag held lock across RPC."

---

## 27. Adapter Handle Caching

### Rule 27.1: Cache Key Consistency

<!-- Enhancement note: scheduler cache is namespace-aware tuple `(namespace, handle)`, not just actor-name keyed. -->
Constraint: `_adapter_handle_cache[pipeline_id]` must store `(registered_namespace, actor_handle)` and may be reused only when cached namespace matches current `pipeline_registry[pipeline_id]["namespace"]`. Cache entry must be removed on `unregister_pipeline`.

Audit Prompt: "Review `_get_or_lookup_adapter_handle_locked` and `unregister_pipeline` in `scheduler.py`. Verify cached handles are namespace-validated before reuse, and `_adapter_handle_cache.pop(pipeline_id, None)` runs during unregister."

### Rule 27.2: Fresh Handle on Cache Miss

Constraint: On cache miss or namespace mismatch, scheduler must call `ray.get_actor(f"schedrl:adapter:{pipeline_id}", namespace=registered_namespace)` and must never create adapter actors itself.

Audit Prompt: "Examine `_get_or_lookup_adapter_handle_locked`. Verify actor lookup uses pipeline-specific namespace from `pipeline_registry`, not hardcoded `'schedrl'`. Verify scheduler never calls `ray.remote(...Adapter...)` for creation."

---

## 28. Scheduling Cycle Phase Ordering & Snapshot Consistency

### Rule 28.1: Phase Execution Order Is Load-Bearing

Constraint: Inside `scheduling_cycle`, phases must execute in strict order: Phase 0.5 (planned releases) → Phase 2 (non-generation pending) → Phase 3 (generation gap-ratio) → Phase 4 (validation) → Phase 5 (resize RPCs, outside lock) → Phase 6 (commit). Phase 2 must complete before Phase 3 because `non_gen_reserved_gpus` (populated in Phase 2) is subtracted from `planned_available_gpus` before Phase 3 reads it.

Audit Prompt: "Review `scheduling_cycle` in `scheduler.py`. Verify Phase 2 populates `non_gen_reserved_gpus` before Phase 3 calls `_plan_generation_gap_ratio`. Verify the subtraction `planned_available_gpus -= non_gen_reserved_gpus` occurs after Phase 2 completes. Flag any code that reorders phases."

### Rule 28.2: Plan-to-Commit State Invalidation Window

Constraint: The plan is built under `_lock`, then the lock is released for `_execute_resize_calls` (resize RPCs), and re-acquired for commit. State can change during this window. If `unregister_pipeline` runs during the RPC window, commit raises `RuntimeError("Planned allocation has no pending waiter")` at the `_has_pending_request_locked` check.

Audit Prompt: "Review `scheduling_cycle` around the lock release between planning and commit. Trace what happens if `unregister_pipeline` runs during the RPC window. Verify `_apply_plan_and_signal` checks `_has_pending_request_locked` for `signal_pending_allocation_ops` and raises on missing waiter. Verify GPUs pre-removed from `idle_gpus` at line ~2214 are not leaked on this error path."

### Rule 28.3: Shrink-Before-Expand Execution Order Across All Pipelines

Constraint: `_execute_resize_calls` must issue ALL shrink RPCs (all pipelines) concurrently first and await them via `asyncio.gather`, then issue ALL expand RPCs and await them. No expand RPC may be dispatched before all shrinks complete. This prevents VRAM OOM during reallocation.

Audit Prompt: "Review `_execute_resize_calls` in `scheduler.py`. Verify `asyncio.gather(*shrink_tasks)` completes before any expand task is created or dispatched. Flag any code that interleaves shrink and expand RPCs."

---

## 29. Topology Initialization Ordering

### Rule 29.1: ResourceManager.init_topology Before Scheduler.initialize

Constraint: `resource_manager.init_topology(required_gpus_per_node=...)` must be called and awaited BEFORE `scheduler.initialize(resource_manager=...)`. The scheduler raises `RuntimeError("ResourceManager topology not initialized")` if this ordering is violated.

Audit Prompt: "Review `orchestrator._ensure_scheduler_singleton`. Verify `init_topology.remote()` is awaited via `ray.get()` before `scheduler.initialize.remote()`. Verify the check at `scheduler.py:~992-995`."

### Rule 29.2: _topology_ready Gates All Public RPCs

<!-- Enhancement note: gate applies to topology-dependent methods; some state-only methods intentionally skip it. -->
Constraint: All topology-dependent methods (`register_pipeline*`, `request_gpus`, `release_gpus`, `release_and_request_gpus`, `notify_ready_to_release`, `scheduling_cycle`) must `await self._topology_ready.wait()`. State-only methods (`admit_pipeline`, `unregister_pipeline`, `report_progress`) may skip direct gating and rely on orchestrator initialization order.

Audit Prompt: "Grep for `_topology_ready.wait()` in `scheduler.py`. Verify all topology-dependent methods call it, and `_topology_ready.set()` occurs once in `initialize()`. Do not flag state-only methods that intentionally avoid topology gating."

### Rule 29.3: Homogeneous GPU Nodes Required

Constraint: `ResourceManager` requires all GPU nodes have identical GPU counts. Heterogeneous clusters fail fast at `init_topology`. `ResourceManager.init_topology()` may only be called once per instance.

Audit Prompt: "Review `ResourceManager.init_topology`. Verify it checks that all nodes report the same GPU count and raises clearly for heterogeneous clusters. Verify a second call to `init_topology` is rejected (idempotency guard)."

---

## 30. _apply_plan_and_signal Commit Ordering

### Rule 30.1: Commit Operation Ordering Is Fixed

Constraint: Inside `_apply_plan_and_signal`, operations must execute in this exact order: (1) shrinks, (2) cluster removals, (3) pending allocations + their signals, (4) generation expansions state commit, (5) generation expansion signals, (6) planned release signals. Reordering causes GPU state inconsistency.

Audit Prompt: "Review `_apply_plan_and_signal` in `scheduler.py`. Verify the six phases execute in documented order. Flag any code that reorders operations between the shrink and planned-release-signal phases."

### Rule 30.2: signal_pending_allocation_ops Requires Active Waiter at Commit Time

Constraint: For each `signal_pending_allocation_ops` entry in the plan, `_has_pending_request_locked` must return `True` at commit time. If no waiter exists (cancelled during the RPC window between planning and commit), `RuntimeError` is raised immediately. GPUs already removed from `idle_gpus` must not be left stranded.

Audit Prompt: "Review `_apply_plan_and_signal` handling of `signal_pending_allocation_ops`. Verify the `_has_pending_request_locked` check. Verify whether GPUs removed from `idle_gpus` are returned to `idle_gpus` on the error path, or if they are permanently lost."

---

## 31. Request Deduplication & Idempotency

### Rule 31.1: One Pending Request Per cluster_id Across All Priority Buckets

Constraint: `request_gpus` must reject a second pending request for the same `cluster_id` regardless of which priority bucket it would go into. `_has_any_pending_request_locked` scans all 7 priority buckets. Duplicate raises `RuntimeError`.

Audit Prompt: "Review `request_gpus` at the duplicate check. Verify `_has_any_pending_request_locked` searches all priority buckets, not just the requested one. Verify it raises `RuntimeError` on duplicate."

### Rule 31.2: Idempotent Return for Already-Allocated Clusters

Constraint: If `request_gpus` is called for a `cluster_id` that already has an active allocation with matching priority and non-empty `gpu_ids` (or `active_dp_ranks` for GENERATION), it must return the existing allocation immediately without creating a new `PendingRequest`.

Audit Prompt: "Review `request_gpus` early-return path. Verify the check for matching priority and non-empty `gpu_ids` or `active_dp_ranks`. Verify it returns the existing allocation immediately."

---

## 32. TP Group Topology Validation

### Rule 32.1: Contiguous TP Groups Within Node Boundaries

Constraint: For `tp_size` in {2, 4, 8}, each TP group must consist of contiguous GPU IDs all within one node. For `tp_size >= required_gpus_per_node`, groups must align to node boundaries. For `tp_size == 1`, no topology check is applied.

Audit Prompt: "Review `_validate_and_canonicalize_device_mapping` in `scheduler.py`. Verify the contiguity check for tp_size in {2,4,8}. Verify the node-boundary alignment check for tp_size >= required_gpus_per_node. Verify tp_size=1 is a no-op."

### Rule 32.2: device_mapping Length Divisible by tp_size

Constraint: `len(device_mapping) % tp_size != 0` must raise `ValueError`. This check must be enforced both in `validate_register_pipeline` (orchestrator-side) and `register_pipeline_topology` (scheduler-side).

Audit Prompt: "Verify the divisibility check exists in both `validate_register_pipeline` in `validation.py` and `register_pipeline_topology` in `scheduler.py`. Flag any location that misses this check."

---

## 33. Cross-Pipeline GPU Isolation

Rule Class: KNOWN_GAP_CHECK

### Rule 33.1: Cross-Pipeline GPU Overlap Not Checked at Registration Time

Constraint: Registration does NOT enforce cross-pipeline GPU uniqueness. Two pipelines can register identical `device_mapping`s. GPU exclusivity is only enforced at plan commit time by Condition 10 in `validate_execution_plan`.

Audit Prompt: "Review `register_pipeline_topology`. Verify there is NO cross-pipeline device_mapping overlap check. Verify Condition 10 in `validation.py` catches cross-pipeline GPU overlap at plan commit time."

### Rule 33.2: Gap-Ratio Planner Can Shrink Foreign Pipelines

Constraint: The gap-ratio planner can issue shrink operations on ANY pipeline with a negative gap, not just the requesting pipeline. Pipeline A's progress report can trigger a shrink of pipeline B's workers.

Audit Prompt: "Review `_plan_generation_gap_ratio`. Verify donor DP-rank selection can cross pipeline boundaries. Verify the adapter RPC for the shrunken pipeline is correctly resolved."

---

## 34. Config Validation Gaps

Rule Class: KNOWN_GAP_CHECK

### Rule 34.1: cluster_name Not Validated Against Known Set at Registration

Constraint: `register_pipeline_topology` accepts arbitrary `cluster_name` strings without checking against the known set `{"actor_train", "actor_infer", "critic", "reference"}` defined in `parse_cluster_id`. A pipeline registering `cluster_name="foobar"` (or `"reward"`) succeeds at registration but fails later when `parse_cluster_id` is called.

Audit Prompt: "Review `register_pipeline_topology`. Verify cluster_name is NOT validated against the known set. Review `parse_cluster_id` in `types.py`. Assess whether registration should reject unknown cluster names to fail earlier."

### Rule 34.2: actor_infer Presence Not Enforced at Orchestrator Level

Constraint: The `actor_infer` cluster presence check exists only in `scheduler.register_pipeline_topology`, NOT in `validate_register_pipeline` (called at orchestrator level). A malformed registration passes orchestrator validation but fails inside the scheduler actor.

Audit Prompt: "Verify `validate_register_pipeline` in `validation.py` does NOT check for `actor_infer` presence. Verify `register_pipeline_topology` in `scheduler.py` does. Assess if this check should be duplicated in `validate_register_pipeline`."

### Rule 34.3: Progress Metrics Non-Numeric Values Crash Scheduler

Constraint: `metrics["remaining"]` is cast with `float(...)` inside `_pipeline_progress_totals_locked`, which is called from within the scheduling cycle. A non-numeric value raises `ValueError`, propagates through `scheduling_cycle`, and triggers `_fail_fast_shutdown`.

Audit Prompt: "Trace `float(metrics['remaining'])` in `scheduler.py`. Verify a non-numeric value causes an unhandled `ValueError`. Verify this propagates to the scheduling cycle exception handler and triggers fail-fast shutdown."

---

## 35. Async Event Lifecycle Correctness

Rule Class: KNOWN_GAP_CHECK

### Rule 35.1: _fail_fast_shutdown Is Fire-and-Forget

Constraint: `_fail_fast_shutdown` calls `orchestrator.shutdown.remote(...)` without awaiting the returned `ObjectRef`. If the orchestrator is unreachable, the method returns silently without triggering any shutdown, while the scheduler continues running.

Audit Prompt: "Review `_fail_fast_shutdown` in `scheduler.py`. Verify `orchestrator.shutdown.remote(...)` is NOT awaited. Verify the method returns without error even when `ray.get_actor` raises (orchestrator already dead). Assess whether this silent failure is acceptable."

### Rule 35.2: _loop_task Never Cancelled on Shutdown

Constraint: `shutdown()` acquires the lock and cleans up tracing but never calls `_loop_task.cancel()`. The scheduling loop continues running after `shutdown()` returns.

Audit Prompt: "Review `shutdown()` in `scheduler.py`. Verify `_loop_task` is not cancelled. Assess whether scheduling cycles after `shutdown()` are harmless or could cause state corruption."

### Rule 35.3: _topology_ready.wait() Has No Timeout

Constraint: All public methods calling `_topology_ready.wait()` have no timeout. If `initialize()` is never called, ALL public RPCs block forever inside the actor with no escape path.

Audit Prompt: "Grep for `_topology_ready.wait()` in `scheduler.py`. Verify none has a timeout parameter. Assess whether a timeout should be added to prevent permanent actor deadlock."

---

## 36. Assertions Stripped with Python -O

Rule Class: KNOWN_GAP_CHECK

### Rule 36.1: Critical Invariants Encoded as assert Statements

Constraint: Several critical invariants use `assert` instead of `raise`: (1) one-request-per-cluster trace state, (2) idle/non-gen GPU disjointness checks in `_snapshot_generation_dp_workers` and `_plan_generation_gap_ratio`. Running Python with `-O` strips these checks silently.

Audit Prompt: "Grep for `assert` in `scheduler.py`. Classify each as: (a) debug-only, safe to strip, or (b) correctness invariant that should be a `raise`. For category (b), assess whether they should be converted to proper `if not ...: raise AssertionError(...)`."

---

## 37. NCCL Group Lifecycle Ordering

### Rule 37.1: All NCCL Participants Must Init Before First Collective

Constraint: In `_make_collective_group`, all sender and receiver `setup_collective_group` remote calls must complete (via `ray.get(refs)`) before any broadcast is issued. A premature broadcast causes an NCCL hang.

Audit Prompt: "Review `_make_collective_group` in `model_update_service.py`. Verify `ray.get(refs)` waits for ALL setup calls (both sender and receiver sides) before returning. Flag any code that issues a broadcast before all participants have initialized."

### Rule 37.2: NCCL Group Destroy Must Precede dist.barrier() in SchedRL Path

Constraint: In the SchedRL selective sync path, `ncclCommDestroy` must happen inside `selective_sync_active_cache` on the sender side, BEFORE any `dist.barrier()` call spanning the same process group. Calling `ncclCommDestroy` after `dist.barrier()` on the same communicator causes a deadlock.

Audit Prompt: "Review `selective_sync_active_cache` in `megatron_strategy.py`. Verify NCCL group destruction occurs before any `dist.barrier()` call. Flag any code path where barrier precedes group destruction."

### Rule 37.3: ReloadableProcessGroup Null Check

Constraint: `ReloadableProcessGroup` wraps NCCL groups and sets `self.group = None` after `destroy_process_groups`. Any collective operation on a `None` group raises `RuntimeError`. Callers must call `reload_process_groups` before issuing any collective after destroy.

Audit Prompt: "Review `ReloadableProcessGroup` in `offload_nccl.py`. Verify all collective methods check `self.group is not None`. Verify `reload_process_groups` is called after `destroy_process_groups` before any collective operation. Flag any path that may use a destroyed group."

---

## 38. Weight Versioning Consistency

### Rule 38.1: Forward Version Application Is Forbidden

Constraint: In `ModelUpdateService.selective_update`, if `cached_global_step > requested_global_step`, a `ValueError` is raised immediately. Applying future (newer) weights to a past step is a hard error. Stale weights (behind) log a warning but proceed.

Audit Prompt: "Review `selective_update` in `model_update_service.py`. Verify `cached_global_step > max_allowed_cached_step` raises `ValueError`. Verify `cached_global_step < max_allowed_cached_step` only logs a warning and continues."

### Rule 38.2: Cache Key Must Exist Before Promote

Constraint: `promote_active_checkpoint(version, step)` must raise `RuntimeError` if the key `(version, step)` is not in `_cache_map`. The required call ordering is: `_build_latest_bucket_cache` → `promote_active_checkpoint` → `selective_sync_active_cache`.

Audit Prompt: "Review `promote_active_checkpoint` in `megatron_strategy.py`. Verify it checks `cache_key in self._cache_map` and raises `RuntimeError` on miss. Verify calling `selective_sync_active_cache` before `promote_active_checkpoint` raises."

### Rule 38.3: Bucket Count Consistency Across PP Ranks

Constraint: All PP ranks building the sender cache must produce the same number of buckets. `refresh_sender_cache` validates this and raises `ValueError` on mismatch before any cache state is committed.

Audit Prompt: "Review `refresh_sender_cache` in `model_update_service.py`. Verify it checks that all PP ranks report the same `num_buckets`. Verify `ValueError` is raised before cache state is written."

---

## 39. Offload State Machine Guards

### Rule 39.1: Idempotent Offload/Reload Guards via offloaded_states Set

Constraint: `needs_offload(target, ...)` returns `True` only when `target not in offloaded_states` — prevents double-offload. `needs_reload(target, ...)` returns `True` only when `target in offloaded_states` — prevents double-reload. `offloaded_states.add(target)` must run after offload; `offloaded_states.remove(target)` must run after reload.

Audit Prompt: "Review `needs_offload` and `needs_reload` in `offload_states.py`. Verify the set membership checks. Verify `offloaded_states.add(target)` after offload and `offloaded_states.remove(target)` after reload are not skipped on any code path."

### Rule 39.2: CUDA Synchronize After Every Offload/Reload

Constraint: Both offload and reload paths must call `current_platform.synchronize()` before returning, to ensure all async `.to()` copies are complete. Missing sync causes silent data corruption from concurrent CUDA stream access.

Audit Prompt: "Review all offload and reload methods in `offload_states_patch.py`. Verify `current_platform.synchronize()` is called at the end of every offload and reload path. Flag any path that returns before synchronizing."

### Rule 39.3: ChainedOptimizer Last Sub-Optimizer Skips Grad Hook Registration

Constraint: In `ChainedOptimizer.reload_states`, all sub-optimizers except the LAST one reload normally. The last sub-optimizer reloads with `skip_grad_hook_register=True`.

Audit Prompt: "Review `ChainedOptimizer.reload_states` in `offload_states_patch.py`. Verify `[:-1]` indexing and `skip_grad_hook_register=True` on the last optimizer. Flag if the last optimizer also registers grad hooks."

---

## 40. Separated Model Update PP Stage Serialization

### Rule 40.1: Single PP Stage Holds Broadcast Lock at a Time

Constraint: `MegatronWeightUpdater._separated_model_update` uses a Ray named actor `Locker` to serialize broadcast across PP stages. Only one PP stage's trainer holds the lock at a time. The lock must be released in a `finally` block to avoid deadlock if broadcast fails.

Audit Prompt: "Review `_separated_model_update` in `model_update.py`. Verify `_model_update_locker.acquire.remote()` is polled in a loop. Verify `release.remote()` is called in a `try/finally` block. Flag any code path that could leave the lock acquired on exception."

### Rule 40.2: Broadcast Group Name Uniqueness Per PP/EP Rank

Constraint: Broadcast group names must include both PP rank and EP rank to ensure uniqueness: `f"{model_update_name}_pp{pp_rank}_ep{ep_rank}"`. Duplicate names cause NCCL rendezvous collisions.

Audit Prompt: "Review `_setup_broadcast_group` in `model_update.py`. Verify the name includes both `pp_rank` and `ep_rank`. Verify no two workers in different roles can produce the same group name."

---

## 41. IPC Handle Invalidation

### Rule 41.1: Source Tensor Must Outlive Receiver Deserialization

Constraint: When using CUDA IPC (`MultiprocessingSerializer.serialize(gpu_bucket)`), the source `gpu_bucket` tensor must remain alive until ALL receivers have finished deserializing and applying. Premature GC of the source tensor invalidates the IPC handle.

Audit Prompt: "Review `selective_model_update_from_cache` in `megatron_strategy.py`. Verify `gpu_bucket` reference is held alive until `ray.get(p2p_refs)` completes. Flag any code path where the source tensor could be garbage-collected before receivers finish."

### Rule 41.2: ROLL_schedrl Selective-Sync Barrier and Teardown Ordering

Constraint: In `selective_sync_active_cache`, sender-side collective groups must be destroyed **before** the final distributed barrier. Bucket tensors used for broadcast must stay alive until receiver apply RPCs complete (`ray.get(recv_refs)`). Violating this order can deadlock NCCL or invalidate transfer state.

Audit Prompt: "Review `selective_sync_active_cache` in `roll/distributed/strategy/megatron_strategy.py`. Verify per-bucket flow is: broadcast handles wait -> `ray.get(recv_refs)` -> bucket cleanup. Verify sender destroys collective groups (`collective.destroy_collective_group` and worker `destroy_collective_group.remote`) before `_safe_dist_barrier()`. Flag any path where barrier happens before group teardown."

---

## 42. Source-Type Immutability

### Rule 42.1: Full-Finetune vs Adapter Streams Are Mutually Exclusive Per Pipeline

Constraint: Once a pipeline reports progress with `adapter_id=None` (full-finetune mode), it cannot switch to adapter-specific reporting, and vice versa. `report_progress` raises `RuntimeError` on any mode switch within a single pipeline's lifetime.

Audit Prompt: "Review `report_progress` in `scheduler.py`. Verify the mutex check between `adapter_id=None` and adapter-specific streams for the same `pipeline_id`. Verify `RuntimeError` is raised on any mode switch."

---

## 43. Wakeup Event Semantics

### Rule 43.1: Wakeup Event Cleared Only on Event-Driven Wakeups

Constraint: `_wakeup_event.clear()` is called only when the loop was woken by `_wakeup_event.set()`, NOT on timeout-driven background rebalance wakeups. An event set during a timeout-triggered cycle persists and is picked up by the next iteration.

Audit Prompt: "Review `_central_scheduling_loop` in `scheduler.py`. Verify `_wakeup_event.clear()` is conditional on `woke_by_event=True`. Verify that events set during a timeout-triggered cycle are preserved for the next iteration."

### Rule 43.2: All State-Mutating RPCs Must Set _wakeup_event

Constraint: Every RPC that mutates scheduler state — `request_gpus`, `release_gpus`, `release_and_request_gpus`, `report_progress`, `unregister_pipeline`, `notify_ready_to_release` — must call `self._wakeup_event.set()` after its mutation. Missing wakeup delays the next scheduling cycle by up to 1 second.

Audit Prompt: "Grep for `_wakeup_event.set()` in `scheduler.py`. Verify it is called in ALL state-mutating RPC methods. Flag any method that mutates `_state` but does not set the wakeup event."

---

## 44. notify_ready_to_release Idempotency

### Rule 44.1: Second Caller Reuses First Caller's Event and Snapshot

Constraint: If `notify_ready_to_release` is called twice for the same `cluster_id`, the second caller reuses the first caller's `PendingPlannedReleaseRequest`, including its `result_released_gpu_ids` snapshot taken at first-call time. If DP ranks changed between calls, the second caller receives stale data.

Audit Prompt: "Review `notify_ready_to_release` in `scheduler.py`. Verify the idempotency path reuses the existing `req` and `event`. Assess whether `result_released_gpu_ids` needs to be refreshed if DP ranks changed between the two calls."

---

## 45. Selective Sync Sender Eligibility

### Rule 45.1: Only dp_rank=0, tp_rank=0, cp_rank=0 Workers Can Be Senders

Constraint: In `_select_sender_ranks_by_pp`, only workers at `dp_rank==0, tp_rank==0, cp_rank==0` can be senders because only they own the CPU bucket cache built by `_build_latest_bucket_cache`. Selecting any other rank as sender would broadcast stale or zeroed data.

Audit Prompt: "Review `_select_sender_ranks_by_pp` in `model_update_service.py`. Verify the sender eligibility filter enforces `dp_rank==0, tp_rank==0, cp_rank==0`. Verify the comment documenting this constraint."

### Rule 45.2: Exactly One Sender Per PP Rank

Constraint: `selective_update` validates that exactly one sender exists per PP rank (`len(pp_plan) != 1` raises `RuntimeError`). Multiple senders per PP rank would cause receiver data corruption from conflicting concurrent broadcasts.

Audit Prompt: "Review `selective_update` in `model_update_service.py`. Verify the one-sender-per-PP-rank check and the `RuntimeError` message. Verify this check runs before any NCCL group setup or broadcast."

---

## 46. Audit Prompt Rigor (Applies to ALL Rules 1–45)

<!-- Enhancement note: these rules harden every existing audit prompt into reproducible, falsifiable checks with concrete evidence requirements. -->

### Rule 46.1: Mandatory Evidence Triplet per Finding

Constraint: Every audit response must include three artifacts for each rule: (1) exact code anchor (`file + function + line`), (2) proof snippet or precise logic trace, (3) pass/fail verdict with one-sentence rationale.

Audit Prompt: "For the target rule, return exactly: `Anchor`, `Evidence`, and `Verdict`. `Anchor` must include file path and line. `Evidence` must cite concrete statements or branches. `Verdict` must be only `PASS` or `FAIL` with one sentence."

### Rule 46.2: Contradiction Search Is Required

Constraint: Audits must search for contradictory code paths, not just confirm the happy path. A rule is PASS only if both positive proof and contradiction search are satisfied.

Audit Prompt: "After proving the intended path, run a contradiction search for alternative branches, helper methods, and call sites that bypass the invariant. If any bypass exists, return `FAIL`."

### Rule 46.3: Mutation-Site Completeness

Constraint: For state invariants, audits must enumerate ALL write sites of the guarded state and verify the invariant at each site. Sampling one function is insufficient.

Audit Prompt: "Enumerate all write sites for the target state variables (via symbol search). For each write site, verify rule compliance. Return `FAIL` if any write site is unverified."

### Rule 46.4: Exact Failure Trigger Definition

Constraint: Every fail-fast rule must state the exact exception type and trigger condition. Generic wording like 'should fail' is not sufficient.

Audit Prompt: "Identify the exact trigger predicate and the exact exception path (`raise RuntimeError(...)`, shutdown call, or equivalent). Mark `FAIL` if failure behavior is ambiguous or implicit."

### Rule 46.5: Temporal Ordering Proof for Async Rules

Constraint: For async/locking rules, audits must prove ordering around `await` boundaries (what happens before await, during external window, and after re-entry).

Audit Prompt: "Construct a three-step timeline (`before await`, `await window`, `after await`) for the target function and verify invariant preservation across the full timeline."

---

## 47. Plan/Commit Atomicity and Leak Prevention

### Rule 47.1: Idle GPU Rollback on Commit Failure

Constraint: If `_apply_plan_and_signal` raises after GPUs were removed from `idle_gpus`, those GPUs must be deterministically restored before exit (or process must terminate immediately after a guaranteed global shutdown). Silent loss is forbidden.

Audit Prompt: "Trace all `_apply_plan_and_signal` exception paths after idle GPU subtraction. Verify each path either restores identical GPU IDs to `idle_gpus` or triggers immediate fail-fast shutdown that cannot continue scheduling."

### Rule 47.2: Pending Request Settlement Is Exactly-Once

Constraint: Each `PendingRequest` must be settled exactly once: either signaled with allocation or failed/cancelled with cleanup. Double-signal and never-signal states are both invalid.

Audit Prompt: "Track lifecycle of one `PendingRequest` object from creation to terminal state. Verify there is exactly one terminal transition and no code path leaves it unresolved."

---

## 48. Lock Ordering and Deadlock Safety

### Rule 48.1: Global Lock Acquisition Order Is Acyclic

Constraint: If both scheduler `_lock` and request `routing_lock` can be used in the same call chain, acquisition order must be globally consistent (single order only). Mixed order is a deadlock risk.

Audit Prompt: "Build a lock-order graph from call sites acquiring `_lock` and `routing_lock`. Verify no path acquires them in opposite orders. Return `FAIL` on any potential cycle."

### Rule 48.2: No External RPC While Holding Critical Locks

Constraint: Calls that cross actor boundaries (`.remote`, `ray.get`, blocking waits) must not occur while holding `_lock` or `routing_lock`, except for explicitly documented and bounded critical sections.

Audit Prompt: "Search lock scopes for external RPC calls and blocking waits. Verify lock is released before cross-actor operations. Flag any undocumented exception as `FAIL`."

---

## 49. Register/Unregister Symmetry

### Rule 49.1: Registration Indexes Must Be Fully Reversible

Constraint: Every index/cache/map populated during `register_pipeline` (or topology registration) must have a matching deletion path in `unregister_pipeline`/`kill_pipeline`. Partial teardown is invalid.

Audit Prompt: "List all registration-time writes to scheduler/orchestrator state. For each write target, verify an unregister/kill deletion path exists. Return `FAIL` for any orphaned entry."

### Rule 49.2: Unregister Purges Pipeline-Scoped Pending State

Constraint: `unregister_pipeline` must remove pipeline-scoped pending allocations, planned releases, and progress-tracking entries so future cycles cannot act on stale pipeline IDs.

Audit Prompt: "Trace `unregister_pipeline` and confirm purge of pending request buckets, planned release structures, and progress maps for that `pipeline_id`. Flag stale references as `FAIL`."

---

## 50. Request Lifecycle Invariants

### Rule 50.1: Request Must Not Exist in Multiple Lifecycle Sets

Constraint: A request ID must never simultaneously exist in more than one lifecycle container (for example pending + running, or running + completed bookkeeping) at any instant.

Audit Prompt: "Map request state transitions and container membership updates. Verify atomic remove-then-add ordering across transitions. Any overlap window is `FAIL`."

### Rule 50.2: Abort Completion Must Cleanup All Secondary Indexes

Constraint: On abort ACK/completion, request ID must be removed from primary and secondary indexes (`running_requests`, rank-index maps, sticky mappings, adapter-specific trackers).

Audit Prompt: "Follow abort-complete path and enumerate all indexes touched during request admission. Verify symmetric cleanup for each index. Missing cleanup in any index is `FAIL`."

---

## 51. Progress Accounting Integrity

### Rule 51.1: Remaining/Collected Counters Are Non-Negative and Bounded

Constraint: Progress counters used by scheduling (`remaining`, `collected`, target totals) must stay within valid numeric bounds and never become negative.

Audit Prompt: "Audit every write to progress counters. Verify guard checks enforce numeric type and non-negative bounds before values enter scheduling math."

### Rule 51.2: Generation Planner Must Use a Single Snapshot per Cycle

Constraint: Gap-ratio planning within one cycle must use a consistent snapshot of progress totals. Mixing pre- and post-mutation values in one cycle is invalid.

Audit Prompt: "Inspect `scheduling_cycle` and gap-ratio helpers. Verify progress totals used for planning are read once per cycle (or explicitly versioned) and not recomputed mid-plan from mutated state."

---

## 52a. Registration Policy Parity (Orchestrator vs Scheduler)

### Rule 52a.1: Reward Cluster Must Be CPU-Only in Both Validation Layers

Constraint: Both `protocol/validation.py::validate_register_pipeline` and `scheduler.py::register_pipeline_topology` must reject non-empty `reward.device_mapping`.

Audit Prompt: "Verify both validation layers reject `cluster_name='reward'` with non-empty `device_mapping`. Return `FAIL` if only one layer enforces this."

### Rule 52a.2: Non-actor_infer GPU Overlap Is Forbidden, actor_infer Overlap Is Allowed

Constraint: Registration must disallow GPU overlap across non-`actor_infer` clusters, while allowing `actor_infer` colocation by policy.

Audit Prompt: "Trace overlap checks in orchestrator-side and scheduler-side registration validation. Verify overlaps among non-`actor_infer` clusters raise, while `actor_infer` overlap is not rejected by that check."

### Rule 52a.3: Cluster Key Set Equality Must Hold Across TP Config and Device Mapping

Constraint: `cluster_tp_configs.keys()` must exactly match `cluster_device_mappings.keys()` in both validation layers; missing/extra cluster names must fail fast.

Audit Prompt: "Verify both validation layers enforce exact key-set equality and return explicit missing-key diagnostics."

---

## 53a. Cluster ID Grammar and Parser Safety

### Rule 53a.1: cluster_id Character/Length Gate Must Run Before Parsing

Constraint: `validate_cluster_id` must enforce non-empty string, max length (`<=256`), and regex-safe characters before `parse_cluster_id` suffix parsing.

Audit Prompt: "Review `validate_cluster_id` and `parse_cluster_id` in `scheduler/types.py`. Verify parser always calls validator first and rejects invalid charset/length before suffix logic."

### Rule 53a.2: parse_cluster_id Must Be Suffix-Aware With Known Cluster Set Only

Constraint: `parse_cluster_id` must accept only known suffixes and must not use ambiguous `rsplit('_', 1)` fallback.

Audit Prompt: "Verify `parse_cluster_id` matches against explicit known suffixes and raises `ValueError` for unknown suffixes. Flag any fallback split parser."

### Rule 53a.3: Registration/Parsing Cluster-Set Drift Must Be Explicitly Audited

Constraint: If registration accepts a cluster name that `parse_cluster_id` cannot decode (for example `reward`), this must be documented as an intentional unsupported path or rejected during registration.

Audit Prompt: "Compare cluster names accepted by registration with names accepted by `parse_cluster_id`. Return `FAIL` if they diverge without explicit guard or documented intent."

---

## 54. Fatal-Path Waiter Settlement

### Rule 54.1: _signal_all_waiters_with_error Must Settle Both Pending Buckets and Planned Releases

Constraint: `_signal_all_waiters_with_error` must set `error` and `event` for every `PendingRequest` and every `PendingPlannedReleaseRequest`, then clear both stores.

Audit Prompt: "Review `_signal_all_waiters_with_error` in `scheduler.py`. Verify complete fan-out to all waiters, queue-slice close for each pending request, and post-condition that both pending stores are empty."

### Rule 54.2: Central Loop Exception Path Must Signal Waiters Before Shutdown RPC

Constraint: `_central_scheduling_loop` exception handler must call `_signal_all_waiters_with_error` under lock before invoking `_fail_fast_shutdown`.

Audit Prompt: "Trace exception handler ordering in `_central_scheduling_loop`. Verify waiter signaling precedes fail-fast shutdown invocation."

### Rule 54.3: unregister_pipeline Malformed-State Detection Must Trigger Global Fail-Fast

Constraint: If malformed `cluster_id` is found while scanning active allocations/pending queues/planned releases during unregister, scheduler must signal all waiters, invoke `_fail_fast_shutdown`, and raise.

Audit Prompt: "Audit malformed `cluster_id` branches in `unregister_pipeline`. Verify all three actions occur: waiter signal, fail-fast shutdown call, and raised exception."

---

## 55. Non-Generation Preemption Semantics

### Rule 55.1: Non-GEN Demand May Preempt GEN Donors in Phase 2

Constraint: During Phase 2 planning, missing GPUs for non-generation requests may be sourced by planning generation DP-rank shrinks.

Audit Prompt: "Review Phase 2 logic in `scheduling_cycle`. Verify donor selection iterates generation allocations and appends `SchedGuidedShrinkOp` for required bundles."

### Rule 55.2: Planned-Free GPU Double-Counting Must Be Prevented

Constraint: If a donor `dp_rank` is already listed in shrink ops, planner must not add its bundle to `planned_available_gpus` a second time.

Audit Prompt: "Locate `already_in_shrink` handling in Phase 2 planning. Verify bundle is added to `planned_available_gpus` only for newly-added shrink ranks."

### Rule 55.3: Generation Budget Must Exclude Non-GEN Reservations Before Gap-Ratio Planning

Constraint: `planned_available_gpus` must subtract `non_gen_reserved_gpus` before calling `_snapshot_generation_dp_workers` / `_plan_generation_gap_ratio`.

Audit Prompt: "Verify subtraction ordering in `scheduling_cycle`: `planned_available_gpus -= non_gen_reserved_gpus` occurs before generation planning starts."

---

## 56. Generation Wake-Only Signaling

### Rule 56.1: Empty-Allocation Signal for Pending GEN Requests Is Allowed Only When No Inactive DP Workers Exist

Constraint: Scheduler may emit `SignalPendingAllocationOp(..., gpus_to_allocate=[])` for pending generation requests only when target pipeline has active workers and zero inactive workers.

Audit Prompt: "Review Phase 3 pre-gap-ratio pending-GEN loop. Verify wake-only signaling requires `inactive_dp_workers` empty and `active_dp_workers` non-empty."

### Rule 56.2: Wake-Only Commit Must Return Existing Allocation, Not Empty Result

Constraint: In `_apply_plan_and_signal`, when handling wake-only op (`gpus_to_allocate=[]`), if an existing allocation is present it must signal pending waiter with existing `gpu_ids`, not `[]`.

Audit Prompt: "Audit wake-only branch in `_apply_plan_and_signal`. Verify existing allocation path returns current `gpu_ids` and only falls back to `[]` when no allocation exists."

---

## 57. Resize Preparation Guards

### Rule 57.1: Same dp_rank Cannot Be Both Removed and Added in One Cycle for a Pipeline

Constraint: `_prepare_resize_calls_locked` must raise `RuntimeError` if `set(dp_ranks_to_remove) ∩ set(dp_ranks_to_add)` is non-empty for the same pipeline.

Audit Prompt: "Review `_prepare_resize_calls_locked`. Verify overlap detection and fail-fast error message for overlapping dp ranks."

### Rule 57.2: Adapter Handle Resolution Must Fail Fast on Namespace Drift

Constraint: If cached adapter namespace differs from current registered namespace, scheduler must refresh handle via `ray.get_actor` in current namespace; lookup failure must raise immediately.

Audit Prompt: "Trace cache-hit and cache-miss paths in `_get_or_lookup_adapter_handle_locked`. Verify namespace mismatch forces lookup and unresolved actor raises `RuntimeError`."

---

## 58. Planned Release API Contracts

### Rule 58.1: notify_ready_to_release Input Contract Is XOR(pipeline_id, cluster_id)

Constraint: `notify_ready_to_release` must require exactly one of `pipeline_id` or `cluster_id`; both missing or both provided must raise `ValueError`.

Audit Prompt: "Review argument validation at method entry. Verify XOR semantics and explicit error on invalid combinations."

### Rule 58.2: notify_ready_to_release Is Generation-Only and Returns Immediate [] for Zero Active DP

Constraint: `notify_ready_to_release` must reject non-generation clusters and return `[]` immediately when generation allocation has no active DP ranks.

Audit Prompt: "Trace early validation and return branches in `notify_ready_to_release`. Verify non-GEN rejection and zero-active fast return."

### Rule 58.3: notify_ready_to_release Timeout Is Fatal

Constraint: `asyncio.TimeoutError` while waiting for planned release acknowledgment must trigger `_fail_fast_shutdown(...)` before re-raising timeout.

Audit Prompt: "Inspect timeout exception branch in `notify_ready_to_release`. Verify fail-fast shutdown call precedes timeout propagation."

---

## 59. Progress Timestamp Normalization

### Rule 59.1: oldest_unfinished_creation_ts Must Fallback to fifo_timestamp

Constraint: If `report.oldest_unfinished_creation_ts` is `None` and `fifo_timestamp` exists, scheduler must normalize by storing `oldest_unfinished_creation_ts=fifo_timestamp` before persistence.

Audit Prompt: "Review `report_progress` normalization path and `normalize_progress_oldest_ts`. Verify fallback rewrite occurs before report is inserted into `latest_progress_by_pipeline`."

### Rule 59.2: Progress Writes Require Registered Pipeline

Constraint: `report_progress` must fail fast with `RuntimeError` when `report.pipeline_id` is not registered in `pipeline_registry`.

Audit Prompt: "Inspect `report_progress` under lock. Verify registration existence check runs before mutating `latest_progress_by_pipeline`."

---

## 52b. Client Initialization & Race Condition Handling

### Rule 52b.1: Orchestrator Get-or-Create Backoff & Race Mitigation
Constraint: `_get_or_create_orchestrator` must attempt to get the actor first. On `ValueError`, if `create_if_missing` is true, it iterates over `opts.backoff_s`. If `ray.remote(Orchestrator).remote()` raises an exception (e.g., due to another client winning the race), it must immediately catch it and try `ray.get_actor` again in the `except` block.
Audit Prompt: "Review `_get_or_create_orchestrator` in `client.py`. Verify the outer loop uses `opts.backoff_s`. Verify the inner `try...except Exception:` block around `Orchestrator.remote()`. Confirm the `except` block immediately calls `ray.get_actor` to catch the winner of the race. Flag any implementation that sleeps before retrying the `get_actor` in the exception path."

### Rule 52b.2: Library Mode Rank 0 Connection Gate
Constraint: `schedrl.init()` must parse the `RANK` environment variable (defaulting to 0). If `rank != 0`, it must silently return `None`. It must only proceed to `connect()` for rank 0, ensuring only the master process connects to the orchestrator.
Audit Prompt: "Review `schedrl.init()` in `init.py`. Verify `os.environ.get("RANK", "0")` is parsed. Verify `if rank != 0: return None`. Confirm `connect()` is only executed when `rank == 0`. Flag any execution path that allows non-zero ranks to connect."

---

## 53b. Node Affinity & Timeouts

### Rule 53b.1: Strict Head Node Affinity for Orchestrator
Constraint: The Orchestrator actor MUST be scheduled on the Ray head node. `head_node_affinity_strategy` must be called with `soft=False` to ensure strict placement. Hard-failing is preferred over placing the orchestrator on a worker node.
Audit Prompt: "Review `head_node_affinity_strategy` in `ray_head.py`. Verify it returns `NodeAffinitySchedulingStrategy` with `soft=False`. Review `_get_or_create_orchestrator`. Verify this strategy is explicitly passed to the orchestrator's `.options(scheduling_strategy=...)`. Flag `soft=True` or missing strategy assignment."



