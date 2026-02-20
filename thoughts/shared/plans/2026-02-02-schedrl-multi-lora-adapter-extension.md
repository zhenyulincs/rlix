# SchedRL Multi‑LoRA Adapter Extension (ROLL SchedRL Integration) Implementation Plan

**Date**: 2026-02-02 (Updated: 2026-02-19; Phase 5 added 2026-02-19)

## Overview

Port the multi-LoRA feature from `external/ROLL_multi_lora` into `external/ROLL_schedrl` to enable SchedRL-controlled multi-LoRA training. The goal is to make the multi-LoRA pipeline coordinator integrate with the SchedRL scheduler the same way the current concurrent pipeline does.

### Current State
- **external/ROLL_schedrl**: Has `SchedRLConcurrentPipeline` with full SchedRL integration (shrink/expand, `notify_ready_to_release`, selective sync via `ModelUpdateService`)
- **external/ROLL_multi_lora**: Has `AgenticMultiLoraPipeline` with multi-LoRA support (per-tag schedulers, `model_update_lora_subset`, partial GPU mode)
- **Gap**: ROLL_schedrl only supports single LoRA (via `actor_lora_target` check); multi-LoRA patterns exist in ROLL_multi_lora but aren't integrated with SchedRL

### Goal
Create `SchedRLMultiLoraPipeline` in `external/ROLL_schedrl` that:
1. Reuses existing multi-LoRA patterns from ROLL_multi_lora
2. Integrates with SchedRL scheduler like `SchedRLConcurrentPipeline` does
3. Supports per-adapter progress tracking and reporting
5. Eliminates Adapter Garbage Collection (GC) by enforcing static VRAM limits for fixed adapter sets
6. Operates under SchedRL's `sleep_level=2` constraint (GPU release only; actors remain alive)
7. Ensures isolated, asynchronous LR decay per adapter via `per_adapter` optimizer mode

---

## NEW IMPLEMENTATION PLAN (Porting Multi-LoRA to ROLL_schedrl)

Based on codebase research and feedback integration, here is the refined implementation plan.

### Critical Architecture Adjustments

#### 1. Architecture Decision: Sequential Adapter Training
We enforce strict **sequential processing** of adapters during training (`Adapter A -> Step`, then `Adapter B -> Step`). We deliberately avoid interleaved or concurrent multi-adapter training within a single `train_step_lora` call. This is a **foundational architecture decision** that ensures correctness across multiple dimensions.

- **Rationale (Correctness)**:
  - **Activation Checkpointing (AC)**: AC re-runs the forward pass during backward. If the adapter context changes (e.g., mixing adapters in one batch), the re-computation uses wrong weights, causing silent gradient corruption. Sequential processing ensures consistent global state.
  - **Gradient Accumulation (GA)**: Mixing adapters with different accumulation schedules breaks GA logic. Sequential processing respects per-adapter GA counters.
  - **Per-Adapter RNG**: Dropout masks and other random operations need isolation between adapters. Sequential processing allows us to save/restore per-adapter RNG states, preventing cross-adapter RNG pollution.
  - **Code Reuse**: This aligns with the existing single-LoRA training pattern, allowing us to reuse `inner_forward_step` and optimizer logic without complex modification.

#### 2. SchedRL Execution Model (`sleep_level=2` & Actor Lifespan)
Unlike `AgenticMultiLoraPipeline` which uses `partial_gpu_mode=True` (inference and training overlap), SchedRL requires **full GPU release** (`sleep_level=2`) during the training phase to allow for elastic resizing.
- **Consequence**: `SchedRLMultiLoraPipeline` will operate in a **sequential** cycle: `Expand -> Rollout (all adapters) -> Shrink -> Train (dirty adapters) -> Repeat`.
- **Actor Lifespan**: Codebase review confirms Ray actors are **NOT** destroyed during shrink. They remain alive; only GPU memory is freed via `offload_states()`.
- **State Location**: Optimizer and model states reside in the actor's **CPU RAM** during the "Sleep" phase. This removes the need for expensive checkpoint-to-disk cycles during elastic scaling.
- **Constraint**: `partial_gpu_mode` will be **disabled** or ignored. The pipeline must not rely on concurrent training/inference on the same GPU resources.
- **Requirement**: Explicitly **REMOVE** the `sleep_level=1` validation check found in `AgenticMultiLoraPipeline` when porting.
- **Validation**: Verify that reloading multiple adapters on every expand cycle is performant enough and that vLLM `sleep_level=2` restores both base model and adapter weights correctly.

#### 3. Megatron-Only Constraint
The `adapters_to_update` feature in `model_update` currently only supports the **Megatron-Core** strategy.
- **Constraint**: `SchedRLMultiLoraPipeline` will initially support only `megatron_train` backend for training.
- **Action**: Add runtime validation in `__init__` to fail fast if a different strategy is used.

#### 4. Per-Tag Scheduler Resizing
The existing `_shrink_workers` / `_expand_workers` methods in `AgenticPipeline` operate on the pipeline's main `train_rollout_scheduler` and `val_rollout_scheduler`.
- **Change**: `SchedRLMultiLoraPipeline` has `self.rollout_schedulers: dict[str, RolloutScheduler]`.
- **Implementation**: We must implement new helper methods `_shrink_all_schedulers` and `_expand_all_schedulers` that iterate over all per-tag schedulers and apply the resize operation.

#### 5. Progress Aggregation
- **Location**: The **SchedRL Scheduler** (specifically `RoundRobinGlobalScheduler`) is responsible for aggregating progress metrics.
- **Mechanism**: `GroupQueueManager` will report metrics including `adapter_id`. The Scheduler uses these aggregated metrics (per-adapter load) for global admission control decisions.

#### 6. Expand Warmup Strategy — ⏸ DEFERRED
- **Gap**: How does the coordinator know which adapters to warm up on expand before admission opens?
- **Strategy**: The Coordinator restores/maintains the "Active Model Spec", which includes the list of all currently active adapters.
- **Action**: On `expand_workers`, the Coordinator ensures that **all** adapters in the Active Model Spec are loaded and warmed up on the new workers **before** admission opens to the Scheduler.
- **Optimization**: Run a dummy forward pass (batch size 1) after loading adapters to initialize CUDA kernels/graphs before exposing to scheduler (prevents first-request timeout).
- **Status**: Not implemented. `TODO(item-6)` marker in `_expand_all_schedulers`. Deferred until first-request latency after expand is observed as a problem in practice (vLLM may handle CUDA kernel warmup internally via its own warmup path).

### Critical Safety & Validation Requirements

#### 7. Distributed State Consistency (ID Skew)
- **Problem**: vLLM generates adapter IDs based on load order. Non-deterministic loading leads to inconsistent IDs (e.g., Worker A has "Math"=1, Worker B has "Math"=2).
- **Requirement**: Implement `_verify_lora_model_update` (ported from `AgenticMultiLoraPipeline`) or an equivalent check.
- **Action**: After every `expand` or `load_adapters` operation, the coordinator must query all workers to verify that `adapter_name` -> `lora_int_id` mapping is consistent.
- **Constraint**: Fail fast if a mismatch is detected.

#### 8. Megatron-Core Training (DDP Hang)
- **Problem**: DistributedDataParallel (DDP) expects gradients for all bucketed parameters every step. If an adapter is not in the current batch, DDP hangs waiting for its gradients.
- **Requirement**: Ensure `adapters_to_update` logic correctly handles idle adapters.
- **Action**: Add a validation task in Phase 4 to verify multi-adapter training loop does not hang when batches contain single-adapter data.

#### 9. DDP Bucket Cache Consistency (State Pollution) — ✅ DONE
- **Problem**: Megatron DDP buckets cache gradients/params. Sequential adapter steps pollute these caches (Adapter A's step leaves stale cache used by Adapter B).
- **Requirement**: In `train_step_lora`, explicitly clear `model.bucket_groups` caches *between* adapter optimizer steps.
- **Action**: Modify the training loop to reset bucket caches after each adapter's `optimizer.step()`.
- **Status**: Implemented in `megatron_strategy.py:train_step_lora` — mirrors the `train_step` pattern (lines 1337-1341). With `use_distributed_optimizer=False` (current constraint) these caches are never set so it is a defensive no-op; becomes active if `use_distributed_optimizer=True` is enabled in future.

#### 10. DataLoader Iterator State & RNG (Data Duplication/Desync) — ⏸ DEFERRED
- **Problem**: Workers recreate DataLoaders from scratch on expand. `seed=42` restarts from batch 0. `sleep_level=2` does not persist iterator state or RNG state.
- **Requirement**: Track `consumed_samples` AND snapshot RNG state (CUDA/Python/Numpy) in `SchedRLMultiLoraPipeline` (persisted in `WorkerState`).
- **Action**: Pass `consumed_samples` to `_expand_all_schedulers` -> `expand_sampler` to fast-forward the iterator to the correct global step. Restore RNG state on worker initialization.
- **Status**: Not implemented. `TODO(item-10)` markers in `_shrink_all_schedulers` (save) and `_expand_all_schedulers` (restore). Deferred — only significant when shrink/expand cycles are frequent. For initial deployments with stable GPU allocation, data duplication risk is low.

#### 11. `lora_optimizer_mode` Configuration (Silent Failure)
- **Problem**: `shared` optimizer mode applies weight decay to idle adapter weights and shares a single Loss Scaler, coupling stability across adapters (Issue 11).
- **Requirement**: Enforce `lora_optimizer_mode="per_adapter"` to ensure optimizer state AND loss scaling isolation.
- **Action**: Add validation in `SchedRLMultiLoraPipeline.__init__` to fail if mode is not `per_adapter`.

#### 12. Metric Namespacing (Observability)
- **Problem**: Mixed metrics are useless (e.g., generic `loss` key overwritten by last adapter).
- **Requirement**: `train_step_lora` must namespace metrics (e.g., `{adapter}/loss`).
- **Action**: Ensure ported strategy enforces metric namespacing.

#### 13. Activation Checkpointing (AC) Compatibility & Correctness
- **Problem**: AC re-runs the forward pass during backward. If the adapter context changes (e.g., mixing adapters in one batch) or if gradients are accumulated across adapter swaps, the re-computation uses wrong weights. Also, Gradient Accumulation (GA) logic breaks if adapters with different GA schedules are mixed.
- **Requirement**: Enforce **Sequential Adapter Training**.
- **Action**: `train_step_lora` must loop through adapters sequentially, processing *all* microbatches and stepping the optimizer for Adapter A **before** touching Adapter B. This ensures the global adapter state remains consistent during the entire forward-backward-step cycle for each adapter.

#### 14. Resource Hygiene (Offload States Management & CPU RAM)
- **Problem**: After Megatron training completes, optimizer and model states may remain in GPU memory, preventing full release to SchedRL.
- **Requirement**: Proper offload of states before releasing GPUs.
- **Persistence**: Since Ray actors remain alive during `sleep_level=2`, states are moved to **CPU RAM** via `offload_states()`. Metadata (RNG, step counters) is persisted via `WorkerState.kv` (JSON).
- **Action**:
  - **Use existing `offload_states` mechanism** (already in ROLL_multi_lora):
    - Call `optimizer.offload_states(include=[OffloadStateType.optimizer_states])` after training completes.
    - This moves optimizer states (main_weights, optimizer states) to CPU via flat tensors.
  - **No explicit `del` required** - The existing codebase does not use `del optimizer`; it relies on `offload_states` to free GPU memory.
  - **Note**: This is handled by the Megatron strategy's existing offload/reload mechanism, not manual cleanup.
- **Validation**: Verify that `offload_states` is called after `train_step_lora` completes, before releasing GPUs to SchedRL.

#### 15. Per-Adapter RNG State Management (Determinism) — **DONE**

> **DDP analysis**: The original concern was that idle adapter B's backward hooks would never fire during adapter A's sequential pass, causing a hang. This only applies to `overlap_grad_reduce=True` where DDP uses per-bucket async backward hooks. With `overlap_grad_reduce=False` (the default, and the only valid setting for `per_adapter` mode since it requires `use_distributed_optimizer=False`), `finalize_model_grads()` does a **synchronous** all-reduce over all bucket groups including B's zero-gradient buckets — no hook waiting, no hang. A fail-fast check (`overlap_grad_reduce=True` raises `ValueError`) is added to enforce this.


- **Problem**: In sequential adapter training, Adapter A advances the global RNG during its training step. When Adapter B trains next in the same loop, it sees RNG state polluted by Adapter A. This causes non-deterministic dropout masks and breaks reproducibility.
- **Requirement**: Each adapter must have isolated RNG state that persists across training steps, checkpoint/resume, and shrink/expand cycles.
- **Action**:
  - **Extend ROLL's per-rank RNG mechanism to be per-adapter AND per-rank**:
    ```python
    # In MegatronStrategy (per-rank storage)
    self.adapter_rng_states: dict[str, dict] = {}  # {adapter_name: {cpu, cuda, python, numpy}}
    ```
  - **Initialize per-adapter RNG states** (during setup or checkpoint load):
    ```python
    for adapter_name in adapter_names:
        self.adapter_rng_states[adapter_name] = {
            "cpu": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(),
            "python": random.getstate(),
            "numpy": np.random.get_state(),
        }
    ```
  - **RNG state swapping in `train_step_lora`**:
    ```python
    for adapter_name in adapters_to_update:
        # RESTORE adapter's RNG state before training
        rng_state = self.adapter_rng_states[adapter_name]
        torch.set_rng_state(rng_state["cpu"])
        torch.cuda.set_rng_state(rng_state["cuda"])
        random.setstate(rng_state["python"])
        np.random.set_state(rng_state["numpy"])
        
        # Train this adapter
        self._train_single_adapter(adapter_name)
        
        # SAVE adapter's RNG state after training
        self.adapter_rng_states[adapter_name] = {
            "cpu": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(),
            "python": random.getstate(),
            "numpy": np.random.get_state(),
        }
    ```
  - **Checkpoint/Resume**: Save/load per-adapter RNG states using `WorkerState.save_rng_state()` / `load_rng_state()` for each adapter (same format as single-LoRA training).
  - **Shrink/Expand Persistence**: Store `adapter_rng_states` in `WorkerState.kv` before shrink, restore after expand.
- **Validation**: Add "Restart Determinism" test:
  1. Train Adapter A (step 1) → Train Adapter B (step 1) → Checkpoint
  2. Resume from checkpoint → Train Adapter A (step 2)
  3. Compare with continuous training (no checkpoint): weights must match exactly

#### 16. Norm Layer Stats Isolation (Requires Buffer Check) — **DONE**
- **Problem**: If LoRA adds Batch/Layer Norm layers, shared running stats (`running_mean`, `running_var`) could leak between adapters. These are **buffers** (not parameters) with `requires_grad=False`, so they are **NOT caught** by the existing parameter assertion.
- **Requirement**: Ensure BN/LN running stats buffers are isolated per adapter.
- **Action**:
  - **Extend the per-adapter validation** in `megatron_strategy.py` to check **named buffers** (not just parameters):
    ```python
    # After checking parameters, also check buffers
    for name, buffer in model.named_buffers():
        if "running_mean" in name or "running_var" in name or "num_batches_tracked" in name:
            if not any(marker in name for marker in markers.values()):
                raise ValueError(
                    f"BN/LN running stats not adapter-scoped: {name}. "
                    f"Wrap BN/LN layers in nn.ModuleDict keyed by adapter ID."
                )
    ```
  - This ensures running stats are isolated per adapter, preventing cross-adapter contamination.
- **Validation**: Add test with shared BN/LN layer to verify the buffer check raises `ValueError`.

#### 17. Partial Failure Handling (Fatal Error with Orchestrator Shutdown)
- **Problem**: During `expand_workers`, a single worker failing to load the base model or adapters in a tensor-parallel (TP) group leaves the group in a zombie state.
- **Cause**: TP requires all ranks to participate; one failed rank = corrupted group.
- **Note**: This is NOT related to adapter GC/eviction. It occurs during initial worker expansion when loading model weights into fresh GPU workers.
- **Action**:
  - **In `SchedRLMultiLoraPipeline._expand_all_schedulers()`**: Wrap `load_adapter` calls in try/except block.
  - **On failure**: Catch `WorkerLoadError`, log fatal error, and raise `RuntimeError(f"PARTIAL_TP_GROUP_FAILURE: {e}")`.
  - **In `SchedRLAdapter.resize_infer()`**: Catch exceptions from coordinator, check for `PARTIAL_TP_GROUP_FAILURE` in error message.
  - **Return `ActionResponse` with error reason**:
    ```python
    return ActionResponse(
        success=False,
        error=f"FATAL: {error_msg}"
    )
    ```
  - **In SchedRL Orchestrator**: On receiving `ActionResponse(success=False)`, trigger cluster shutdown:
    ```python
    self.shutdown(
        force=True,
        reason=response.error,  # "FATAL: PARTIAL_TP_GROUP_FAILURE: ..."
        source="pipeline_adapter"
    )
    ```
  - **Recovery**: SchedRL restarts pipeline from clean state.

#### 18. Asynchronous Learning Rate Decay (LR Desync Prevention) — **DONE**
- **Requirement**: Adapters that train sparsely (e.g., 10 samples) must not have their learning rate decayed based on a global step dominated by highly active adapters (e.g., 10,000 samples).
- **Mechanism**: Use `lora_optimizer_mode="per_adapter"`.
- **Correct Behavior**:
  - Each adapter has its own `adapter_schedulers[adapter_name]`.
  - `sch.step()` is ONLY called when that specific adapter is active in a training batch.
  - This ensures LR decay is mathematically bounded to the adapter's **local progress**, not the pipeline's wall-clock global step.
- **Checkpoint Fix (done)**: `save_checkpoint` previously saved `self.scheduler.state_dict()` (the shared ChainedOptimizer's scheduler, never stepped in per_adapter mode). Fixed to save `{"mode": "per_adapter", "schedulers": {k: v.state_dict() ...}}`. `load_checkpoint` now detects the mode key and restores each `adapter_schedulers[name]` from the saved state.

### Key Files to Port/Modify

#### 1. `roll/pipeline/base_pipeline.py`
**Current State**: ROLL_schedrl has basic `model_update()` without adapter subset support
**Change Required**: Add `model_update_lora_subset()` method (copy from ROLL_multi_lora)
```python
def model_update_lora_subset(self, global_step: int, *, adapters_to_update: set[str] | None = None) -> dict:
    """Adapter-subset model update helper for multi-LoRA pipelines."""
    metrics: dict = {}
    for model_update_group in self.model_update_groups:
        metrics.update(model_update_group.model_update(step=global_step, adapters_to_update=adapters_to_update))
    return metrics
```

#### 2. `roll/distributed/executor/model_update_group.py`
**Current State**: `model_update()` takes only `step` parameter
**Change Required**: Add `adapters_to_update` parameter and pass to workers
```python
def model_update(self, step=None, adapters_to_update: set[str] | None = None):
    if step % self.frequency != 0:
        return {}
    kwargs = {"model_update_name": self.model_update_name}
    if adapters_to_update is not None:
        kwargs["adapters_to_update"] = sorted(adapters_to_update)
    # ... rest of implementation
```

#### 3. Create `roll/schedrl_adapter/multi_lora_pipeline.py`
**New File**: `SchedRLMultiLoraPipeline` class.

**Key Components**:
- **Initialization**:
  - Inherit from `BasePipeline` (or `SchedRLConcurrentPipeline` if feasible, but `BasePipeline` is safer for custom run loop).
  - Initialize `self.rollout_schedulers` (per-tag) as in `AgenticMultiLoraPipeline`.
  - **Remove** `partial_gpu_mode` checks/requirements.
  - **Remove** `sleep_level=1` validation check.
  - **Add** validation for `megatron_train` strategy AND `lora_optimizer_mode="per_adapter"`.
- **Run Loop (Sequential SchedRL Cycle)**:
  1. **Expand**: Request GPUs (`_request_actor_infer_gpus`) -> `_expand_all_schedulers` (passing `consumed_samples` AND `adapter_rng_states`).
     - *Warmup*: Ensure `Active Model Spec` adapters are loaded on new workers.
     - *Validation*: **Call `_verify_lora_model_update` to ensure ID consistency.**
     - *Dry Run*: Execute dummy forward pass to initialize kernels.
     - *Error Handling*: Catch load failures -> Raise `RuntimeError(f"PARTIAL_TP_GROUP_FAILURE: {e}")` for orchestrator shutdown.
  2. **Rollout**: Iterate over per-tag schedulers to collect batches.
  3. **Train**:
     - Identify `dirty_adapters` from collected batches.
     - `model_update_lora_subset(..., adapters_to_update=dirty_adapters)`.
     - *Offload*: After training completes, call `offload_states(include=[OffloadStateType.optimizer_states])` to move optimizer states to CPU.
  4. **Shrink**: `_notify_ready_to_release_actor_infer` -> `_shrink_all_schedulers` -> Release GPUs.
     - *Persistence*: Save `consumed_samples` AND `adapter_rng_states` to `WorkerState` before shrink.
- **Helper Methods**:
  - `_shrink_all_schedulers(dp_ranks_to_remove)`: Iterate `self.rollout_schedulers.values()` and call `shrink_sampler`.
  - `_expand_all_schedulers(dp_ranks_to_add)`: Iterate `self.rollout_schedulers.values()` and call `expand_sampler`.

#### 4. `roll/utils/lora_routing.py`
**Action**: Copy from ROLL_multi_lora to ROLL_schedrl (if not present).

#### 5. `roll/distributed/scheduler/rollout_scheduler.py` (and `GroupQueueManager`)
**Change Required**: Enable passing `adapter_id` through to `GroupQueueManager` for progress reporting.
- Update `GroupQueueManager.__init__` to extract `adapter_id` from `env_manager_config.tags[0]`.
- Update `GroupQueueManager._maybe_emit_progress` to include `adapter_id` in `metrics`.

### Implementation Phases

#### Phase 1: Base Pipeline & Core Updates
**Files**:
- `roll/pipeline/base_pipeline.py`: Add `model_update_lora_subset()`
- `roll/distributed/executor/model_update_group.py`: Add `adapters_to_update` parameter
- `roll/distributed/scheduler/rollout_scheduler.py`: Add `adapter_id` extraction
- `roll/third_party/megatron/model_update.py`: Port adapter-aware logic from `ROLL_multi_lora`.
- `roll/distributed/strategy/megatron_strategy.py`:
  - Port `train_step_lora` method.
  - **CRITICAL**: Enforce sequential adapter processing (one at a time) in `train_step_lora` to satisfy AC and GA requirements.
  - **CRITICAL**: Clear bucket caches between adapter steps in `train_step_lora`.
  - **CRITICAL**: Enforce metric namespacing in `train_step_lora`.
  - Implement multi-component caching (base + adapters).
- `roll/schedrl_adapter/model_update_service.py`: Update `sync_selected_workers` to support `adapters_to_sync`.

**Success Criteria**:
- [x] `make test` passes in `external/ROLL_schedrl`
- [x] `model_update_lora_subset` delegates correctly.
- [x] `train_step_lora` processes adapters sequentially (forward/backward interleaved; per-adapter optimizer steps sequential).
- [x] Metrics are correctly namespaced (e.g. `actor_train/adapter_A/grad_norm`).

**Bugs discovered and fixed during port (not in original plan)**:
- `MegatronInferStrategy.initialize()` was missing `self.is_lora` — pure infer workers would raise `AttributeError` on first inference with LoRA enabled.
- `inner_forward_step` was missing the `set_adapter(routing.lora_name)` call — all microbatches used whatever adapter was last set during optimizer creation, silently using wrong weights. Bug was invisible in the equivalence test because all adapters start with identical weights via `copy_lora_params`.

**Additional change (not in original plan)**:
- `train_step_lora` added to base `ActorWorker` in `roll/pipeline/base_worker.py` (not just `SFTWorker`) so agentic actor workers expose the multi-LoRA training RPC.

#### Phase 2: Static Memory Hard-Capping (Issue 2 Fix)
**Files**:
- `roll/schedrl_adapter/multi_lora_pipeline.py`
- **Adapter Validation**:
  - Since adapters are fixed in config, we apply static validation and remove complex GC.
  - Implement `max_resident_adapters` validation in `__init__`.
  - Compare `len(active_model_spec.adapter_names)` to `max_resident_adapters` and fail fast with clear ValueError if the limit is exceeded.
  - *No runtime GC, no LRU eviction required.*

**Success Criteria**:
- [x] VRAM limits are enforced during `initialize_pipeline()` before any GPUs are allocated.
- [x] A pipeline given > N adapters fails to start and raises `RuntimeError`.

#### Phase 3: SchedRLMultiLoraPipeline Implementation
**Files**:
- `roll/schedrl_adapter/multi_lora_pipeline.py`: New file.

**Status**: Complete. Inherits from `SchedRLConcurrentPipeline` (not `BasePipeline` as originally considered) to reuse all SchedRL helper methods (`_request_actor_infer_gpus`, `_notify_ready_to_release_actor_infer`, etc.) without duplication. `initialize_pipeline()` calls `super()` first (which owns `_init_lock`), then adds per-tag scheduler setup guarded by `_rollout_schedulers_initialized` flag.

**Key Implementation Details**:
```python
class SchedRLMultiLoraPipeline(SchedRLConcurrentPipeline):  # actual; plan originally said BasePipeline
    def __init__(self, ...):
        # ... setup per-tag schedulers ...
        if strategy_name != "megatron_train":
             raise RuntimeError("SchedRLMultiLoraPipeline currently requires megatron_train strategy.")
        if lora_optimizer_mode != "per_adapter":
             raise RuntimeError("SchedRLMultiLoraPipeline requires lora_optimizer_mode='per_adapter'.")
        # NOTE: Explicitly REMOVED sleep_level=1 check. We require sleep_level=2 for SchedRL.

    def run(self):
        # ... SchedRL-style sequential loop ...
        # 1. Expand (load base + all Active Model Spec adapters)
        #    - Pass self.consumed_samples to expand_sampler for fast-forwarding
        # 2. Verify IDs: self._verify_lora_model_update(adapters=active_adapters, where="expand")
        # 3. Rollout (mixed batches)
        # 4. Train (update dirty adapters)
        #    - After training: offload_states(include=[OffloadStateType.optimizer_states])
        # 5. Shrink (offload everything)
        #    - Save current iterator position to self.consumed_samples
        #    - Save adapter_rng_states to WorkerState

    def resize_infer(self, dp_ranks_to_remove, dp_ranks_to_add):
        # ... SchedRL integration ...
        if dp_ranks_to_remove:
            self._shrink_all_schedulers(dp_ranks_to_remove)
        else:
            try:
                self._expand_all_schedulers(dp_ranks_to_add)
            except WorkerLoadError as e:
                logger.fatal(f"[schedrl][{self._pipeline_id}] Partial TP group failure: {e}")
                raise RuntimeError(f"PARTIAL_TP_GROUP_FAILURE: {e}")

    def _verify_lora_model_update(self, *, adapters: set[str] | None, where: str) -> None:
         # Query all workers for their lora_id mappings and ensure they match.
         # Raise RuntimeError if mismatch.
         pass
```

#### Phase 4: Validation & Registration
- [x] Register `SchedRLMultiLoraPipeline` in `adapter.py` — `create_coordinator()` auto-selects based on `pipeline_config.actor_train.model_args.adapters` being non-empty.
- **Validation Task 1**: Verify `sleep_level=2` compatibility & Memory Hygiene (A05).
  - Test that `SchedRLMultiLoraPipeline` can:
    1. Expand from zero -> load base + 2 adapters.
    2. Serve requests for both adapters.
    3. Shrink to zero (full release).
    4. Expand again -> reload base + 2 adapters (verify weights are correct).
  - **NEW**: Verify zero tensor references remain on GPU after shrink (using `objgraph`).
- **Validation Task 2**: Verify DDP correctness.
  - Test training with batch containing only Adapter A samples while Adapter B exists.
  - Ensure no DDP hang occurs (verify gradients propagate correctly).
- **Validation Task 3**: Verify Per-Adapter RNG Determinism (B04).
  - Train Adapter A (step 1) → Train Adapter B (step 1) → Checkpoint → Resume → Train Adapter A (step 2).
  - Compare with continuous training (no checkpoint): Adapter A's step 2 must produce identical weights.
  - Verify per-adapter RNG isolation: Adapter B's training does not pollute Adapter A's RNG stream.
- **Validation Task 4**: Verify Activation Checkpointing (if enabled).
  - Ensure gradients match a run without AC. Ensure no stale weights are used during re-computation.
- **Validation Task 5**: Verify Norm Layer Isolation (B05).
  - Verify that BN/LN **buffers** (`running_mean`, `running_var`, `num_batches_tracked`) trigger the new buffer check if not properly scoped with adapter prefix.
  - Confirm the extended validation catches shared BN/LN running stats.
  - Test with shared BN layer to verify `ValueError` is raised during per-adapter optimizer initialization.
- **Validation Task 6**: Verify Partial Failure Handling (A02).
  - Inject failure on Rank 1 during `load_adapter`.
  - Verify `ActionResponse(success=False, error="FATAL: PARTIAL_TP_GROUP_FAILURE: ...")` is returned.
  - Verify SchedRL orchestrator triggers `shutdown(force=True, reason=..., source="pipeline_adapter")`.
  - Verify Ray cluster is fully shut down and no zombie state remains.

#### Phase 5: Per-Adapter Selective Sync — **DONE**

**Bug fixed**: `multi_lora_pipeline.py` Phase 14 called `worker.promote_active_checkpoint.remote(checkpoint_version, global_step)` after `train_step_lora`, but `train_step_lora` never called `_build_latest_bucket_cache`. This caused `RuntimeError: promote_active_checkpoint missing cache_key` on the first training step.

**Root cause**: The full fine-tune path has `_build_latest_bucket_cache` called inside `train_step` (megatron_strategy.py line ~1372) only when `SCHEDRL_CONTROL_PLANE=schedrl`. The multi-LoRA path skips this because only adapter weights change per step — building the full model cache on every step wastes memory and bandwidth.

**Solution**: Per-adapter versioned cache (build → promote → notify → offload), parallel to the full fine-tune pattern but scoped to individual adapters.

**Architecture**:
- `_adapter_cache_map[adapter][key]` stores CPU-resident serialized bucket bytes, same format as `_cache_map`
- GC per adapter: only `_latest_adapter_cached[a]` and `_active_adapter_cached[a]` retained (2 versions max)
- `_op_lock` in `RequestScheduler` serializes `notify_adapter_updated` with `shrink_workers`/`expand_workers` to prevent races with routing changes
- Lock order: `_op_lock → routing_lock` (never reversed)
- All per-tag `RolloutSchedulers` share the **same** underlying `RequestScheduler` actor; only the first scheduler's `notify_adapter_updated` is called per step

**`selective_sync_active_cache` dispatch**:
- `adapters_to_sync` given → use `_adapter_cache_map` for named adapters (post-training sync to active workers)
- `adapters_to_sync=None` + `is_lora=True` → sync all `_active_adapter_cached` entries (expand path: newly woken workers need all adapters)
- `adapters_to_sync=None` + `is_lora=False` → existing full fine-tune path unchanged (`_cache_map`)

**Files changed**:

| File | Change |
|---|---|
| `roll/distributed/strategy/megatron_strategy.py` | Added `_adapter_cache_map`, `_latest_adapter_cached`, `_active_adapter_cached` fields; `adapter_name` param on `_build_latest_bucket_cache`; new `promote_active_adapter_checkpoint`; refactored `selective_sync_active_cache` cache-selection block |
| `roll/distributed/executor/worker.py` | `adapter_name` param on `build_latest_bucket_cache`; new `promote_active_adapter_checkpoint` thin wrapper |
| `roll/distributed/scheduler/generate_scheduler.py` | `_op_lock` field; `shrink_workers` and `expand_workers` bodies wrapped in `async with self._op_lock`; new `notify_adapter_updated` method |
| `roll/distributed/scheduler/rollout_scheduler.py` | New `notify_adapter_updated` thin delegate to `RequestScheduler` |
| `roll/schedrl_adapter/multi_lora_pipeline.py` | `initialize_pipeline`: build + promote initial per-adapter caches for all adapters before first shrink; Phase 14 `run()`: replaced broken `promote_active_checkpoint` block with build→promote→notify→offload sequence |

**Data flow**:
```
initialize_pipeline:
  for each adapter:
    build_latest_bucket_cache(0, 0, adapter_name)     # gathers adapter weights → CPU bytes
    promote_active_adapter_checkpoint(adapter, 0, 0)  # marks as active, GC old versions

run() Phase 14 (each training step):
  train_step_lora(batch)                               # trains 1+ adapters on GPU
  for each trained adapter:
    build_latest_bucket_cache(cv, gs, adapter_name)   # update CPU cache (before offload)
    promote_active_adapter_checkpoint(adapter, cv, gs) # GC old, mark new active
  first_scheduler.notify_adapter_updated(trained_adapters)
    → RequestScheduler.notify_adapter_updated()
      → reads active_dp_ranks under routing_lock (snapshot)
      → model_update_service.sync_selected_workers(active_ranks, adapters_to_sync=[...])
        → selective_sync_active_cache(adapters_to_sync=[...])
  actor_train.offload_states()                         # AFTER cache build

resize_infer(expand):
  expand_workers()                                     # wakes idle workers
    → sync_selected_workers(load_ranks)                # adapters_to_sync=None
      → selective_sync_active_cache(is_lora=True)      # sends ALL active adapter caches
```

**Success Criteria**:
- [x] No `RuntimeError: promote_active_checkpoint missing cache_key` on multi-LoRA training steps
- [x] `_adapter_cache_map` populated at init and updated after each training step
- [x] `selective_sync_active_cache` sends only trained-adapter buckets after training; all adapter buckets on expand
- [x] Non-LoRA pipeline: `selective_sync_active_cache(adapters_to_sync=None)` uses `_cache_map` path unchanged
- [x] `notify_adapter_updated` strictly serialized with `shrink_workers`/`expand_workers` via `_op_lock`

---

### Phase 0 Unit Tests: `per_adapter` Single-LoRA Step Correctness

**File**: `external/ROLL_schedrl/tests/integration/test_per_adapter_single_lora_step_equivalence.py`

#### Rationale — Cross-Mode Equivalence

With `lora_optimizer_mode="per_adapter"` and a **single adapter** per `train_step_lora` call, the per-adapter code path degenerates to the same sequence as the upstream single-LoRA `train_step`:

```
forward/backward over 1 micro-batch → optimizer.step() for that adapter → scheduler.step() → zero_grad
```

By running both modes on identical weights and identical tokens we get a cheap, ground-truth-free correctness signal across DP × TP configurations, without needing a separate oracle or numerical baseline.

#### Test Matrix

| TC | `dp` | `tp` | Adapters | GPUs needed | Description |
|----|------|------|----------|-------------|-------------|
| 1  | 1    | 1    | a, b     | 2           | Baseline: single GPU per side |
| 2  | 2    | 1    | a, b, c  | 4           | Data parallelism |
| 3  | 1    | 2    | a, b, c  | 4           | Tensor parallelism |
| 4  | 2    | 2    | a, b, c  | 8           | Combined DP + TP |

#### Setup (per test)

1. **per_adapter cluster** (ROLL_schedrl `SFTWorker`, `lora_optimizer_mode="per_adapter"`) on GPUs `[0, dp*tp)`.
2. **Reference cluster** (ROLL_schedrl `SFTWorker`, standard `megatron_train`, no `lora_optimizer_mode`) on GPUs `[dp*tp, 2*dp*tp)`.
3. All adapters in the per_adapter cluster are seeded with identical initial weights via `copy_lora_params`.
4. For each adapter in sequence:
   - Weights are copied from per_adapter cluster → reference cluster via `get/set_lora_tensors`.
   - One `train_step_lora(mb)` call (single adapter only) is run on the per_adapter cluster.
   - One `train_step(mb)` call (upstream single-LoRA) is run on the reference cluster.
   - Final LoRA weights are compared with `torch.testing.assert_close(rtol=1e-5, atol=1e-6)`.
   - The per_adapter adapter weight is reset to its pre-step init so each adapter's test is independent.

#### Skip Conditions
- `torch.cuda.device_count() < 2*dp*tp` — skipped cleanly on smaller machines.
- ModelScope cache missing for `Qwen/Qwen2.5-0.5B-Instruct`.

#### How to Run

```bash
# All 4 test cases
cd external/ROLL_schedrl
pytest tests/integration/test_per_adapter_single_lora_step_equivalence.py -v

# Individual test cases
pytest tests/integration/test_per_adapter_single_lora_step_equivalence.py -k "tc1" -v
pytest tests/integration/test_per_adapter_single_lora_step_equivalence.py -k "tc2" -v
pytest tests/integration/test_per_adapter_single_lora_step_equivalence.py -k "tc3" -v
pytest tests/integration/test_per_adapter_single_lora_step_equivalence.py -k "tc4" -v
```

#### Dependencies (must be ported in Phase 0 before tests pass)
- `MegatronTrainStrategy.train_step_lora` with `lora_optimizer_mode="per_adapter"` (from ROLL_multi_lora).
- `Worker.{get_lora_tensors, set_lora_tensors, copy_lora_params, train_step_lora}` (from ROLL_multi_lora).

---

### Design Decisions & Clarifications

1.  **Sleep Level vs Partial GPU**:
    - We strictly use `sleep_level=2`.
    - We **drop** `partial_gpu_mode` support for this SchedRL adapter. The execution is sequential.

2.  **Known Production Risks (Backlog)**:
    - **Cold Start Latency**: Loading all adapters sequentially on expand is a known bottleneck. Future work: Implement "Lazy Loading".
    - **VRAM Fragmentation**: Repeated load/unload may cause fragmentation. Future work: LRU eviction policy handles this via `remove_lora`.
    - **Partial Failure**: A single worker failing to load invalidates the group. Handled by SchedRL's standard fault tolerance (worker restart).

---

## Backlog: `use_distributed_optimizer=True` Support for `per_adapter` Mode

### Why not in scope

In `per_adapter` mode, only **one adapter's optimizer state is loaded onto GPU at a time** — the training loop loads adapter A's optimizer, steps it, then offloads back to CPU before moving to adapter B. Each adapter's optimizer state (fp32 main weights + adam m + adam v) is typically small:

- Rank-8 LoRA on a 7B model: ~88MB per adapter
- Even 100 adapters: ~8.8GB total, but only ~88MB on GPU at any given moment

`DistributedOptimizer` saves memory by **sharding across DP ranks**. For the GPU-resident portion, the saving is trivial (88MB / DP size). For the CPU-offloaded portion, it would save CPU RAM — but CPU RAM is typically large enough that 8.8GB across 100 adapters is not a bottleneck. The engineering cost outweighs the benefit for the common case.

The hard block (`raise ValueError`) and the forced `use_distributed_optimizer=False` in `OptimizerConfig` construction are both intentional for the current prototype.

**DDP gradient buffer memory** is a secondary problem with the current single-DDP approach. Megatron's `_ParamAndGradBuffer.grad_data` is a flat GPU tensor allocated at DDP init time for ALL `requires_grad` params (all N adapters). With `use_distributed_optimizer=False`, `param_data` is not allocated, but `grad_data` always is:

| Adapters | Gradient buffer (bf16) | On GPU during 1-adapter step |
|---|---|---|
| 10 | ~146 MB | ~146 MB (99% zeros) |
| 100 | ~1.46 GB | ~1.46 GB (99% zeros) |
| 500 | ~7.3 GB | ~7.3 GB (99% zeros) |

`finish_grad_sync()` all-reduces the entire flat buffer including zero-gradient slots for idle adapters. The N-DDP-wrappers approach would also enable explicit CPU offload of inactive gradient buffers between steps, recovering most of this memory — another motivation beyond `DistributedOptimizer`. (Not automatic: requires an explicit `_offload_adapter_grad_buffer(other)` call in the training loop before activating each adapter.)

**Checkpoint complexity**: N `DistributedOptimizer` instances each require `dist_checkpointing.save()` with ZeRO sharding, coordination across all DP ranks, and namespaced state dict keys per adapter. Extending `save_checkpoint`/`load_checkpoint` to loop over N adapters with distributed coordination adds ~50-80 lines of non-trivial code on top of the N-DDP init changes.

**Trigger threshold**: Revisit when `adapter_count > 200` (gradient buffer ~2.9 GB) OR when `overlap_grad_reduce` support is explicitly required.

### Why it would work architecturally (N separate DDP wrappers)

The clean design is: **N separate DDP wrapper instances + N DistributedOptimizers, one per adapter. Only the base model is shared.**

This works because:

1. **DDP filters by `requires_grad`** — creating DDP_A with only adapter A's params active (`requires_grad=True`) and DDP_B with only adapter B's params active produces disjoint flat buffers and disjoint gradient hooks. No conflict.

2. **`unmap_weight_tensor` is safe** — it runs `self.module.apply(...)` on the entire model, setting `weight_tensor = None` on any TE submodule. This is idempotent: running it N times (once per DDP init) has the same effect as once. No param data is touched, only a stale reference is cleared. Verified by code inspection of `distributed_data_parallel.py:394-398`. Note: this is internal Megatron-Core code; verify idempotency after Megatron-Core version upgrades.

3. **Backward hooks are disjoint** — DDP registers hooks only on `requires_grad=True` params at its init time (`distributed_data_parallel.py:404-411`). DDP_A's hooks are on adapter A's params; DDP_B's hooks are on adapter B's params. No double-accumulation.

4. **`model_config.no_sync_func`** is the only shared-state write — each DDP init overwrites it on the shared model config. This is handled by swapping it in the training loop alongside `self.models_wrapped`.

5. **`overlap_grad_reduce=True` becomes unblocked** — the current single-DDP hang occurs because DDP has hooks for ALL adapters' params; during adapter A's backward, only adapter A's hooks fire but `finish_grad_sync` waits for ALL buckets. With N separate DDP wrappers, DDP_A has hooks only for adapter A's params — all buckets complete normally during adapter A's backward. The `ValueError` guard can be removed for the N-DDP path. (Note: a review suggested N DDP wrappers don't fix this; that reasoning was incorrect — DDP_B's hooks are irrelevant during adapter A's pass because DDP_B is not in use.)

### Implementation sketch (~150-200 lines in `megatron_strategy.py` + checkpoint changes)

**Init** — create N DDP wrappers + N DistributedOptimizers:
```python
# per_adapter + use_distributed_optimizer=True path
self.adapter_models_wrapped = {}
for adapter_name in adapter_names:
    _apply_trainability_mask_for_adapter(adapter_name)  # only adapter_name requires_grad
    adapter_ddp = [
        DistributedDataParallel(
            config=m.config, ddp_config=ddp_config_with_dist_opt, module=m,
            disable_bucketing=(i > 0),
        )
        for i, m in enumerate(self.models_unwrapped)
    ]
    self.adapter_models_wrapped[adapter_name] = adapter_ddp
    self.adapter_optimizers[adapter_name] = get_megatron_optimizer(optimizer_config, adapter_ddp)
    self.adapter_schedulers[adapter_name] = get_megatron_lr_scheduler(..., optimizer=self.adapter_optimizers[adapter_name])
# After all DDP wrappers created — restore all params to requires_grad=False (base model frozen)
_restore_trainability()
```

**Training loop** — swap active DDP wrapper per adapter:
```python
for adapter_name in adapters_in_order:
    # Swap active DDP — must update both self.models_wrapped AND self.model.models
    # (forward_backward_func uses self.model.get_models() which returns self.model.models)
    self.models_wrapped = self.adapter_models_wrapped[adapter_name]
    self.model.models = self.models_wrapped
    self.model.config.no_sync_func = self.models_wrapped[0].no_sync

    # Restore per-adapter RNG state (preserved from current per_adapter implementation)
    rng = self.adapter_rng_states[adapter_name]
    torch.set_rng_state(rng["cpu"]); torch.cuda.set_rng_state(rng["cuda"])

    self.zero_grad()
    for mb in adapter_to_mbs[adapter_name]:
        self.forward_backward_only(mb, loss_func)  # finish_grad_sync → reduce-scatter adapter_name's buffer only
    self.adapter_optimizers[adapter_name].step()   # all-gather adapter_name's updated params
    self.adapter_schedulers[adapter_name].step()
    # Offload this adapter's optimizer + grad_data before next adapter (per-adapter, not bulk)
    self.adapter_optimizers[adapter_name].offload_states(...)
```

**Remove** the `use_distributed_optimizer=False` force in `OptimizerConfig` and the `ValueError` check — replace with conditional branching to the N-wrapper path.

**FP16 loss scaler constraint**: Each `DistributedOptimizer` has its own `grad_scaler` with its own `found_inf_flag`. Do NOT wrap N optimizers in a `ChainedOptimizer` — `ChainedOptimizer.prepare_grads()` aggregates all `found_inf_flag`s, so adapter A's overflow would force all adapters to skip their step. The sketch already avoids this by calling `adapter_optimizers[adapter_name].step()` independently per adapter. This must be preserved.

**Checkpoint namespacing**: `ChainedOptimizer.sharded_state_dict()` uses `chained_{idx}.` prefix. With N separate `DistributedOptimizer` instances, checkpoint save/load must namespace by adapter name (e.g., `adapter_math.`, `adapter_code.`) rather than optimizer index — otherwise state dicts from different runs collide if adapters are added/removed. This is part of the ~50-80 lines of checkpoint code estimate.

**Training loop invariant / hook isolation**: Before training adapter X, DDP_X must be loaded (hooks active on adapter X's params) and DDP_Y (Y≠X) must be offloaded. Root cause: `offload_states_patch.py:chained_optimizers_reload_states` registers gradient hooks on ALL `requires_grad=True` params at reload time (all adapters have `requires_grad=True` after `_restore_trainability()`), then sets `skip_grad_hook_register=True` for the last optimizer — meaning the last optimizer gets NO hooks at all. Benign in the current code (single optimizer, hook redundancy is harmless), but breaks with N DDPs: optimizer_A gets hooks on adapter B's params (dropped silently since B's params aren't in A's `param_to_bucket_group`); last optimizer gets no hooks. **Concrete fix**: modify `move_grad_data_to_device` to filter hook registration by `managed_param_ids = {id(p) for p in optimizer.buffers[0].params}` — only register hooks for params this optimizer manages. This requires changes to `offload_states_patch.py`, which is the primary additional complexity beyond the ~150-200 lines in `megatron_strategy.py`.

**Non-issues at implementation time**: `finalize_model_grads_func` — safe, operates on `self.models_wrapped[0].bucket_groups` at call time (always the active DDP). ZeRO-1 all-gather — safe, all DP ranks train the same adapter simultaneously. MoE `expert_parallel_bucket_groups` — handled by `zero_grad_buffer()` internally. VPP model chunks — sketch already handles `M` chunks per adapter via `enumerate(self.models_unwrapped)`. `_allreduce_non_tensor_model_parallel_grads` — iterates `requires_grad=True` params but LoRA params don't have `sequence_parallel=True` and base params are frozen — non-issue for typical LoRA training. `overlap_param_gather` — already blocked at strategy level; keep blocked for N-DDP initially.

### Memory benefit

| Adapters | Optimizer state (CPU) | With ZeRO-1 DP=8 | Savings/rank |
|---|---|---|---|
| 10 | 880 MB | 110 MB | 770 MB |
| 100 | 8.8 GB | 1.1 GB | 7.7 GB |
| 500 | 44 GB | 5.5 GB | 38.5 GB |

Meaningful only at 100+ adapters where CPU RAM pressure is genuine. Below that, the current `use_distributed_optimizer=False` path is sufficient.

---

## References

- Shared protocol: `design_doc/multi-pipeline-adaptation-plan.md`
- Dual-mode scheduler plan: `thoughts/shared/plans/2026-01-28-schedrl-dual-mode-final-plan.md`
- ROLL adaptation plan: `thoughts/shared/plans/2026-01-28-roll-schedrl-adaptation.md`
- SkyRL-train adaptation plan: `thoughts/shared/plans/2026-01-28-skyrl-train-adaptation-plan.md`
- NeMo-RL adaptation plan: `thoughts/shared/plans/2026-01-28-nemo-rl-schedrl-adaptation.md` (deferred; archived)
- ROLL_multi_lora: `external/ROLL_multi_lora/roll/pipeline/agentic/agentic_multi_lora_pipeline.py`
- ROLL_schedrl: `external/ROLL_schedrl/roll/schedrl_adapter/concurrent_pipeline.py`
