# SchedRL Multi‑LoRA Adapter Extension (ROLL SchedRL Integration) Implementation Plan

**Date**: 2026-02-02 (Updated: 2026-02-18)

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

#### 6. Expand Warmup Strategy
- **Gap**: How does the coordinator know which adapters to warm up on expand before admission opens?
- **Strategy**: The Coordinator restores/maintains the "Active Model Spec", which includes the list of all currently active adapters.
- **Action**: On `expand_workers`, the Coordinator ensures that **all** adapters in the Active Model Spec are loaded and warmed up on the new workers **before** admission opens to the Scheduler.
- **Optimization**: Run a dummy forward pass (batch size 1) after loading adapters to initialize CUDA kernels/graphs before exposing to scheduler (prevents first-request timeout).

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

#### 9. DDP Bucket Cache Consistency (State Pollution)
- **Problem**: Megatron DDP buckets cache gradients/params. Sequential adapter steps pollute these caches (Adapter A's step leaves stale cache used by Adapter B).
- **Requirement**: In `train_step_lora`, explicitly clear `model.bucket_groups` caches *between* adapter optimizer steps.
- **Action**: Modify the training loop to reset bucket caches after each adapter's `optimizer.step()`.

#### 10. DataLoader Iterator State & RNG (Data Duplication/Desync)
- **Problem**: Workers recreate DataLoaders from scratch on expand. `seed=42` restarts from batch 0. `sleep_level=2` does not persist iterator state or RNG state.
- **Requirement**: Track `consumed_samples` AND snapshot RNG state (CUDA/Python/Numpy) in `SchedRLMultiLoraPipeline` (persisted in `WorkerState`).
- **Action**: Pass `consumed_samples` to `_expand_all_schedulers` -> `expand_sampler` to fast-forward the iterator to the correct global step. Restore RNG state on worker initialization.

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

#### 15. Per-Adapter RNG State Management (Determinism)
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

#### 16. Norm Layer Stats Isolation (Requires Buffer Check)
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

#### 18. Asynchronous Learning Rate Decay (LR Desync Prevention)
- **Requirement**: Adapters that train sparsely (e.g., 10 samples) must not have their learning rate decayed based on a global step dominated by highly active adapters (e.g., 10,000 samples).
- **Mechanism**: Use `lora_optimizer_mode="per_adapter"`. 
- **Correct Behavior**:
  - Each adapter has its own `adapter_schedulers[adapter_name]`.
  - `sch.step()` is ONLY called when that specific adapter is active in a training batch.
  - This ensures LR decay is mathematically bounded to the adapter's **local progress**, not the pipeline's wall-clock global step.

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
- [ ] `make test` passes in `external/ROLL_schedrl`
- [ ] `model_update_lora_subset` delegates correctly.
- [ ] `train_step_lora` processes adapters sequentially.
- [ ] Metrics are correctly namespaced (e.g. `adapter_A/loss`).

#### Phase 2: Static Memory Hard-Capping (Issue 2 Fix)
**Files**:
- `roll/schedrl_adapter/multi_lora_pipeline.py`
- **Adapter Validation**:
  - Since adapters are fixed in config, we apply static validation and remove complex GC.
  - Implement `max_resident_adapters` validation in `__init__`.
  - Compare `len(active_model_spec.adapter_names)` to `max_resident_adapters` and fail fast with clear ValueError if the limit is exceeded.
  - *No runtime GC, no LRU eviction required.*

**Success Criteria**:
- [ ] VRAM limits are enforced during `__init__` before any GPUs are allocated.
- [ ] A pipeline given > N adapters fails to start and shuts down.

#### Phase 3: SchedRLMultiLoraPipeline Implementation
**Files**:
- `roll/schedrl_adapter/multi_lora_pipeline.py`: New file.

**Key Implementation Details**:
```python
class SchedRLMultiLoraPipeline(BasePipeline):
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
- Register `SchedRLMultiLoraPipeline` in `adapter.py`.
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

### Phase 1 Unit Tests: `per_adapter` Single-LoRA Step Correctness

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

#### Dependencies (must be ported in Phase 1 before tests pass)
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

## References

- Shared protocol: `design_doc/multi-pipeline-adaptation-plan.md`
- Dual-mode scheduler plan: `thoughts/shared/plans/2026-01-28-schedrl-dual-mode-final-plan.md`
- ROLL adaptation plan: `thoughts/shared/plans/2026-01-28-roll-schedrl-adaptation.md`
- SkyRL-train adaptation plan: `thoughts/shared/plans/2026-01-28-skyrl-train-adaptation-plan.md`
- NeMo-RL adaptation plan: `thoughts/shared/plans/2026-01-28-nemo-rl-schedrl-adaptation.md` (deferred; archived)
- ROLL_multi_lora: `external/ROLL_multi_lora/roll/pipeline/agentic/agentic_multi_lora_pipeline.py`
- ROLL_schedrl: `external/ROLL_schedrl/roll/schedrl_adapter/concurrent_pipeline.py`
