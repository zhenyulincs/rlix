---
date: 2026-01-28T05:51:01Z
researcher: tao
git_commit: 9f8c65b054b9f68cba0864ddd09cd0f8240f0c9d
branch: main
repository: SchedRL
topic: "SchedRL Framework Integration: Existing Mechanisms for Scheduler Implementation"
tags: [research, codebase, schedrl, weight-sync, rollout, vllm, sglang, megatron, nemo-rl, roll, miles, skyrl]
status: complete
last_updated: 2026-01-28
last_updated_by: tao
---

# Research: SchedRL Framework Integration - Existing Mechanisms for Scheduler Implementation

**Date**: 2026-01-28T05:51:01Z
**Researcher**: tao
**Git Commit**: 9f8c65b054b9f68cba0864ddd09cd0f8240f0c9d
**Branch**: main
**Repository**: SchedRL

## Research Question

Document existing components and mechanisms across ROLL, NeMo-RL, Miles, and SkyRL-train frameworks that can be reused for implementing the SchedRL scheduler, focusing on:
1. Model weight sync between train and inference engine
2. Async multi-turn agentic rollout loop per trajectory
3. Pipeline coordinator control of train and inference workers/GPUs
4. Offloading of model weights, KV cache, and optimizer
5. Megatron as trainer and vLLM v1 engine as inference backend (SGLang as fallback)
6. Request dispatcher load balancing among rollout DP workers

## Summary

This research documents the existing infrastructure across four RL training frameworks for implementing the SchedRL multi-pipeline GPU sharing scheduler. Each framework has mature components that can be adapted:

- **ROLL**: Most complete for agentic multi-turn rollouts with abort/retry at turn boundaries; uses bucket-based NCCL broadcast for weight sync
- **NeMo-RL**: Best async GRPO implementation with version-tagged trajectories; supports both CUDA-IPC/ZMQ and NCCL collective weight sync
- **Miles**: Explicit onload/offload patterns with staged memory management (weights, KV cache, CUDA graphs separately); global data source buffer enables retry
- **SkyRL-train**: Modular weight sync strategies (broadcast vs CUDA IPC); pause/resume with automatic retry for fully-async training

---

## Detailed Findings

### 1. Model Weight Sync Between Training and Inference

#### 1.1 ROLL Framework

**Entry Points:**
- `third_party/ROLL/roll/distributed/executor/model_update_group.py:139` - `ModelUpdateGroup.model_update()`
- `third_party/ROLL/roll/distributed/strategy/megatron_strategy.py:1141` - `MegatronTrainStrategy.model_update()`

**Mechanism:**
- Uses **bucket-based NCCL broadcast**; bucket target size is implementation-specific. For example, NeMo-RL computes the target from `NRL_REFIT_BUFFER_MEMORY_RATIO` (default 0.02 of GPU memory) and caps the bucket at 5GB — it is not a fixed 256MB default.
- `ModelUpdateGroup` maintains two communication plans:
  - `broadcast_comm_pan`: Maps PP rank → src rank → target devices
  - `p2p_comm_plan`: Same-GPU fallback communication
- Weight gathering: `all_gather_weights_as_hf_bucket()` iterates model params, gathers across TP ranks, converts to HuggingFace format
- Creates **per-subset NCCL groups** for each broadcast plan via `make_collective_group()`

**Key Data Structures:**
```python
# model_update_group.py:25
broadcast_comm_pan: Dict[pp_rank, Dict[src_actor_rank, List[tgt_actor_rank]]]

# send_recv_utils.py:72
tensors_meta = {name: {bucket_start, tensor_start, save_bytes, tensor_meta}}
```

**Selective Sync Status:** Not implemented. Full model sync every `frequency` steps. LoRA-only sync available when LoRA is enabled.

#### 1.2 NeMo-RL Framework

**Entry Points:**
- `third_party/nemo-rl/nemo_rl/algorithms/grpo.py:917` - `refit_policy_generation()`
- `third_party/nemo-rl/nemo_rl/models/generation/vllm/vllm_backend.py:128` - `update_weights_via_ipc_zmq()`

**Two Sync Paths:**

1. **Colocated (CUDA-IPC/ZMQ):**
   - Uses ping-pong double buffering for overlap
   - `stream_weights_via_ipc_zmq()` packs tensors, creates IPC handles
   - ZMQ REQ-REP socket with 120s timeout
   - Receiver rebuilds CUDA tensor from IPC handle

2. **Non-Colocated (NCCL Collective):**
   - `packed_broadcast_producer()` / `packed_broadcast_consumer()`
   - Multiple CUDA streams for overlapped broadcast
   - `NRL_REFIT_NUM_BUFFERS` env var controls parallelism (default 2)

**Configuration:**
- `master_config["policy"]["generation"]["colocated"]["enabled"]` selects path
- Buffer size: 30% of free memory (IPC), 2% (NCCL)

#### 1.3 Miles Framework

**Entry Points:**
- `third_party/miles/miles/ray/rollout.py:191` - `RolloutManager.onload_weights()`
- `third_party/miles/miles/backends/sglang_utils/sglang_engine.py:250` - `update_weights_from_tensor()`

**Two Sync Strategies:**

1. **Colocated (Gloo IPC Gather):**
   - `UpdateWeightFromTensor` class creates Gloo groups per engine
   - Uses `FlattenedTensorBucket` for serialization
   - `dist.gather_object()` collects from TP ranks to engine

2. **Distributed (NCCL Broadcast):**
   - `UpdateWeightFromDistributed` uses NCCL process groups
   - Requires `rollout_engine_lock` to prevent NCCL deadlock
   - Sends metadata via Ray RPC, tensor data via `dist.broadcast()`

**Version Tracking:**
- `weight_version` integer incremented on each sync
- Verified in CI via `engine.get_weight_version()` endpoint

#### 1.4 SkyRL-train Framework

**Entry Points:**
- `third_party/SkyRL/skyrl-train/skyrl_train/weight_sync/broadcast_strategy.py:72` - `BroadcastWeightTransferSender`
- `third_party/SkyRL/skyrl-train/skyrl_train/weight_sync/cuda_ipc_strategy.py:90` - `CudaIpcWeightTransferSender`

**Strategy Pattern Architecture:**
- `WeightTransferStrategy` abstract class with factory methods
- `WeightSyncInitInfo` contains rank offsets, group names
- Selection: CUDA IPC when `colocate_all=true` and `weight_sync_backend=nccl`

**Broadcast Strategy:**
- Only rank 0 training worker joins process group with inference engines
- Uses `asyncio.to_thread()` for blocking broadcast operations
- Barrier after each chunk

**CUDA IPC Strategy:**
- Packs all tensors into contiguous buffer
- Creates IPC handle via `torch.multiprocessing.reductions.reduce_tensor()`
- Gathers handles across ranks with `all_gather_object()`
- Receiver uses physical GPU UUID to lookup correct handle

---

### 2. Async Multi-Turn Agentic Rollout Loop

#### 2.1 ROLL Framework (Most Complete)

**Entry Points:**
- `third_party/ROLL/roll/pipeline/agentic/agentic_pipeline.py:134` - `AgenticPipeline.run()`
- `third_party/ROLL/roll/pipeline/agentic/env_manager/traj_env_manager.py:92` - `run_rollout_loop()`

**Pipeline Loop Structure:**
```
for global_step in range(max_steps):
    1. offload_states() - CPU offload training models
    2. suspend() → abort_request() - Stop generation, abort in-flight
    3. model_update() - Sync weights to inference
    4. start_server() - Resume inference
    5. get_batch() - Collect trajectories
    6. train_step() - Update policy
```

**Multi-Turn Rollout (`traj_env_manager.py:92-139`):**
- Each `TrajEnvManager` runs in its own thread
- `make_decision()` calls LLM generation
- On `GenerateStopReason.ABORT`: Does NOT step env, loops to retry same turn
- On `GenerateStopReason.FINISH`: Calls `env.step(action)`
- Trajectory ID format: `{tag}_{group_id}_{episode_id}_{group_seed}_{env_id}`

**Abort/Retry Pattern:**
- `RequestScheduler.suspend()` sets `need_suspend=True`, calls `abort_request()`
- `abort_request()` sends `GenerateRequestType.ABORT` to all DP ranks
- `_check_suspend()` blocks new requests until `resume()` is called
- Env loop detects abort via `lm_output is None`, returns `ABORT` stop reason

**GroupQueue Async Generation:**
- `async_generation_ratio` controls how many training steps old trajectories remain valid
- Groups older than `async_generation_ratio` are expired
- `GroupQueueManager.get_batch()` filters stale rollouts at collection time

**Sticky Routing Map:**
- `src_rank2_dp_rank: Dict[int, int]` maintains session affinity from source rank to DP rank
- New requests from unseen `src_rank` are assigned via round-robin, then cached
- Location: [`third_party/ROLL/roll/distributed/scheduler/generate_scheduler.py:518`](third_party/ROLL/roll/distributed/scheduler/generate_scheduler.py:518)

**GroupQueue/GroupQueueManager:**
- `GroupQueue` manages rollout groups with enqueue/dequeue operations
- `GroupQueueManager` coordinates multiple GroupQueues and filters stale items
- Entry point: [`third_party/ROLL/roll/distributed/scheduler/group_queue.py`](third_party/ROLL/roll/distributed/scheduler/group_queue.py)

#### 2.2 NeMo-RL Framework

**Entry Points:**
- `third_party/nemo-rl/nemo_rl/algorithms/async_utils.py:239` - `AsyncTrajectoryCollector`
- `third_party/nemo-rl/nemo_rl/experience/rollouts.py:786` - `run_async_multi_turn_rollout()`

**AsyncTrajectoryCollector:**
- Ray actor with background collection thread
- `_inflight_threads: set[threading.Thread]` tracks active generation workers
- `_refit_pause_cleared: threading.Event` gates new work during weight update
- `_inflight_sema: Semaphore` limits max concurrent generations

**Version Tagging:**
- Each trajectory tagged with `(generation_weight_version, target_weight_version)` tuple
- `ReplayBuffer.sample()` only consumes trajectories where `target_weight_version == current_weight_version`
- `max_trajectory_age_steps` controls staleness window

**ReplayBuffer Admission Gating:**
- `push_with_wait_signal()` blocks trainer until rollouts are ready
- Buffer enforces consumption ordering based on version matching
- Location: [`third_party/nemo-rl/nemo_rl/algorithms/async_utils.py`](third_party/nemo-rl/nemo_rl/algorithms/async_utils.py)

**_refit_pause_cleared Event:**
- `threading.Event` that acts as admission gate during weight updates
- `clear()` pauses new generation starts; `set()` resumes admission
- Located in `AsyncTrajectoryCollector` for coordinating refit boundaries

**Refit Coordination:**
- `prepare_for_refit()`: Clears `_refit_pause_cleared`, optionally waits for pending
- If `in_flight_weight_updates=True`: Allows ongoing generations to complete
- `resume_after_refit()`: Sets event, optionally invalidates KV caches

**Multi-Turn Execution:**
- `run_async_multi_turn_rollout()` uses `asyncio.gather()` for concurrent per-sample rollouts
- `run_sample_multi_turn_rollout()` iterates turns with environment feedback
- Per-sample tasks execute independently with shared policy generation

#### 2.3 Miles Framework

**Entry Points:**
- `third_party/miles/miles/rollout/sglang_rollout.py:322` - `generate_rollout_async()`
- `third_party/miles/miles/rollout/data_source.py:157` - `RolloutDataSourceWithBuffer`

**GenerateState Singleton:**
- Tracks all in-flight requests globally
- `aborted` flag prevents new requests from starting
- `remaining_batch_size` tracks how many more samples needed

**Abort Flow:**
```python
async def abort(args, rollout_id):
    state.aborted = True
    # POST /abort_request with abort_all=True to all workers
    await asyncio.gather(*[post(f"{url}/abort_request", {"abort_all": True}) for url in urls])
    # Drain pending tasks, collect partial samples for retry
    while state.pendings:
        done, state.pendings = await asyncio.wait(state.pendings, ...)
```

**Retry Buffer:**
- `RolloutDataSourceWithBuffer` extends base data source with `buffer` list
- Aborted samples added to buffer via `add_samples()`
- `get_samples()` drains buffer first (FIFO), then fetches fresh samples

**MilesRouter Health Checks:**
- Background health check loop at configurable interval
- `worker_request_counts` tracks active requests per worker
- `dead_workers` set marks workers after consecutive failures
- Location: [`third_party/miles/miles/router/router.py`](third_party/miles/miles/router/router.py)

**SGLang Targeted Abort:**
- `/abort_request` endpoint supports `rid` parameter for targeted abort
- `abort_all=True` aborts all requests on the engine
- Request ID must be coordinator-provided for targeted cancellation

**Multi-Turn Examples:**
- `examples/retool/generate_with_retool.py`: Treats `finish_reason=abort` as terminal
- `examples/tau-bench/trainable_agents.py`: Similar pattern
- No generic "pause/update/resume same trajectory" mechanism

#### 2.4 SkyRL-train Framework

**Entry Points:**
- `third_party/SkyRL/skyrl-train/examples/async/async_trainer.py:18` - One-step-off async
- `third_party/SkyRL/skyrl-train/skyrl_train/fully_async_trainer.py:79` - Fully async

**One-Step-Off Async:**
- Two concurrent tasks: generator and trainer
- Generator runs one step ahead via `asyncio.Queue(maxsize=1)`
- Trainer waits for `generation_ack` before syncing weights
- `sync_finished` event unblocks generator after weight sync

**Fully Async:**
- `_AsyncStalenessManager` bounds generation ahead of training
- `max_staleness_steps` controls maximum trajectory age
- Multiple `num_parallel_generation_workers` run concurrently
- Each worker acquires capacity slot from staleness manager

**Pause/Resume with Retry:**
```python
async def pause_generation():
    generation_paused_event.set()
    await asyncio.sleep(5)  # Grace period
    await self._run_on_all_engines("abort_generation")

async def resume_generation():
    generation_paused_event.clear()
```

**_AsyncStalenessManager Capacity Slots:**
- Uses capacity slots to bound generation ahead of training
- Each worker acquires slot from manager before generating
- `max_staleness_steps` controls maximum trajectory age
- Location: [`third_party/SkyRL/skyrl-train/skyrl_train/fully_async_trainer.py`](third_party/SkyRL/skyrl-train/skyrl_train/fully_async_trainer.py)

**Fully-Async Admission Gating:**
- `generation_paused_event: asyncio.Event` gates worker execution
- Workers check event before each generation attempt
- Automatic retry loop handles abort with token accumulation

**Automatic Retry:**
- `_generate_single_with_retry()` loops while `stop_reason == "abort"`
- Accumulates tokens from previous attempts
- Adjusts remaining `max_tokens` for continuation

---

### 3. Pipeline Coordinator Control

#### 3.1 ROLL: AgenticPipeline

**Location:** `third_party/ROLL/roll/pipeline/agentic/agentic_pipeline.py`

**Control Hierarchy:**
```
AgenticPipeline.run() [Coordinator]
    ├── actor_train.offload_states() / load_states()
    ├── actor_infer.start_server() / stop_server()
    ├── train_rollout_scheduler.suspend() / resume()
    └── model_update_group.model_update()
```

**Phase Transitions:**
1. Offload training states to CPU
2. Suspend generation (abort in-flight)
3. Model update (sync weights)
4. Start inference server
5. Collect batch via `RolloutScheduler.get_batch()`
6. Train step

#### 3.2 NeMo-RL: GRPO Trainer

**Location:** `third_party/nemo-rl/nemo_rl/algorithms/grpo.py`

**Control Pattern:**
```
grpo_train() [Coordinator]
    ├── policy.offload_before_refit() / offload_after_refit()
    ├── policy_generation.prepare_for_generation() / finish_generation()
    ├── trajectory_collector.prepare_for_refit() / resume_after_refit()
    └── refit_policy_generation() [Weight sync]
```

**Async GRPO:**
- `AsyncTrajectoryCollector` runs as separate Ray actor
- Coordinator signals via `prepare_for_refit()` / `resume_after_refit()`
- Weight version managed via `set_weight_version()`

#### 3.3 Miles: train.py / train_async.py

**Location:** `third_party/miles/train.py`, `train_async.py`

**Sync Training:**
```
train() [Coordinator]
    ├── rollout_manager.onload() / offload()
    ├── rollout_manager.generate()
    ├── actor_model.async_train()
    └── actor_model.update_weights()
```

**Async Training:**
- Overlaps generation N+1 with training N
- Critical sync fence before `update_weights()`:
  ```python
  # Sync generate before update to prevent mid-generation weight changes
  rollout_data_curr_ref = ray.get(rollout_data_next_future)
  actor_model.update_weights()
  ```

#### 3.4 SkyRL-train: Async Trainers

**Location:** `third_party/SkyRL/skyrl-train/skyrl_train/fully_async_trainer.py`

**Control Flow:**
```
FullyAsyncRayPPOTrainer [Coordinator]
    ├── inference_engine_client.pause_generation()
    ├── async_sync_policy_weights_to_inference_engines()
    └── inference_engine_client.resume_generation()
```

---

### 4. Memory Offloading (Weights, KV Cache, Optimizer)

#### 4.1 ROLL Offload

**Location:** `third_party/ROLL/roll/distributed/strategy/megatron_strategy.py:1196-1209`

**Megatron Offload:**
```python
def offload_states(self, include=None, non_blocking=False, pin_memory=True):
    if OffloadStateType.model_params in include:
        offload_megatron_no_grad_module(model_chunks, pin_memory=pin_memory)
    self.optimizer.offload_states(include=include, ...)
    RotaryEmbedding.forward.cache_clear()
    current_platform.empty_cache()
```

**vLLM Offload:**
```python
def offload_states(self, include=None, non_blocking=False):
    self.model.offload_states(self.sleep_level)  # Configurable level
    self.is_model_in_gpu = False
```

#### 4.2 NeMo-RL Offload

**Location:** `third_party/nemo-rl/nemo_rl/models/generation/vllm/vllm_worker.py:807-823`

**vLLM Sleep/Wake:**
```python
def sleep(self):
    self.llm.llm_engine.reset_prefix_cache()  # Reset KV cache
    self.llm.sleep(level=1)  # Level 1 for colocated
    gc.collect()
    torch.cuda.empty_cache()

def wake_up(self, **kwargs):
    await self.llm.wake_up(**wake_up_args)  # Optional tags for staged loading
```

**Tags for Staged Loading:**
- `"weights"`: Load model weights only
- `"kv_cache"`: Load KV cache only

#### 4.3 Miles Offload

**Location:** `third_party/miles/miles/ray/rollout.py:176-195`

**Staged Memory Management:**
```python
def offload(self):
    self.health_monitoring_pause()
    ray.get([engine.release_memory_occupation.remote() for engine in self.rollout_engines])

def onload_weights(self):
    self.onload(tags=[GPU_MEMORY_TYPE_WEIGHTS])

def onload_kv(self):
    self.onload(tags=[GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_CUDA_GRAPH])
```

**SGLang Memory Tags:**
- `GPU_MEMORY_TYPE_WEIGHTS`
- `GPU_MEMORY_TYPE_KV_CACHE`
- `GPU_MEMORY_TYPE_CUDA_GRAPH`

#### 4.4 SkyRL Offload

**Location:** `third_party/SkyRL/skyrl-train/skyrl_train/inference_engines/vllm/vllm_engine.py:447-468`

**Sleep with Abort:**
```python
async def sleep(self, *args, **kwargs):
    if output_processor.has_unfinished_requests():
        await engine.abort(unfinished_request_ids)
    await self.reset_prefix_cache()  # Avoid KV pollution
    level = 1 if self._is_lora else kwargs.get("level", 2)
    await self.llm.sleep(level=level)
```

---

### 5. vLLM v1 Engine Integration

#### 5.1 Integration Patterns Comparison

| Feature | ROLL | NeMo-RL | SkyRL-train |
|---------|------|---------|-------------|
| **Request ID** | External from meta | `uuid.uuid4()` | `uuid4().hex` |
| **Abort** | `model.abort_request(id)` | via `collective_rpc` | `engine.abort(ids)` |
| **Weight Update** | `broadcast_bucket` | IPC ZMQ or collective | `collective_rpc("load_weights")` |
| **Sleep Level** | Configurable | Fixed at 1 | 1 for LoRA, else 2 |
| **Prefix Cache Reset** | Before offload | Before sleep | Before sleep |

#### 5.2 vLLM v1 Native APIs

**Location:** `third_party/vllm/vllm/v1/engine/async_llm.py`

**Abort:**
```python
async def abort(self, request_id: str | Iterable[str], internal: bool = False):
    all_request_ids = self.output_processor.abort_requests(request_ids, internal)
    await self.engine_core.abort_requests_async(all_request_ids)
```

**Pause/Resume (RL-optimized):**
```python
async def pause_generation(self, wait_for_inflight_requests=False, clear_cache=True):
    self._paused = True
    if not wait_for_inflight_requests:
        await self.abort(request_ids, internal=True)
    if self.output_processor.has_unfinished_requests():
        await self.output_processor.wait_for_requests_to_drain()
    if clear_cache:
        await self.reset_prefix_cache()
```

**Sleep/Wake:**
```python
async def sleep(self, level: int = 1):
    await self.reset_prefix_cache()
    await self.engine_core.sleep_async(level)

async def wake_up(self, tags: list[str] | None = None):
    await self.engine_core.wake_up_async(tags)
```

---

### 6. SGLang Engine Integration (Fallback)

#### 6.1 Integration Patterns

**Miles:** Full integration with MilesRouter, abort via `/abort_request` endpoint, staged memory management

**ROLL:** Version-specific patches (v046, v052, v054), custom `EngineSA` class for weight sync

**SkyRL:** The `InferenceEngineClient` base's `abort_generation()` raises `NotImplementedError`, but concrete engine implementations (e.g., vLLM, SGLang, remote/ray-wrapped engines) implement `abort_generation()`; the client delegates aborts to engines during `pause_generation()`.

#### 6.2 Key APIs

**Abort:**
```python
# HTTP endpoint
@app.post("/abort_request")
async def abort_request(obj: AbortReq):
    _global_state.tokenizer_manager.abort_request(rid=obj.rid, abort_all=obj.abort_all)
```

**Memory Management:**
```python
def release_memory_occupation(self):
    self.flush_cache()  # Wait for queue empty
    return self._make_request("release_memory_occupation")

def resume_memory_occupation(self, tags: list[str] = None):
    return self._make_request("resume_memory_occupation", {"tags": tags})
```

---

### 7. Request Dispatcher Load Balancing

#### 7.1 ROLL RequestScheduler

**Location:** `third_party/ROLL/roll/distributed/scheduler/generate_scheduler.py:866-970`

**Sticky Routing:**
```python
src_rank2_dp_rank = {}  # Maps source rank → DP rank

async def generate_one_request(self, data):
    src_rank = data.meta_info["src_rank"]
    if src_rank not in self.src_rank2_dp_rank:
        dp_rank = next(self.worker_iter)  # Round-robin
        self.src_rank2_dp_rank[src_rank] = dp_rank
    dp_rank = self.src_rank2_dp_rank[src_rank]
```

**Load Balance Coordinator (GenerateScheduler):**
```python
def get_available_dp_rank(self):
    sorted_ranks = sorted(
        self.load_balance_coordinator.keys(),
        key=lambda rank: (self.load_balance_coordinator[rank], rank)
    )
    if self.load_balance_coordinator[sorted_ranks[0]] < self.max_running_requests:
        yield sorted_ranks[0]
```

#### 7.2 Miles MilesRouter

**Location:** `third_party/miles/miles/router/router.py`

**Least-Connections:**
```python
def _use_url(self):
    valid_workers = (w for w in self.worker_request_counts if w not in self.dead_workers)
    url = min(valid_workers, key=self.worker_request_counts.get)
    self.worker_request_counts[url] += 1
    return url
```

**Health-Based Dead Worker Tracking:**
- Background health check loop at configurable interval
- Consecutive failures tracked per worker
- Workers marked dead after threshold failures

## 8. Rollout Buffer / Backlog Accounting Primitives

This section captures the **existing** per-framework primitives that track “how much rollout work is left” and/or how much is currently in flight.

### 8.1 ROLL

- `GroupQueueManager` (used by the agentic rollout pipeline) manages queued vs ready rollout items at the *group* level and filters stale items based on `async_generation_ratio`.

### 8.2 NeMo-RL

- `AsyncTrajectoryCollector` tracks in-flight generation workers/threads and gates new collection during refit via an event.
- `ReplayBuffer.sample()` enforces consumption based on per-trajectory version tags (consumes only those matching the current target version).

### 8.3 Miles

- `GenerateState.remaining_batch_size` tracks how many more rollout samples are still required for the current target data size in the async SGLang rollout path.
- `RolloutDataSourceWithBuffer` provides a FIFO buffer for retry/oversampling handling.

### 8.4 SkyRL-train

- One-step-off async uses an `asyncio.Queue(maxsize=1)` to bound generator-vs-trainer lag.
- Fully async uses `_AsyncStalenessManager` to bound how far generation can run ahead of training.

## Code References

### Weight Sync
- `third_party/ROLL/roll/distributed/executor/model_update_group.py:139` - ROLL bucket broadcast
- `third_party/nemo-rl/nemo_rl/algorithms/grpo.py:917` - NeMo-RL refit orchestration
- `third_party/miles/miles/backends/fsdp_utils/update_weight_utils.py:32` - Miles UpdateWeight base
- `third_party/SkyRL/skyrl-train/skyrl_train/weight_sync/transfer_strategy.py:52` - SkyRL strategy pattern

### Async Rollout
- `third_party/ROLL/roll/pipeline/agentic/env_manager/traj_env_manager.py:92` - ROLL multi-turn loop
- `third_party/ROLL/roll/distributed/scheduler/group_queue.py` - ROLL GroupQueue/GroupQueueManager
- `third_party/nemo-rl/nemo_rl/algorithms/async_utils.py:239` - NeMo-RL AsyncTrajectoryCollector
- `third_party/nemo-rl/nemo_rl/algorithms/async_utils.py` - NeMo-RL ReplayBuffer push_with_wait_signal
- `third_party/miles/miles/rollout/sglang_rollout.py:322` - Miles async rollout
- `third_party/miles/miles/router/router.py` - MilesRouter health checks
- `third_party/SkyRL/skyrl-train/skyrl_train/fully_async_trainer.py:79` - SkyRL staleness manager

### Abort/Retry
- `third_party/ROLL/roll/distributed/scheduler/generate_scheduler.py:939` - ROLL abort_request
- `third_party/miles/miles/rollout/sglang_rollout.py:282` - Miles abort function
- `third_party/SkyRL/skyrl-train/skyrl_train/inference_engines/inference_engine_client.py:580` - SkyRL pause_generation

### Memory Management
- `third_party/ROLL/roll/distributed/strategy/vllm_strategy.py:416` - ROLL offload_states
- `third_party/nemo-rl/nemo_rl/models/generation/vllm/vllm_worker.py:807` - NeMo-RL sleep
- `third_party/miles/miles/ray/rollout.py:176` - Miles staged offload
- `third_party/vllm/vllm/v1/engine/async_llm.py:904` - vLLM native sleep

---

## Architecture Documentation

### Common Patterns Across Frameworks

1. **Bucket-Based Weight Transfer**: Most frameworks use packing/bucketing for weight transfer, but sizing and transport vary (e.g., NeMo uses `NRL_REFIT_BUFFER_MEMORY_RATIO` default 0.02 of GPU memory with a 5GB cap; SkyRL uses packed CUDA-IPC handles).
2. **Event-Based Admission Gating**: threading.Event or asyncio.Event controls new request admission
3. **Version Tagging**: Trajectories can be tagged with a generation weight version for staleness control.
4. **Two-Phase Offload**: Weights and KV cache can be managed independently.
5. **Retry at Turn Boundary**: Aborted turns retry without restarting the entire trajectory.

---

### SchedRL Integration Points

Per design docs, these are the key extension points:

| Framework | Subset Lifecycle | Shrink Migration | Expand Rebalance | Admission Close | Selective Sync |
|-----------|------------------|------------------|------------------|-----------------|----------------|
| **ROLL** | Missing (cluster-wide only) | Partial (abort exists) | Missing | Missing | Missing |
| **NeMo-RL** | Missing (all workers) | Missing | Partial (round-robin) | Present | Partial |
| **Miles** | Missing (cluster-wide) | Partial (abort exists) | Partial (global buffer) | Missing | Partial |
| **SkyRL** | Present (config-based) | Partial (pause/abort) | Partial (routing) | Present | Present |

---

## Optional Enhancements (Additional Mechanisms)

The following mechanisms exist in the codebase and may be relevant for extended SchedRL functionality:

### A.1 vLLM Sleep Level Semantics

vLLM v1 engine supports multiple sleep levels for different offloading strategies:

| Sleep Level | Behavior | Use Case |
|-------------|----------|----------|
| **Level 1** | Drop KV cache, keep model weights in GPU memory | Colocated inference (fast wake-up) |
| **Level 2** | Drop both KV cache and model weights | Non-colocated inference (full GPU release) |

**References:**
- NeMo-RL: Uses `level=1` for colocated inference ([`third_party/nemo-rl/nemo_rl/models/generation/vllm/vllm_worker.py:817`](third_party/nemo-rl/nemo_rl/models/generation/vllm/vllm_worker.py:817))
- SkyRL: Uses `level=1` for LoRA, `level=2` otherwise ([`third_party/SkyRL/skyrl-train/skyrl_train/inference_engines/vllm/vllm_engine.py:455`](third_party/SkyRL/skyrl-train/skyrl_train/inference_engines/vllm/vllm_engine.py:455))

### A.2 SkyRL-train Async Constraints

SkyRL-train async modes have specific configuration requirements:

**One-Step-Off Async:**
- Requires `trainer.placement.colocate_all=false`
- Generator and trainer must be on separate GPU sets

**Fully-Async:**
- Requires `trainer.placement.colocate_all=false`
- Requires `generator.batched=false` (pause/resume not supported for batched generate)
- Requires `generator.async_engine=true`
- vLLM-only (SGLang abort not implemented in SkyRL-train)

**References:**
- [`third_party/SkyRL/skyrl-train/skyrl_train/fully_async_trainer.py`](third_party/SkyRL/skyrl-train/skyrl_train/fully_async_trainer.py) (assertions at initialization)
- [`third_party/SkyRL/skyrl-train/examples/async/async_trainer.py`](third_party/SkyRL/skyrl-train/examples/async/async_trainer.py)

### A.3 NeMo-Gym Mini-SWE Agent Integration

NeMo-Gym provides a Mini-SWE agent server that can be integrated with NeMo-RL for agentic training:

**Components:**
- Agent server: [`third_party/nemo-gym/responses_api_agents/mini_swe_agent/`](third_party/nemo-gym/responses_api_agents/mini_swe_agent/)
- Resource configs: [`third_party/nemo-gym/resources_servers/mini_swe_agent/`](third_party/nemo-gym/resources_servers/mini_swe_agent/)
- Task definition: Uses OpenAI-compatible responses API with tool calling

**Integration Pattern:**
- NeMo-RL rollout task calls the Mini-SWE agent server via HTTP
- Agent server executes shell/git commands in sandboxed environment
- Responses include tool calls that become the environment's next observation

### A.4 NeMo-RL ReplayBuffer Version Consumption

NeMo-RL's `ReplayBuffer` enforces strict version matching for consumption:

**Mechanism:**
```python
# Pseudo-code from third_party/nemo-rl/nemo_rl/algorithms/async_utils.py
if trajectory.target_weight_version == current_weight_version:
    consume(trajectory)
else:
    hold_until_version_matches(trajectory)
```

**Key Points:**
- Trajectories tagged with `(generation_weight_version, target_weight_version)` tuple
- Training only consumes when `target_weight_version == current_weight_version`
- Staleness bounded by `max_trajectory_age_steps` (configurable)
- Enables bounded staleness async training without strict quiescence

---

## Related Research

- `design_doc/multi-pipeline-adaptation-plan.md` - Main protocol specification
- `design_doc/adaptation_roll.md` - ROLL-specific implementation gaps
- `design_doc/archive/adaptation_nemo_rl.md` - NeMo-RL-specific gaps (deferred; archived)
- `design_doc/archive/adaptation_miles.md` - Miles-specific gaps (deferred; archived)
- `design_doc/adaptation_skyrl.md` - SkyRL-specific gaps

---

## Open Questions

1. **Subset NCCL Groups**: How to efficiently create/teardown per-subset communication groups for selective sync?
2. **Request ID Ownership**: NeMo-RL uses worker-generated UUIDs; SchedRL requires coordinator-provided IDs for targeted abort
3. **SGLang Abort in SkyRL**: Currently `NotImplementedError` - needs implementation for fallback support
4. **Router Worker Removal**: MilesRouter missing `/remove_worker` endpoint for subset admission control
5. **Creation Timestamp Tracking**: Required for `oldest_unfinished_creation_ts` but not tracked in most frameworks
