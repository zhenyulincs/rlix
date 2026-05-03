# Phase C Implementation — F4, F5/F6 (Weight Transfer)

## Files Changed

### `miles/backends/megatron_utils/update_weight/cpu_bucket_cache.py` (new)
CPU bucket cache core:
- `BucketRecord` dataclass: `param_names, shapes, dtypes, offsets, used_bytes, cpu_uint8_bucket`
- `_pack_tensors_to_buckets(hf_named_tensors, bucket_size_bytes)`: greedy packing into pinned CPU uint8 tensors; fail-fast on single param > bucket_size_bytes
- `CpuBucketCache`: singleton per Megatron worker process; `_cache_lock` (threading.Lock) held for full build + full transport; `_cache_ready_step` version tag only (not slot selector)
- `build(hf_named_tensors, step, bucket_size_bytes, check_host_ram)`: host RAM check (2× model_bytes < 80% available); packs tensors
- `serialize_bucket_to_bytes(bucket_idx)`: produces `{"bucket": cpu_uint8_tensor, "tensors_meta": list[dict]}` for cpu_serialize wire format
- `get_staging_tensor(bucket_idx)`: H2D staging for NCCL broadcast (caller must free after broadcast)

### `miles/backends/megatron_utils/actor.py` (modified)
Added F4 sender API to `MegatronTrainRayActor`:
- `_rlix_is_cache_owner()`: reuses exact `_is_distributed_src_rank` criterion (pp0+dp0+tp0+cp0)
- `build_cpu_bucket_cache(step)`: all ranks participate in HF gather collective; only owner stores result
- `report_cache_owner_role()`: returns `(global_rank, is_cache_owner)` for Step 6.5 discovery
- `get_bucket_count()`, `serialize_bucket_to_objref(bucket_idx)`: cpu_serialize sender
- `setup_collective_group(group_name, src_rank, master_addr, master_port, world_size, timeout_s)`: creates dynamic NCCL group + warmup allreduce
- `broadcast_bucket(group_name, bucket_idx, src_rank)`: per-bucket H2D staging → `dist.broadcast` → free staging
- `destroy_collective_group(group_name)`, `get_cache_ready_step()`, `get_bucket_meta(bucket_idx)`

### `miles/ray/actor_group.py` (modified)
Added to `RayTrainGroup`:
- `async def build_cpu_bucket_cache(step)`: fan-out via `_broadcast`
- `async def collect_cache_owner_roles()`: gathers (rank, is_owner, handle) from all workers

### `miles/backends/sglang_utils/sglang_engine.py` (modified)
Added receiver methods (F5+6):
- `setup_collective_group(group_name, mode, comm_plan, timeout_s)`: HTTP to SGLang
- `broadcast_parameter(group_name, broadcast_local_ranks, bucket_meta)`: receives NCCL broadcast
- `destroy_collective_group(group_name)`: with `is_group_exist` guard semantics (HTTP)
- `update_weights_from_cpu_bucket(payload_bytes, load_format, flush_cache, weight_version, cpu_serialize_local_ranks)`: Ray auto-derefs ObjectRef → bytes; writes to `/dev/shm/miles_cpu_bucket_{uuid}.pt`; HTTP to SGLang; `try/finally os.unlink`

### `miles/ray/rollout.py` (modified)
Added `call_engine_method(engine_index, method_name, *args, **kwargs)` to `RolloutManager`: allows `MilesModelUpdateService` to call engine methods by index without holding engine handles directly.

### `rlix/pipeline/miles_model_update_service.py` (new)
`MilesModelUpdateService` Ray actor:
- `sync_selected_workers(sync_id, tgt_engine_indices)`: 5-phase protocol
  1. Get bucket count from cache owner
  2. Classify: colocate (cpu_serialize) vs non-colocate (NCCL) based on `train_devices ⊂ engine_gpus`
  3. NCCL: claim port via SharedStorage, setup group on cache owner + engines
  4. Per-bucket: cpu_serialize (serial per engine per tmpfs constraint) + NCCL broadcast (parallel)
  5. Destroy NCCL groups + release port claim in finally

Timeout: `ROLL_SELECTIVE_MODEL_UPDATE_TIMEOUT_S` (default 150s)

### `rlix/pipeline/miles_pipeline.py` (modified)
Filled in `initialize_pipeline()` Steps 1b-10:
- Step 1b: Create `RayTrainGroup` with `pg=create_placement_groups(args)` (Phase E replaces with worker_placements), `num_gpus_per_actor=0.01`
- Steps 2-3: `run(actor_train.init())`, `run(actor_train.onload())`
- Step 4: `run(actor_train.build_cpu_bucket_cache(step=-1))`, `_cache_ready_step=-1`
- Step 5: `run(actor_train.offload())`
- Step 6.5: `run(actor_train.collect_cache_owner_roles())` → `_cache_owner_actor`
- Step 7: `_request_cluster_gpus(GENERATION)` + M1 full-alloc assert
- Step 8: Create `RolloutManager` + `initialize_rlix_engine_map()`
- Step 9: `get_engine_count()` sanity check
- Step 10: `bootstrap_active_engines()` + `register_model_update_resources()`

Implemented `_after_training(step)`:
1. `build_cpu_bucket_cache(step)` + `_cache_ready_step = step`
2. `actor_train.offload()`
3. `coordinator.sync_base_weights_to_active()` (active in-flight refresh)
4. `finalize_engine()` on synced engines
5. Version publish (`set_weight_version`)
6. `_notify_release_cluster_gpus(actor_train)` — must be AFTER sync+finalize

Implemented `_expand_workers(engine_indices)`:
1. `actor_infer.wake_partial(engine_indices)`
2. `MilesModelUpdateService.sync_selected_workers(sync_id, engine_indices)`
3. `finalize_engine()` on expanded engines
4. Version publish with specific engine_indices
5. Version published BEFORE routing activates (caller's resize_infer updates active set after return)

## Key Invariants
- `_cache_ready_step` is version tag only; single slot overwrites on each step
- `_cache_lock` held for full build OR full transport (not both simultaneously — ordering invariant prevents concurrent build+transport)
- Per-bucket serial for cpu_serialize (tmpfs peak = 1× bucket); parallel NCCL across engines per bucket
- `notify_release_cluster_gpus` AFTER sync+finalize (ordering invariant)
- Version published BEFORE routing activates (expand_workers ordering)
