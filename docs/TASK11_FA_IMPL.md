# Phase A Implementation — F7, F8, F10, F11 (Control Plane Skeleton)

## Files Changed

### `rlix/pipeline/miles_coordinator.py` (new)
`MilesCoordinator` Ray actor implementing `Coordinator` ABC.

**Why not subclass `PipelineCoordinator`:** PipelineCoordinator's `__init__` runs 4 ROLL config validators (`_validate_config_schema`, `_validate_cpu_only_reward`, `_validate_vllm_sleep_level`, `_validate_offload_nccl`) that expect ROLL `WorkerConfig` attributes absent from MILES `args` namespace. Manually initializing all fields avoids triggering these validators while preserving full Coordinator ABC compliance.

Key invariants:
- `_resize_sync_lock` serializes `resize_infer` and `sync_base_weights_to_active` (same lock pattern as `PipelineCoordinator`)
- `_active_engine_indices` starts empty; bootstrapped exactly once by `initialize_pipeline` Step 10 via `bootstrap_active_engines()`; subsequent updates via `resize_infer()`
- `bootstrap_active_engines()` raises `RuntimeError` on second call (prevents silent overwrite during live resize)
- `sync_lora_weights()` raises `NotImplementedError` (LoRA = M11.5 scope)
- `_get_or_create_model_update_service()` lazy-creates `MilesModelUpdateService` on first sync call (Phase C) to avoid extra actor round-trip during init bootstrap
- Identity mapping `dp_rank == engine_index` in first-build topology (F12 enforces sorted contiguous infer_device_mapping; non-contiguous adapter is follow-up)

### `rlix/pipeline/miles_pipeline.py` (new)
`MilesPipeline` Ray actor with F10 topology validation fully implemented.

**`_validate_rlix_topology(args)`** (F10): All 14 topology asserts run before any GPU allocation:
1. `train ⊂ infer` (partial overlap requirement)
2. `engine_count >= 2`
3. `len(non_overlap_gpus) >= rollout_num_gpus_per_engine` (≥1 full engine always active)
4. `fullasync` rollout function path
5. `offload_train=True` (actor.sleep() has internal assert)
6. `model_update_transport == "cpu_serialize"` (M11.1 vast.ai constraint)
7. `sglang_data_parallel_size == 1`
8. No PD disaggregation
9. No MoE/EP (`expert_model_parallel_size == 1`, `moe_router_topk == 0`)
10. No `async_save` (races with actor.sleep() torch_memory_saver.pause())
11. No `RadixTreeMiddleware`
12. Single updateable model (no critic/reward model path)
13. Megatron parallelism divisibility (`n_train % tp*pp*cp*ep == 0`)
14. Cross-node TP blocked (M11.3 scope): `rollout_num_gpus_per_engine <= num_gpus_per_node`
15. Sorted contiguous `infer_device_mapping` (first-build constraint)
16. No `rollout_force_stream` (router metadata injection requires JSON body)
17. `/dev/shm` writable + capacity check (cpu_serialize tmpfs path)

**`initialize_pipeline()`**: sync def with `threading.Lock` (not async — avoids half-async/sync blocking the event loop via `ray.get` calls to `_request_cluster_gpus`). Phase A implements the structural flow with Phase C stubs for Steps 1b-5 (require cpu_bucket_cache) and Steps 6.5/8/9 (require RolloutManager + MilesModelUpdateService).

**M4 hard cleanup**: Two-phase cleanup correctly handles partial init failure:
- Phase 1 finally: kills train actors only when `train_init_succeeded=False`; always releases train GPU scheduler allocation
- Phase 2 except: kills train actors (even when `train_init_succeeded=True` since Phase 2 failure means pipeline is unusable), calls `shutdown_hard` on infer, conditionally releases infer scheduler (gated by `actor_infer_allocated` bool)

### `examples/rlix/run_miles_rlix.py` (new)
F8 registration driver. Enforces `allocate_pipeline_id → register_pipeline → admit_pipeline → create coordinator → create_pipeline_actor(pipeline_config=)` order (keyword-only call per Coordinator ABC).

Does NOT call `ray.shutdown()` — user calls `ray stop` CLI (F11 behavior; driver exit propagates failure naturally).

### `external/miles/miles/utils/rlix_hooks.py` (new)
`RLixHooks` protocol + `NoOpRLixHooks`. Import seam: `fully_async_rollout.py` only calls hook methods, never imports `ProgressReport` or RLix types. All RLix wire construction is in `MilesRLixHooks` (Phase D).

### `external/miles/train_async.py` (modified)
Two F11 guards:
1. Module-level: raises immediately if `RLIX_CONTROL_PLANE=rlix` (wrong entry point)
2. `__main__` block: `_check_partial_overlap_topology(args)` fails fast when train ⊂ infer but RLix control is absent (would OOM silently from full-broadcast weight sync)

### `rlix/pipeline/__init__.py` (modified)
Added exports for `MilesCoordinator`, `MILES_COORDINATOR_MAX_CONCURRENCY`, `MilesPipeline`.

## Key Invariants
- F7 namespace isolation: all MILES Ray actors created in `get_pipeline_namespace(pipeline_id)` namespace with `pipeline_id` in actor names
- F8 registration: allocate → register → admit order is enforced by orchestrator (register before admit, admit before GPU allocation)
- F10 validation runs before any GPU allocation — init fails early with clear message
- F11 standalone vs RLix entry point separation is hard (module-level raise, not soft warning)
