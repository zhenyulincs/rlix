# Phase B Implementation — F1, F2, F3 (Engine Lifecycle)

## Files Changed

### `miles/rollout/base_types.py`
Added `EnginePreemptedError` and `RLixRouterMetadataError` exception classes.

- `EnginePreemptedError`: raised when turn-level redispatch is exhausted (all engines tried) — signals topology/admission/scheduler bug, caught by `_FatalError` sentinel in `fully_async_rollout.py`
- `RLixRouterMetadataError`: raised when RLix mode detects missing router metadata in a `/generate` response — router upgrade incomplete; NOT catchable by turn retry, propagates to fatal sentinel

### `miles/backends/sglang_utils/sglang_engine.py` (F1)
Added 4 RLix sleep/wake helper methods:

- `get_server_url()`: returns `http://host:port` (used by RolloutManager._build_engines_map)
- `is_idle()`: GET `/v1/loads` → checks `num_total_reqs == 0` for all slots; `raise_for_status()` to fail-fast on hung server (not silently idle)
- `abort_all_requests()`: POST `/abort_request {"abort_all": true}` — clears all in-flight requests before sleep
- `assert_post_sleep_memory_below_threshold(threshold_gb)`: GET `/server_info` → sums `memory_usage.{weight,kvcache,graph}` across `internal_states`; asserts total < threshold (default 1.0 GB); NCCL communicator residual (~50-200 MB) is acceptable and covered by threshold

### `miles/ray/rollout.py` (F2)
Added `EngineInfo` dataclass and engine lifecycle to `RolloutManager`:

**`EngineInfo`** dataclass:
- `handle`: SGLangEngine Ray actor handle
- `worker_url`: `http://host:port` for router admission
- `state`: `Literal["active", "disabling", "offloaded", "loading"]` — single source of truth for dispatch eligibility

**`RolloutManager.__init__` additions**:
- `_engines: Dict[int, EngineInfo]` — per-engine lifecycle map (populated by `initialize_rlix_engine_map`)
- `_routing_lock: threading.Lock` — protects dispatch selection and state transitions; separate from `rollout_engine_lock` (cross-process distributed lock)
- `_active_engine_indices: Set[int]`, `_preempted_engines: Set[int]`
- drain poll/timeout env vars (`MILES_DRAIN_POLL_INTERVAL_S`, `MILES_DRAIN_TIMEOUT_S`)

**New methods**:
- `initialize_rlix_engine_map()`: called by `MilesPipeline` Step 8; enumerates engines via `server.engines`, gets URLs via `engine.get_server_url.remote()`, initializes all as `state=active`
- `get_engine_count()`: returns `len(server.engines)`
- `set_weight_version(version, engine_indices=None)`: fan-out `update_weight_version.remote(str(version))`; waits all engines confirm
- `finalize_engine(engine_index)`: calls `finalize_weight_update.remote()` on specific engine (Phase C receiver API)
- `shutdown_hard()`: stops monitors + `ray.kill` all engine actors (M4 cleanup)
- `_disable/enable_engine_in_router(idx)`: HTTP to router `/disable_worker` / `/enable_worker`
- `_wait_engine_idle(idx)`: polls `engine.is_idle.remote()` until true or `_drain_timeout_s`
- `sleep_partial(engine_indices)`: short critical section (lock) → admission close → abort → drain → sleep → post-sleep VRAM assert
- `wake_partial(engine_indices)`: resume_memory_occupation → enable in router → state=active
- `shrink_engines(engine_indices)`: alias for `sleep_partial` (ROLL `shrink_workers` equivalent)
- `expand_engines(engine_indices)`: alias for `wake_partial` (weight sync done by pipeline after expand returns)

**Invariants**:
- Lock protects only state transitions + set mutations; slow ops (abort/drain/sleep/wake) run outside lock
- `_preempted_engines` set/clear: set in `sleep_partial` before lock release, cleared in `wake_partial` after state=active (covers the wake transition window)

### `miles/router/router.py` (F3)
**New state fields**: `enabled_workers: set[str]`, `worker_engine_index_map: dict[str, int]`

**4 internal helpers** (lifecycle encapsulation):
- `_add_worker_internal(url, engine_index)`: `setdefault` (no counter reset on re-add), `enabled_workers.add`, `dead_workers.discard`
- `_remove_worker_internal(url)`: full unregistration
- `_disable_worker_internal(url)`: admission close + reset failure count (prevents sleep-period probe accumulation)
- `_enable_worker_internal(url)`: re-open admission only if still registered (health check re-probes dead workers)

**3 new endpoints**: `POST /disable_worker`, `POST /enable_worker`, `POST /remove_worker`

**`add_worker` F3 extension**: accepts `?engine_index=N`; uses `_add_worker_internal`

**`_use_url` fix**: routes only to `enabled_workers - dead_workers` (Critical Invariant — otherwise `/disable_worker` only affects metadata, new requests still routed to sleeping engine)

**`_health_check_loop` fix**: only probes `enabled_workers - dead_workers` (otherwise sleeping engines accumulate health failures → permanently marked dead → can't re-admit after wake)

**`do_proxy` metadata injection (F3)**: for `path == "generate"` only:
- Strips `Content-Encoding` before JSON mutation (stale encoding + modified body = corrupt response)
- `meta_info.miles_engine_index`: from `worker_engine_index_map`
- `meta_info.miles_admission_disabled`: `worker_url not in enabled_workers` at response time (false positives → wasted retry; false negatives → practical impossibility at sec-vs-ms scale)
- Non-JSON body passes through; RLix mode triggers `RLixRouterMetadataError` on missing metadata

### `miles/rollout/generate_hub/multi_turn.py` (F3)
- Deleted `assert not args.partial_rollout` (line 29)
- Force `payload["stream"] = False` (SSE has no `meta_info`, breaks preempt classification)
- `_is_scheduler_preempt(output, rlix_mode)`: checks `finish_reason.type == "abort"` + `miles_admission_disabled`; raises `RLixRouterMetadataError` in RLix mode on missing metadata
- Turn-level redispatch loop: `MAX_TURN_REDISPATCH = total_engine_count`; snapshot before each attempt; `_restore_turn_state` on preempt; raise `EnginePreemptedError` on exhaustion (not silent group recycle)
- `update_sample_from_response` called AFTER preempt check (commit point after successful generation)

### `miles/rollout/generate_utils/generate_endpoint_utils.py` (F3)
- `_snapshot_turn_state(sample)`: records lengths of `tokens/response/weight_versions/loss_mask` — O(1) snapshot, not deep copy
- `_restore_turn_state(sample, snapshot)`: truncates fields to snapshot lengths — O(increment) rollback, preserves earlier turns

### `examples/fully_async/fully_async_rollout.py` (F3, F9)
- `_FatalError` sentinel class (not Exception subclass — prevents accidental catch by broad handlers)
- `make_callback` catches `(EnginePreemptedError, RLixRouterMetadataError)` → puts `_FatalError(exc)` in output_queue
- Main loop detects `isinstance(group, _FatalError)` → `raise group.exc` (pipeline crash)
- `generate_rollout_async` accepts `rlix_hooks` parameter (default `NoOpRLixHooks`)
- F9 scaffolding: `begin_progress_batch` before loop, `bump_completed` on each group push, `end_progress_batch` in finally

## Key Invariants
- `EngineInfo.state` is single source of truth for dispatch eligibility; `_active_engine_indices` is derived cache
- `_preempted_engines` tracks preempt attribution window (spans multiple state phases)
- Router health probe skips disabled workers (prevents false dead marking during sleep)
- Turn retry exhaustion = topology/admission bug → fail fast (not silent group recycle)
