# Phase D+E Implementation — F9 (Progress Hooks) + F12 (Placement Adapter)

## Phase D — F9

### `rlix/pipeline/miles_hooks.py` (new)
`MilesRLixHooks` implementing `RLixHooks` protocol.

Import seam: `fully_async_rollout.py` only calls hook methods. Never imports `ProgressReport` or touches coordinator handles.

Local state per wait window: `_progress_target_step`, `_step_target_groups`, `_local_completed`, `_progress_last_bucket`, `_active`.

**Key invariants:**
- `begin_progress_batch`: sets `_local_completed = initial_completed` (NOT reset to 0) — batch-open snapshot semantics
- `bump_completed`: 2% gate in reporter layer; skips groups with wrong `target_weight_version`
- `end_progress_batch`: in `finally` guarantee; calls `clear_progress_stream` on coordinator
- All RPCs are fire-and-forget (no `ray.get`); failures logged and swallowed

## Phase E — F12

### `external/miles/miles/ray/placement_provider.py` (new)
`WorkerPlacement` dataclass + `MilesPlacementProvider`:

- `WorkerPlacement`: multi-node-compatible — `node-local gpu_ids` (NOT global physical IDs)
- `MilesPlacementProvider`: converts RLix/ROLL allocation → MILES `(pg, bundle_indices, gpu_ids)` triple
- Phase C interim path: `standalone_pgs` from `create_placement_groups(args)` (pre-created PGs)
- Phase E full path: `RollResourceManagerProxy.allocate_placement_group(world_size, device_mapping=declared)`
- Structural asserts: engine count matches derived value; each engine's `gpu_ids` are sorted
- `train_device_mapping` / `infer_device_mapping` declared at construction (same source as `register_pipeline`) — prevents multi-pipeline device mapping divergence
