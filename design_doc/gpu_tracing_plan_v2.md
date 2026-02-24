# GPU Timeline Tracing Implementation Plan (v10 - Final)

**Date**: 2024-02-24
**Status**: IMPLEMENTATION-READY

---

## Core Contract

1. **Tracing is optional** and must never trigger fail-fast shutdown
2. **Shutdown is bounded best-effort** (not deterministic if actor is unhealthy)
3. **Flush is throttled during runtime**, forced on shutdown
4. **LoRA labeling requires separate adapter update** (scheduler API ready, call sites not wired)

---

## Refactoring Scope

**All helper extractions are IN-SCOPE for this implementation:**
- `_safe_trace()` / `_safe_trace_get()` - Split guarded helper
- `_safe_final_flush()` - Guarded shutdown flush
- `_get_or_create_gpu_track()` - Track creation logic
- `_build_trace_label()` - Label building, testable
- `_end_traces_for_gpu_ids()` - Reusable release helper
- `_trace_cycle_marker()` / `_maybe_flush_trace()` - Cycle helpers

---

## Executive Summary

Revised plan addressing all critical blockers and behavioral regression risks identified in feedback review.

### Key Changes from v1 → v10

| Issue | Severity | Fix |
|-------|----------|-----|
| Priority enum mismatch | P0 | Correct mapping to actual enum values |
| tg4perfetto import surface | P0 | Conditional import with fallback |
| **Missing class-level field decls** | **P0** | **Add `field(init=False)` declarations** |
| **Tracing errors trigger fail-fast** | **P0** | **`_safe_trace_call` returns `tuple[bool, Optional[T]]`** |
| **`create_track()` unguarded** | **P0** | **Guard via `_safe_trace_call`, check `track is None`** |
| Shutdown guarantee overstated | P0 | Clarify: bounded best-effort, not deterministic |
| Shutdown lifecycle | P1 | Add orchestrator integration with timeout |
| Missing release_and_request_gpus | P1 | Add tracing WITHOUT changing behavior |
| Exception handling policy | P1 | Fail-fast for init, degrade gracefully for runtime |
| Output format incorrect | P1 | Use `.perfetto-trace` extension |
| instant() API correctness | P1 | Verified: `Group.instant()` accepts `kwargs` |
| **_shutdown_tracing leaves active objects** | **P1** | **Clear all tracing state before flush** |
| **Flush policy inconsistent** | **P1** | **Throttled runtime (1s) + forced shutdown** |
| **Context stored on failed open** | **P1** | **Only store context when `open()` succeeds** |
| **Flush timestamp updated on failure** | **P1** | **Only update timestamp on successful flush** |
| **Shutdown flush not guarded** | **P2** | **Add `_safe_final_flush()` for shutdown** |
| **LoRA labeling only half wired** | **P1** | **Document: adapter sites need update** |
| **Test specs incorrect** | **P1** | **Fixed: proper assertions for context checks** |
| **Missing v9 fix tests** | **P2** | **Added flush retry and track failure tests** |
| I/O claim accuracy | P2 | Corrected: auto-flush can fire under lock |
| Flush under lock | P2 | Move flush outside lock |
| `_scheduler_group` None guard | P2 | Add explicit check |
| Configuration path | P2 | Both env vars read by scheduler (consistent) |
| `__del__` safety net overstated | P2 | Weaken: do not rely on GC for correctness |
| 2-second blocking shutdown | P2 | Reduced to 0.5s timeout |
| Redundant parse_cluster_id | low | Reuse existing pipeline_id variable |
| Typing `callable` vs `Callable` | P2 | Use `Callable[..., T]` with TypeVar |
| Duplicate Known Limitations | P2 | Removed duplicate section |
| Code duplication | refactor | **ALL helper extractions in-scope** |

### Already Completed (Do NOT Re-implement)

- **Phase 1**: `lora_name` fields already exist in `types.py`
- **Phase 2.1**: Conditional tg4perfetto import already exists in `scheduler.py`

---

## Phase 1: Type Definitions ✅ COMPLETE

**Status**: Already implemented in `schedrl/scheduler/types.py`

### 1.1 PendingRequest (already has lora_name)

```python
# Already exists at line 84
@dataclass(slots=True)
class PendingRequest:
    request: Request
    event: asyncio.Event
    global_step: Optional[int] = None
    lora_name: Optional[str] = None  # Already present
    result: List[int] = field(default_factory=list)
    error: Optional[str] = None
```

### 1.2 ClusterAllocation (already has lora_name)

```python
# Already exists at line 23
@dataclass(slots=True)
class ClusterAllocation:
    """Active GPU allocation for a cluster_id (format: '{pipeline_id}_{cluster_name}')."""

    cluster_id: str
    gpu_ids: List[int]
    priority: Priority
    active_dp_ranks: Set[int] = field(default_factory=set)
    dp_rank_to_gpus: Dict[int, List[int]] = field(default_factory=dict)
    global_step: Optional[int] = None
    timestamp: Optional[float] = None
    lora_name: Optional[str] = None  # Already present
```

---

## Phase 2: Scheduler Imports and State

### 2.1 Add imports ✅ COMPLETE

**Status**: Already implemented in `schedrl/scheduler/scheduler.py` (lines 40-46)

File: `schedrl/scheduler/scheduler.py`

```python
import atexit
import logging
import os
import time

# Conditional import for GPU timeline tracing (tg4perfetto may not be installed)
try:
    from tg4perfetto import TraceGenerator
    _TG4PERFETTO_AVAILABLE = True
except ImportError:
    TraceGenerator = None  # type: ignore[misc,assignment]
    _TG4PERFETTO_AVAILABLE = False
```

**Rationale**: 
- `Group` and `NormalTrack` are NOT imported (returned by factory methods)
- Conditional import prevents scheduler crash when tg4perfetto unavailable

### 2.2 Add Priority short mapping (CORRECTED)

Place after imports, before class definitions:

```python
# Short names for GPU trace labels - matches actual Priority enum values
_PRIORITY_SHORT = {
    Priority.INITIALIZATION: "INIT",
    Priority.ACTOR_TRAINING: "ACT",
    Priority.CRITIC_TRAINING: "CRT",
    Priority.OLD_LOG_PROBS: "OLD",
    Priority.REF_LOG_PROBS: "REF",
    Priority.VALUE_COMPUTE: "VAL",
    Priority.GENERATION: "GEN",
}
```

### 2.3 Add trace context dataclass

Place before SchedulerImpl class:

```python
@dataclass(slots=True)
class _GPUAllocTraceContext:
    """Context for an active GPU allocation trace slice.
    
    Stored in _gpu_contexts dict keyed by gpu_id.
    Used for debugging and future features (e.g., duration tracking).
    
    NOTE: Currently stored but not read - reserved for:
    - Duration calculation in _end_gpu_trace
    - Validation of end/start matching
    - Label updates mid-slice (if needed)
    """
    pipeline_id: str
    cluster_id: str
    priority: Priority
    alloc_type: str  # "initial" | "proactive"
    dp_ranks: Optional[List[int]] = None  # Generation clusters only
    lora_name: Optional[str] = None  # Training clusters only
```

### 2.4 Add class-level field declarations (CRITICAL for slots=True)

**IMPORTANT**: `SchedulerImpl` is a `@dataclass(slots=True)`. Python generates `__slots__` at class creation time from class-level annotations. Assigning new attributes in `__post_init__` without class-level declarations will raise `AttributeError` at runtime.

Add these field declarations to the `SchedulerImpl` class body (following the pattern at lines 135-145):

```python
@dataclass(slots=True)
class SchedulerImpl:
    _state: SchedulerState = field(init=False)
    _lock: asyncio.Lock = field(init=False)
    _wakeup_event: asyncio.Event = field(init=False)
    _topology_ready: asyncio.Event = field(init=False)
    _loop_task: Optional[asyncio.Task] = field(init=False)
    _resource_manager: Any = field(init=False)
    _cycle_counter: int = field(init=False)
    _request_seq: int = field(init=False)
    _num_gpus: Optional[int] = field(init=False)
    _required_gpus_per_node: Optional[int] = field(init=False)
    _adapter_handle_cache: Dict[str, Tuple[str, Any]] = field(init=False)
    # NEW: Tracing state fields (MUST be declared at class level for slots=True)
    _enable_gpu_tracing: bool = field(init=False, default=False)
    _trace_gen: Optional[Any] = field(init=False, default=None)  # TraceGenerator when enabled
    _trace_file_path: Optional[str] = field(init=False, default=None)
    _scheduler_group: Optional[Any] = field(init=False, default=None)  # Group when enabled
    _gpu_tracks: Dict[int, Any] = field(init=False, default_factory=dict)  # gpu_id -> NormalTrack
    _gpu_contexts: Dict[int, _GPUAllocTraceContext] = field(init=False, default_factory=dict)
    # NEW: Throttled flush state
    _trace_last_flush_ns: int = field(init=False, default=0)
    _trace_flush_interval_ns: int = field(init=False, default=1_000_000_000)  # 1 second
    # NEW: Shutdown guard for idempotency
    _trace_shutdown_started: bool = field(init=False, default=False)
```

### 2.5 Initialize state in __post_init__ (values only, no new attrs)

```python
def __post_init__(self):
    self._state = SchedulerState()
    self._lock = asyncio.Lock()
    self._wakeup_event = asyncio.Event()
    self._topology_ready = asyncio.Event()
    self._loop_task: Optional[asyncio.Task] = None
    self._resource_manager = None
    self._cycle_counter = 0
    self._request_seq = 0
    self._num_gpus: Optional[int] = None
    self._required_gpus_per_node: Optional[int] = None
    self._adapter_handle_cache = {}
    # Tracing state initialized via default/default_factory in field declarations
```

---

## Phase 3: Core Tracing Methods

### 3.0 Guarded Tracing Helper (CRITICAL - Prevents Fail-Fast)

**Problem**: Any uncaught exception in scheduler cycle triggers fail-fast shutdown. Tracing errors must NOT crash the scheduler.

**Solution**: All tracing calls go through this guarded helper that disables tracing on first unexpected error:

```python
from typing import Callable, TypeVar

T = TypeVar("T")

def _safe_trace_call(self, func: Callable[..., T], *args, **kwargs) -> tuple[bool, Optional[T]]:
    """Execute tracing call with fail-safe guard.
    
    Returns:
        tuple[bool, Optional[T]]: (success, result) where success indicates if call completed.
        
    On first unexpected error, disables all tracing to prevent scheduler crash.
    I/O errors are logged at debug level and don't disable tracing.
    """
    if not self._enable_gpu_tracing:
        return False, None
    
    try:
        return True, func(*args, **kwargs)
    except (IOError, OSError) as e:
        # I/O errors are expected - don't disable, don't spam logs
        logging.getLogger(__name__).debug(f"Trace I/O error: {e}")
        return False, None
    except Exception as e:
        # Unexpected errors - disable tracing immediately to prevent crash
        logging.getLogger(__name__).warning(
            f"Tracing disabled due to unexpected error: {e}"
        )
        self._enable_gpu_tracing = False
        return False, None
```

**Usage**: Wrap ALL `instant()`, `open()`, `close()`, `flush()`, `create_track()` calls:
```python
# For fire-and-forget calls (instant, close, flush):
self._safe_trace_call(
    self._scheduler_group.instant,
    time.time_ns(),
    f"C{self._cycle_counter} Start",
    kwargs={"idle_gpus": len(self._state.idle_gpus), "active": len(self._state.active_allocations)}
)

# For calls that need return values (create_track, open):
ok, track = self._safe_trace_call(
    self._scheduler_group.create_track,
    f"GPU{gpu_id}_{node_id}_{local_id}"
)
if not ok:
    return  # Early return on failure
```

**NOTE**: ALL **runtime** trace calls use `_safe_trace_call()`. Shutdown uses dedicated `_safe_final_flush()` below.

### 3.0.1 Final Flush Helper (for shutdown)

```python
def _safe_final_flush(self) -> None:
    """Guarded flush for shutdown - not dependent on _enable_gpu_tracing.
    
    Called only from _shutdown_tracing during explicit shutdown.
    Safe to call even when tracing was already disabled.
    """
    if self._trace_gen is None:
        return
    try:
        self._trace_gen.flush()
    except Exception as e:
        logging.getLogger(__name__).debug(f"Final trace flush failed: {e}")
```

### 3.1 Initialization method

Called from `initialize()` when tracing is enabled:

```python
def _init_tracing(self, trace_output_dir: str | None) -> None:
    """Initialize trace generator.
    
    Called once from initialize() when enable_gpu_tracing=True.
    
    Fail-fast policy:
    - If tg4perfetto not installed when tracing enabled: log warning, disable tracing
    - If trace file creation fails: log warning, disable tracing
    - Runtime I/O errors during tracing: degrade gracefully (don't crash scheduler)
    
    NOTE: atexit handler is a best-effort fallback for non-Ray scenarios only.
    Ray terminates workers with SIGTERM/SIGKILL, which bypasses Python's atexit.
    The reliable path is explicit shutdown() via orchestrator integration.
    """
    # Guard: tg4perfetto must be available
    if not _TG4PERFETTO_AVAILABLE:
        logging.getLogger(__name__).warning(
            "GPU tracing requested but tg4perfetto not installed; tracing disabled"
        )
        self._enable_gpu_tracing = False
        return
    
    # Use correct extension: perfetto-trace (protobuf binary, NOT JSON)
    ts = time.strftime("%Y%m%d_%H%M%S")
    self._trace_file_path = os.path.join(
        trace_output_dir or os.getcwd(),
        f"schedrl_gpu_timeline_{ts}.perfetto-trace"
    )
    
    try:
        self._trace_gen = TraceGenerator(self._trace_file_path)
        self._scheduler_group = self._trace_gen.create_group("SCHEDULER")
        # Best-effort fallback for non-Ray scenarios; unreliable in Ray deployments
        atexit.register(self._shutdown_tracing)
        logging.getLogger(__name__).info(f"GPU tracing enabled: {self._trace_file_path}")
    except (IOError, OSError) as e:
        # I/O errors during init are expected - disable tracing gracefully
        logging.getLogger(__name__).warning(f"GPU tracing disabled (cannot create trace file): {e}")
        self._enable_gpu_tracing = False
        self._trace_gen = None
        self._scheduler_group = None
    except Exception as e:
        # Unexpected init errors - disable tracing but don't crash scheduler
        logging.getLogger(__name__).error(f"GPU tracing disabled (unexpected init error): {e}")
        self._enable_gpu_tracing = False
        self._trace_gen = None
        self._scheduler_group = None
```

### 3.2 Shutdown method (Bounded Best-Effort with Lock Synchronization)

**P0 Fix**: Shutdown must not race with ongoing tracing writes.

**P1 Fix**: Clear all tracing state before flush to prevent orphaned references.

**IMPORTANT**: This is **bounded best-effort**, not deterministic. If the scheduler actor is unhealthy, the orchestrator timeout (0.5s) may fire before shutdown completes.

```python
def _shutdown_tracing(self) -> None:
    """Idempotent shutdown. Safe to call multiple times.
    
    Called by:
    - atexit handler on normal process exit (unreliable in Ray - SIGTERM/SIGKILL bypass atexit)
    - Explicit shutdown() method from orchestrator (reliable path)
    
    Order of operations:
    1. Check idempotency guard
    2. Disable tracing flag (stops new trace calls)
    3. Clear all trace state (prevent orphaned references)
    4. Final flush (write remaining data)
    """
    if self._trace_shutdown_started:
        return
    self._trace_shutdown_started = True
    
    if self._trace_gen is None:
        return
    
    # Step 1: Disable tracing to stop new calls
    self._enable_gpu_tracing = False
    
    # Step 2: Clear all trace state before flush
    # NormalTrack holds parent generator refs - must clear to prevent writes after flush
    self._gpu_tracks.clear()
    self._gpu_contexts.clear()
    self._scheduler_group = None
    
    # Step 3: Final flush via guarded helper
    self._safe_final_flush()
    
    self._trace_gen = None
    # Keep _trace_file_path for post-shutdown assertions
    # Unregister atexit to prevent double-call
    try:
        atexit.unregister(self._shutdown_tracing)
    except Exception:
        pass

async def shutdown(self) -> None:
    """Explicit shutdown - call from orchestrator for clean termination.
    
    Acquires scheduler lock to ensure no concurrent tracing writes.
    Note: This is bounded best-effort - orchestrator uses 0.5s timeout.
    """
    async with self._lock:
        self._shutdown_tracing()
```

**WARNING**: Do NOT rely on `TraceGenerator.__del__` for correctness. While it does call `flush()` + `file.close()`, Python destructor timing is unreliable in Ray kill paths and interpreter teardown. Use explicit `shutdown()` for trace finalization.

### 3.3 Start trace method (with topology guard)

Handles track creation, label building, and trace slice opening:

**P0 Fix**: ALL **runtime** trace operations go through `_safe_trace_call()` - no inline try/except. Shutdown uses `_safe_final_flush()`.

**P1 Fix**: Only write `_gpu_contexts` when `open()` succeeds to prevent stale state.

**I/O Behavior Note**: `track.open()` and `track.close()` call `_flush_if_necessary()` which checks `len(self.trace.packet) > flush_threshold` (default 10000). This means:
- **Normal allocations**: In-memory buffer only, minimal latency
- **Auto-flush**: When buffer exceeds 10000 packets, disk I/O happens immediately inside the lock on that specific call
- **Typical frequency**: At ~50-100 trace events per cycle, auto-flush triggers every ~100-200 cycles

**Performance impact**: Auto-flush under lock is bounded (every ~100-200 cycles), not per-allocation. This is acceptable for a debugging feature.

```python
def _start_gpu_trace(
    self,
    gpu_id: int,
    cluster_id: str,
    pipeline_id: str,
    priority: Priority,
    alloc_type: str,
    dp_ranks: Optional[List[int]] = None,
    lora_name: Optional[str] = None,
) -> None:
    """Start a trace slice for GPU allocation.
    
    Creates track if needed, builds label, opens slice, stores context.
    Silently returns if tracing disabled or on error (best-effort).
    
    Label format:
    - Training with LoRA:    [ACT] pipeline-1_actor_train | job:pipeline-1 | lora:adapter-0 | initial | C5
    - Training without LoRA: [ACT] pipeline-1_actor_train | job:pipeline-1 | initial | C5
    - Generation:            [GEN] pipeline-1_actor_infer | job:pipeline-1 | initial | C10 | DP:[0,1]
    """
    # Guard: tracing must be enabled
    if not self._enable_gpu_tracing:
        return
    
    # Guard: trace generator and group must be initialized
    if self._trace_gen is None or self._scheduler_group is None:
        return
    
    # Guard: topology must be initialized
    if self._required_gpus_per_node is None:
        return
    
    # Create track if needed (lazy) - GUARDED via _safe_trace_call
    if gpu_id not in self._gpu_tracks:
        node_id = gpu_id // int(self._required_gpus_per_node)
        local_id = gpu_id % int(self._required_gpus_per_node)
        ok, track = self._safe_trace_call(
            self._scheduler_group.create_track,
            f"GPU{gpu_id}_{node_id}_{local_id}"
        )
        if not ok or track is None:  # Defensive: check both success and value
            return
        self._gpu_tracks[gpu_id] = track
    
    # Build label (inlined)
    p = _PRIORITY_SHORT.get(priority, priority.name[:3])
    parts = [f"[{p}]", cluster_id, f"job:{pipeline_id}"]
    
    # LoRA only for non-generation clusters (with sanitization)
    if lora_name and priority != Priority.GENERATION:
        safe_lora = lora_name.replace("|", "_").replace(" ", "_")[:64]
        parts.append(f"lora:{safe_lora}")
    
    parts.extend([alloc_type, f"C{self._cycle_counter}"])
    label = " | ".join(parts)
    
    # DP only for generation clusters
    if dp_ranks and priority == Priority.GENERATION:
        label += f" | DP:{dp_ranks}"
    
    # Open slice via _safe_trace_call - ONLY store context on success
    ok, _ = self._safe_trace_call(self._gpu_tracks[gpu_id].open, time.time_ns(), label)
    if not ok:
        return  # Early return - no stale context stored
    
    self._gpu_contexts[gpu_id] = _GPUAllocTraceContext(
        pipeline_id=pipeline_id,
        cluster_id=cluster_id,
        priority=priority,
        alloc_type=alloc_type,
        dp_ranks=dp_ranks,
        lora_name=lora_name,
    )
```

### 3.4 End trace method

Closes the trace slice and removes context:

```python
def _end_gpu_trace(self, gpu_id: int) -> None:
    """End a trace slice for GPU release.
    
    Removes context and closes the slice on the track.
    Silently handles missing context (may happen in edge cases).
    """
    if not self._enable_gpu_tracing:
        return
    
    self._gpu_contexts.pop(gpu_id, None)
    track = self._gpu_tracks.get(gpu_id)
    if track:
        # Close via _safe_trace_call (unified error handling)
        self._safe_trace_call(track.close, time.time_ns())
```

---

## Phase 4: Integration Points

### 4.1 In initialize() - enable tracing

Add after topology initialization, before loop start:

```python
async def initialize(
    self,
    *,
    resource_manager: Any | None = None,
    enable_gpu_tracing: bool = False,
    trace_output_dir: str | None = None,
) -> None:
    # ... existing code up to topology init ...
    
    # Enable tracing if parameter or env var is set
    # NOTE: Both env vars are read here (in scheduler actor) for consistency
    env_tracing = os.environ.get("SCHEDRL_ENABLE_GPU_TRACING", "").lower() in ("1", "true")
    env_trace_dir = os.environ.get("SCHEDRL_TRACE_OUTPUT_DIR")
    self._enable_gpu_tracing = enable_gpu_tracing or env_tracing
    
    if self._enable_gpu_tracing:
        # Prefer explicit parameter, fall back to env var, then cwd
        self._init_tracing(trace_output_dir or env_trace_dir)
    
    # ... rest of existing init ...
```

### 4.2 In request_gpus() - add lora_name parameter

Add parameter and pass to PendingRequest:

```python
async def request_gpus(
    self,
    *,
    cluster_id: str,
    priority: Priority,
    global_step: Optional[int] = None,
    lora_name: Optional[str] = None,  # NEW
) -> List[int]:
    # ... existing validation code ...
    
    self._request_seq += 1
    pending = PendingRequest(
        request=Request(cluster_id=cluster_id, priority=priority, timestamp=float(self._request_seq)),
        event=event,
        global_step=global_step,
        lora_name=lora_name,  # NEW
    )
    
    # ... rest of method ...
```

### 4.3 In _apply_plan_and_signal() - initial allocation tracing

Add after ClusterAllocation creation, before `_signal_pending_request`:

**NOTE**: In the actual implementation, `pipeline_id` is already computed earlier in this method (at the line `pipeline_id, cluster_name = parse_cluster_id(op.cluster_id)`). Reuse that variable instead of re-computing.

```python
# After: self._state.active_allocations[op.cluster_id] = ClusterAllocation(...)
if self._enable_gpu_tracing:
    # Get lora_name from pending request (inlined lookup)
    lora_name = None
    for p in self._state.pending_bucket(priority):
        if p.request.cluster_id == op.cluster_id:
            lora_name = p.lora_name
            break
    
    # Store in allocation for proactive allocation lookup
    if lora_name:
        self._state.active_allocations[op.cluster_id].lora_name = lora_name
    
    # NOTE: pipeline_id already computed earlier in this method - reuse it
    # pipeline_id, _ = parse_cluster_id(op.cluster_id)  # ← REMOVE: redundant
    
    for gpu_id in sorted(op.gpus_to_allocate):
        # Extract DP rank for generation clusters
        dp_ranks = None
        if is_generation_cluster(op.cluster_id):
            for dp_rank, bundle in dp_rank_to_gpus.items():
                if gpu_id in bundle:
                    dp_ranks = [dp_rank]
                    break
        
        self._start_gpu_trace(
            gpu_id, op.cluster_id, pipeline_id, priority,
            "initial", dp_ranks, lora_name
        )
```

### 4.4 In shrink loop - end traces for freed GPUs

Add after GPUs returned to idle pool:

```python
for op in plan.sched_guided_shrink_ops:
    # ... existing shrink logic ...
    
    alloc.gpu_ids = [g for g in alloc.gpu_ids if g not in bundle]
    self._state.idle_gpus |= bundle
    
    # NEW: End traces for shrunk GPUs
    if self._enable_gpu_tracing:
        for gpu_id in bundle:
            self._end_gpu_trace(gpu_id)
```

### 4.5 In cluster removal - end traces

Add when removing cluster from active allocations:

```python
for cluster_id in plan.clusters_to_remove:
    alloc = self._state.active_allocations.pop(cluster_id, None)
    if alloc is not None:
        # NEW: End traces for removed cluster
        if self._enable_gpu_tracing:
            for gpu_id in alloc.gpu_ids:
                self._end_gpu_trace(gpu_id)
        
        self._state.idle_gpus |= set(alloc.gpu_ids)
```

### 4.6 In proactive allocation - start traces

Add after updating allocation with new GPUs:

```python
# After: alloc.gpu_ids = sorted(set(alloc.gpu_ids) | gpu_set)
if self._enable_gpu_tracing:
    pipeline_id, _ = parse_cluster_id(op.cluster_id)
    
    for dp_rank in sorted(op.dp_ranks_to_add):
        bundle = alloc.dp_rank_to_gpus.get(dp_rank, [])
        for gpu_id in bundle:
            self._start_gpu_trace(
                gpu_id, op.cluster_id, pipeline_id,
                Priority.GENERATION, "proactive", [dp_rank]
            )
```

### 4.7 In release_gpus() - end traces

Add after popping allocation:

```python
async def release_gpus(self, *, cluster_id: str, global_step: Optional[int] = None) -> None:
    await self._topology_ready.wait()
    async with self._lock:
        alloc = self._state.active_allocations.pop(cluster_id, None)
        if alloc is None:
            raise RuntimeError(f"cluster_id {cluster_id!r} not found in active_allocations")
        
        # NEW: End traces for released GPUs
        if self._enable_gpu_tracing:
            for gpu_id in alloc.gpu_ids:
                self._end_gpu_trace(gpu_id)
        
        self._state.idle_gpus |= set(alloc.gpu_ids)
        self._wakeup_event.set()
```

### 4.8 In release_and_request_gpus() - add tracing and lora_name

**IMPORTANT**: Preserve original code order - only ADD tracing, don't reorder operations.

Original code order:
1. Check existing allocation for request_cluster_id FIRST
2. If existing allocation valid, return immediately
3. Then handle release_cluster_id
4. Then create pending request

Add `request_lora_name` parameter and tracing for release phase:

```python
async def release_and_request_gpus(
    self,
    *,
    release_cluster_id: Optional[str],
    release_global_step: Optional[int],
    request_cluster_id: str,
    request_priority: Priority,
    request_global_step: Optional[int] = None,
    request_lora_name: Optional[str] = None,  # NEW
) -> List[int]:
    await self._topology_ready.wait()
    event = asyncio.Event()
    pending: PendingRequest | None = None
    async with self._lock:
        pipeline_id, _ = parse_cluster_id(request_cluster_id)
        info = self._state.pipeline_registry.get(pipeline_id)
        if info is None:
            raise RuntimeError(f"pipeline_id {pipeline_id!r} not registered")
        if not bool(info.get("admitted", False)):
            raise RuntimeError(f"pipeline_id {pipeline_id!r} not admitted; call orchestrator.admit_pipeline first")
        
        # CHECK EXISTING ALLOCATION FIRST (preserve original order)
        existing = self._state.active_allocations.get(request_cluster_id)
        if existing is not None:
            if existing.priority != request_priority:
                raise RuntimeError(
                    f"cluster_id {request_cluster_id!r} already allocated with priority={existing.priority}, requested priority={request_priority}"
                )
            if request_priority == Priority.GENERATION and not existing.active_dp_ranks:
                pass
            elif request_priority != Priority.GENERATION and not existing.gpu_ids:
                pass
            else:
                return list(existing.gpu_ids)
        
        # THEN HANDLE RELEASE (preserve original order)
        if release_cluster_id is not None:
            alloc = self._state.active_allocations.pop(release_cluster_id, None)
            if alloc is None:
                raise RuntimeError(f"release_cluster_id {release_cluster_id!r} not found")
            
            # NEW: End traces for released GPUs
            if self._enable_gpu_tracing:
                for gpu_id in alloc.gpu_ids:
                    self._end_gpu_trace(gpu_id)
            
            self._state.idle_gpus |= set(alloc.gpu_ids)
        
        # THEN CREATE PENDING REQUEST
        if self._has_any_pending_request_locked(cluster_id=request_cluster_id):
            raise RuntimeError(f"Duplicate pending request for cluster_id={request_cluster_id!r} is not supported")
        
        self._request_seq += 1
        pending = PendingRequest(
            request=Request(cluster_id=request_cluster_id, priority=request_priority, timestamp=float(self._request_seq)),
            event=event,
            global_step=request_global_step,
            lora_name=request_lora_name,  # NEW
        )
        self._state.pending_bucket(request_priority).append(pending)
        self._wakeup_event.set()
    
    await event.wait()
    if pending is None:
        raise RuntimeError("release_and_request_gpus internal error: pending request not created")
    if pending.error is not None:
        raise RuntimeError(pending.error)
    return list(pending.result)
```

### 4.9 In unregister_pipeline() - end all pipeline traces

Add at start of lock block:

```python
async def unregister_pipeline(self, *, pipeline_id: str) -> None:
    validate_pipeline_id(pipeline_id)
    async with self._lock:
        # NEW: End traces for all GPUs held by this pipeline
        if self._enable_gpu_tracing:
            for cid in list(self._state.active_allocations.keys()):
                if cid.startswith(f"{pipeline_id}_"):
                    alloc = self._state.active_allocations.get(cid)
                    if alloc:
                        for gpu_id in alloc.gpu_ids:
                            self._end_gpu_trace(gpu_id)
        
        # ... existing unregister code ...
```

### 4.10 In scheduling_cycle() - cycle markers and throttled flush

**P0 Fix**: Use guarded helper to prevent tracing errors from crashing scheduler.

**P1 Fix**: Throttled flush during runtime (every 1s) to avoid throughput regression, forced flush on shutdown.

**New fields required** (add to class-level declarations in Phase 2.4):
```python
_trace_last_flush_ns: int = field(init=False, default=0)
_trace_flush_interval_ns: int = field(init=False, default=1_000_000_000)  # 1 second
_trace_shutdown_started: bool = field(init=False, default=False)
```

Add cycle start marker after counter increment:

```python
async with self._lock:
    self._cycle_counter += 1
    
    # NEW: Cycle start marker (guarded)
    if self._enable_gpu_tracing and self._scheduler_group:
        self._safe_trace_call(
            self._scheduler_group.instant,
            time.time_ns(),
            f"C{self._cycle_counter} Start",
            kwargs={
                "idle_gpus": len(self._state.idle_gpus),
                "active": len(self._state.active_allocations),
            }
        )
    
    # ... scheduling logic ...
```

Add cycle end marker and throttled flush OUTSIDE lock:

```python
# Phase 6: commit (Phase 2 simulation: state-only).
async with self._lock:
    self._apply_plan_and_signal(plan)

# NEW: Cycle end marker + throttled flush (OUTSIDE lock to avoid I/O latency)
if self._enable_gpu_tracing and self._scheduler_group:
    self._safe_trace_call(
        self._scheduler_group.instant,
        time.time_ns(),
        f"C{self._cycle_counter} End"
    )
    
    # Throttled flush: at most once per _trace_flush_interval_ns (default 1s)
    if self._trace_gen:
        now_ns = time.time_ns()
        if now_ns - self._trace_last_flush_ns >= self._trace_flush_interval_ns:
            # Only update timestamp on successful flush to allow retry on failure
            ok, _ = self._safe_trace_call(self._trace_gen.flush)
            if ok:
                self._trace_last_flush_ns = now_ns
```

**Rationale**: 
- Throttled flush during runtime balances data persistence vs throughput
- Forced flush on shutdown (see Phase 3.2) ensures final data is captured
- If scheduler crashes, auto-flush (every 10000 packets) provides fallback

---

## Phase 5: Orchestrator Integration

### 5.1 Update orchestrator shutdown

File: `schedrl/orchestrator/orchestrator.py`

Add explicit scheduler shutdown call with short timeout:

```python
def shutdown(self, force: bool = True, reason: Optional[str] = None, source: Optional[str] = None) -> None:
    """..."""
    if self._shutdown_started:
        return
    self._shutdown_started = True
    if not force:
        raise RuntimeError("shutdown(force=False) is not supported in ENG-123 Phase 1")
    
    # NEW: Explicit scheduler shutdown for trace finalization (with short timeout)
    # Use ray.wait() with timeout to avoid blocking indefinitely in fail-fast scenarios
    # 0.5s is enough for flush() under normal conditions, but won't stall on dead actors
    try:
        shutdown_ref = self._scheduler.shutdown.remote()
        ray.wait([shutdown_ref], timeout=0.5)  # 0.5-second timeout
    except Exception:
        pass  # Best-effort, don't stall shutdown
    
    _force_stop_cluster_workers_first()
    # ... rest of shutdown ...
```

### 5.2 Add tracing configuration path

**Environment Variables:**

Both env vars are read directly by the scheduler actor (inside Ray), so they should be set on the worker node(s) where the scheduler runs:

```bash
# Enable GPU tracing
export SCHEDRL_ENABLE_GPU_TRACING=1

# Optional: Set output directory (defaults to cwd)
export SCHEDRL_TRACE_OUTPUT_DIR=/path/to/traces
```

**Orchestrator Integration:**

No changes needed to orchestrator for env var support - scheduler reads env vars directly.

For programmatic control (override env vars), orchestrator can pass parameters:

```python
ray.get(scheduler.initialize.remote(
    resource_manager=resource_manager,
    enable_gpu_tracing=True,  # Override env var
    trace_output_dir="/custom/path",  # Override env var
))
```

**User-facing configuration:**

Users can enable tracing via:
1. Environment variables on scheduler worker node (recommended for production)
2. Direct orchestrator parameter calls (for programmatic control)

---

## Summary

| Metric | Value |
|--------|-------|
| Files modified | 3 (types.py, scheduler.py, orchestrator.py) |
| New dataclasses | 1 (_GPUAllocTraceContext) |
| New methods | 6 (_init_tracing, _shutdown_tracing, _start_gpu_trace, _end_gpu_trace, shutdown, _safe_trace_call) |
| Integration points | 10 |
| Lines of new code | ~250 |
| Functionality | Full (LoRA labels require adapter update) |
| Fail-safe | Yes (unified guarded helper, explicit guards, timeout) |

### Core Contract

1. **Tracing is optional** and must never trigger fail-fast shutdown
2. **Shutdown is bounded best-effort** (not deterministic if actor is unhealthy)
3. **Flush is throttled during runtime** (every 1s), forced on shutdown
4. **LoRA labeling requires separate adapter update** (scheduler API ready, call sites not wired)

### Behavioral Guarantees

1. **No behavior change**: Existing code paths preserved, only tracing added
2. **Tracing errors don't crash scheduler**: `_safe_trace_call()` disables tracing on first error
3. **Bounded best-effort shutdown**: Lock synchronization + 0.5s timeout
4. **Graceful degradation**: Tracing disabled on errors, scheduler continues
5. **Unified error handling**: ALL trace calls go through `_safe_trace_call()`
6. **slots=True compatible**: Class-level field declarations prevent AttributeError
7. **Throughput maintained**: Throttled flush (every 1s) instead of per-cycle

### I/O Performance Characteristics

- **Normal operation**: In-memory protobuf buffer only (no disk I/O per allocation)
- **Buffered writes**: Disk I/O only when buffer exceeds 10000 packets
- **Throttled flush**: Every 1 second during runtime (configurable)
- **Forced flush**: On explicit shutdown
- **Auto-flush**: When buffer fills (~100-200 cycles), happens under lock (bounded)

---

## Testing Requirements

The following tests should be implemented to verify correctness:

### 1. Trace Close Correctness Test

```python
async def test_trace_close_all_paths():
    """Verify traces are closed correctly across all deallocation paths."""
    scheduler = SchedulerImpl()
    await scheduler.initialize(resource_manager=rm, enable_gpu_tracing=True)
    
    # Allocate GPUs
    gpus = await scheduler.request_gpus(cluster_id="p1_actor_train", priority=Priority.ACTOR_TRAINING)
    assert len(scheduler._gpu_contexts) == len(gpus)  # All GPUs have open traces
    
    # Test 1: release_gpus
    await scheduler.release_gpus(cluster_id="p1_actor_train")
    assert len(scheduler._gpu_contexts) == 0  # All traces closed
    
    # Test 2: release_and_request_gpus
    gpus1 = await scheduler.request_gpus(cluster_id="p1_actor_train", priority=Priority.ACTOR_TRAINING)
    gpus2 = await scheduler.release_and_request_gpus(
        release_cluster_id="p1_actor_train",
        request_cluster_id="p1_critic",
        request_priority=Priority.CRITIC_TRAINING
    )
    assert len(scheduler._gpu_contexts) == len(gpus2)  # Old traces closed, new traces open
    
    # Test 3: unregister_pipeline
    await scheduler.unregister_pipeline(pipeline_id="p1")
    assert len(scheduler._gpu_contexts) == 0  # All traces closed
```

### 2. tg4perfetto Unavailable Fallback Test

```python
def test_tracing_disabled_when_tg4perfetto_unavailable(monkeypatch):
    """Verify graceful degradation when tg4perfetto is not installed."""
    monkeypatch.setattr("schedrl.scheduler.scheduler._TG4PERFETTO_AVAILABLE", False)
    
    scheduler = SchedulerImpl()
    # Force tracing enabled BEFORE init to test disable-on-request behavior
    scheduler._enable_gpu_tracing = True
    # Should not crash, should log warning and disable tracing
    scheduler._init_tracing(None)
    
    assert scheduler._enable_gpu_tracing is False
    assert scheduler._trace_gen is None
```

### 3. Same-Cycle Shrink+Reallocate Test

```python
async def test_shrink_reallocate_same_cycle():
    """Verify close/open ordering on same GPU track."""
    scheduler = SchedulerImpl()
    await scheduler.initialize(resource_manager=rm, enable_gpu_tracing=True)
    
    # Allocate GPU 0 to pipeline A
    await scheduler.request_gpus(cluster_id="p1_actor_infer", priority=Priority.GENERATION)
    
    # Trigger shrink + reallocate to pipeline B in same cycle
    # This tests that GPU 0 trace is closed before reopened
    # ...
    
    # Verify: GPU 0 should have exactly 2 trace slices (not interleaved)
    # Check trace file for correct close/open ordering
```

### 4. Zero-GPU Wake-Only Test

```python
async def test_zero_gpu_wake_only_no_trace():
    """Verify zero-GPU wake-only requests do not open traces."""
    scheduler = SchedulerImpl()
    await scheduler.initialize(resource_manager=rm, enable_gpu_tracing=True)
    
    # Allocate all GPUs to a generation cluster
    # Then request another cluster that gets wake-only (empty) allocation
    # ...
    
    # Verify: No traces opened for wake-only allocation
    assert len(scheduler._gpu_contexts) == 0
```

### 5. Shutdown Flush Test

```python
async def test_shutdown_flushes_trace():
    """Verify trace is flushed on shutdown."""
    scheduler = SchedulerImpl()
    await scheduler.initialize(resource_manager=rm, enable_gpu_tracing=True)
    
    trace_path = scheduler._trace_file_path
    await scheduler.request_gpus(cluster_id="p1_actor_train", priority=Priority.ACTOR_TRAINING)
    
    # Shutdown
    await scheduler.shutdown()
    
    # Verify trace file exists and has content
    assert os.path.exists(trace_path)
    assert os.path.getsize(trace_path) > 0
```

### 6. Flush Retry Test (v9 fix)

```python
async def test_flush_retry_when_previous_flush_fails(monkeypatch):
    """Verify flush retry when previous flush fails - timestamp unchanged."""
    scheduler = SchedulerImpl()
    await scheduler.initialize(resource_manager=rm, enable_gpu_tracing=True)
    
    initial_ts = scheduler._trace_last_flush_ns
    
    # Simulate flush failure
    def failing_flush():
        raise IOError("disk full")
    monkeypatch.setattr(scheduler._trace_gen, "flush", failing_flush)
    
    # Trigger throttled flush
    scheduler._trace_last_flush_ns = 0  # Force interval check
    scheduler._maybe_flush_trace()
    
    # Verify: timestamp unchanged on failure (allows retry)
    assert scheduler._trace_last_flush_ns == 0
    
    # Now make flush succeed
    monkeypatch.setattr(scheduler._trace_gen, "flush", lambda: None)
    scheduler._maybe_flush_trace()
    
    # Verify: timestamp updated on success
    assert scheduler._trace_last_flush_ns > 0
```

### 7. Track Creation Failure Test (v9 fix)

```python
async def test_start_trace_create_track_failure_no_context(monkeypatch):
    """Verify create_track failure leaves no context and doesn't crash."""
    scheduler = SchedulerImpl()
    await scheduler.initialize(resource_manager=rm, enable_gpu_tracing=True)
    
    # Force create_track to fail
    def failing_create_track(name):
        raise RuntimeError("track creation failed")
    monkeypatch.setattr(scheduler._scheduler_group, "create_track", failing_create_track)
    
    # Attempt to start trace - should not crash
    scheduler._start_gpu_trace(
        gpu_id=0,
        cluster_id="p1_actor_train",
        pipeline_id="p1",
        priority=Priority.ACTOR_TRAINING,
        alloc_type="initial",
    )
    
    # Verify: no context stored on failure
    assert len(scheduler._gpu_contexts) == 0
    # Verify: tracing still enabled (graceful degradation)
    assert scheduler._enable_gpu_tracing is True
```

---

## Known Limitations

### LoRA Labeling (P1 - Requires Adapter Updates)

The scheduler API supports `lora_name` parameter, but current adapter call sites do NOT pass it:

- `concurrent_pipeline.py#L525`: `request_gpus()` called without `lora_name`
- `concurrent_pipeline.py#L551`: `release_and_request_gpus()` called without `request_lora_name`

**Result**: Tracing works, but `lora:` labels will be empty until adapter sites are updated.

**Fix Required** (out of scope for scheduler implementation):
```python
# In concurrent_pipeline.py - pass lora_name when calling scheduler
gpus = await self._scheduler.request_gpus.remote(
    cluster_id=cluster_id,
    priority=priority,
    lora_name=self._lora_adapter_name,  # NEW: pass LoRA name
)
```

### atexit Unreliable in Ray

Ray terminates worker processes with `SIGTERM`/`SIGKILL`, which bypasses Python's `atexit` handlers.

**Fix**: Use explicit `orchestrator.shutdown()` for trace finalization.

### Do NOT Rely on `__del__` for Correctness

While `TraceGenerator.__del__` calls `flush()` + `file.close()`, Python destructor timing is unreliable in Ray kill paths and interpreter teardown.

**Fix**: Always call explicit `shutdown()` for guaranteed trace finalization.

---

## Helper Extractions (In-Scope)

The following helper extractions can reduce code duplication and improve maintainability. These are optional but recommended:

### 1. Split `_safe_trace_call` into Two Helpers

```python
def _safe_trace(self, func: Callable[..., Any], *args, **kwargs) -> bool:
    """Fire-and-forget trace call. Returns success status."""
    ok, _ = self._safe_trace_call(func, *args, **kwargs)
    return ok

def _safe_trace_get(self, func: Callable[..., T], *args, **kwargs) -> Optional[T]:
    """Trace call with return value. Returns result or None on failure."""
    _, result = self._safe_trace_call(func, *args, **kwargs)
    return result
```

### 2. Extract `_get_or_create_gpu_track`

```python
def _get_or_create_gpu_track(self, gpu_id: int) -> Optional[Any]:
    """Get existing track or create new one. Returns None on failure."""
    if gpu_id in self._gpu_tracks:
        return self._gpu_tracks[gpu_id]
    
    if self._required_gpus_per_node is None:
        return None
    
    node_id = gpu_id // int(self._required_gpus_per_node)
    local_id = gpu_id % int(self._required_gpus_per_node)
    track = self._safe_trace_get(
        self._scheduler_group.create_track,
        f"GPU{gpu_id}_{node_id}_{local_id}"
    )
    if track is not None:
        self._gpu_tracks[gpu_id] = track
    return track
```

### 3. Extract `_build_trace_label`

```python
def _build_trace_label(
    self,
    cluster_id: str,
    pipeline_id: str,
    priority: Priority,
    alloc_type: str,
    dp_ranks: Optional[List[int]] = None,
    lora_name: Optional[str] = None,
) -> str:
    """Build trace label string. Testable in isolation."""
    p = _PRIORITY_SHORT.get(priority, priority.name[:3])
    parts = [f"[{p}]", cluster_id, f"job:{pipeline_id}"]
    
    if lora_name and priority != Priority.GENERATION:
        safe_lora = lora_name.replace("|", "_").replace(" ", "_")[:64]
        parts.append(f"lora:{safe_lora}")
    
    parts.extend([alloc_type, f"C{self._cycle_counter}"])
    label = " | ".join(parts)
    
    if dp_ranks and priority == Priority.GENERATION:
        label += f" | DP:{dp_ranks}"
    
    return label
```

### 4. Extract `_end_traces_for_gpu_ids`

```python
def _end_traces_for_gpu_ids(self, gpu_ids: List[int]) -> None:
    """End trace slices for multiple GPUs. Reusable across release paths."""
    for gpu_id in gpu_ids:
        self._end_gpu_trace(gpu_id)
```

Usage in shrink, removal, release, unregister:
```python
# Before: loop in each location
for gpu_id in bundle:
    self._end_gpu_trace(gpu_id)

# After: single call
self._end_traces_for_gpu_ids(list(bundle))
```

### 5. Extract Cycle Marker and Flush Helpers

```python
def _trace_cycle_marker(self, name: str, payload: Optional[Dict[str, Any]] = None) -> None:
    """Record a cycle marker instant event."""
    if not self._enable_gpu_tracing or self._scheduler_group is None:
        return
    kwargs = {"kwargs": payload} if payload else {}
    self._safe_trace(self._scheduler_group.instant, time.time_ns(), name, **kwargs)

def _maybe_flush_trace(self) -> None:
    """Throttled flush - only flush if interval elapsed."""
    if not self._trace_gen:
        return
    now_ns = time.time_ns()
    if now_ns - self._trace_last_flush_ns >= self._trace_flush_interval_ns:
        ok, _ = self._safe_trace_call(self._trace_gen.flush)
        if ok:
            self._trace_last_flush_ns = now_ns
```

---

## Output Examples

Trace file: `schedrl_gpu_timeline_20260224_143052.perfetto-trace`

Labels in Perfetto UI:

```
[ACT] pipeline-1_actor_train | job:pipeline-1 | lora:adapter-0 | initial | C5
[ACT] pipeline-1_actor_train | job:pipeline-1 | initial | C5
[GEN] pipeline-1_actor_infer | job:pipeline-1 | initial | C10 | DP:[0, 1, 2]
[GEN] pipeline-1_actor_infer | job:pipeline-1 | proactive | C15 | DP:[3]
```
