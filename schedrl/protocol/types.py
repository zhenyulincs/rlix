from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class ModelMode(str, Enum):
    FULL_FT = "FULL_FT"
    MULTI_LORA = "MULTI_LORA"


@dataclass(frozen=True, slots=True)
class PipelineId:
    value: str


@dataclass(frozen=True, slots=True)
class ClusterId:
    value: str


@dataclass(frozen=True, slots=True)
class AdapterId:
    value: str


@dataclass(frozen=True, slots=True)
class ActionResponse:
    success: bool
    error: Optional[str] = None


@dataclass(frozen=True, slots=True)
class PlatformConfig:
    ray_device_key: str
    device_control_env_var: str


@dataclass(frozen=True, slots=True)
class ReleaseReport:
    dp_rank: int
    gpu_map: List[int]
    free_bytes_by_gpu: List[int]
    total_bytes_by_gpu: List[int]


@dataclass(frozen=True, slots=True)
class ReleaseAck:
    aborted: int
    remapped: int
    release_reports: List[ReleaseReport]


@dataclass(frozen=True, slots=True)
class ProgressReport:
    pipeline_id: str
    queued_trajectories: int
    inflight_trajectories: int
    step_target_trajectories: int
    fifo_timestamp: Optional[float] = None
    metrics: Optional[Dict[str, Any]] = None


@dataclass(frozen=True, slots=True)
class SchedRLTimeouts:
    register_timeout_secs: float = -1
    admit_timeout_secs: float = -1
    shrink_timeout_secs: float = -1
    expand_timeout_secs: float = -1
    abort_ack_timeout_secs: float = -1
    offload_timeout_secs: float = -1
    abort_timeout_secs: float = -1


@dataclass(frozen=True, slots=True)
class SchedRLConfig:
    fail_fast_on_restart: bool = True
    timeouts: SchedRLTimeouts = SchedRLTimeouts()


@dataclass(frozen=True, slots=True)
class RayNamespaceContract:
    pipeline_id_env_var: str = "PIPELINE_ID"
    roll_namespace_env_var: str = "ROLL_RAY_NAMESPACE"
