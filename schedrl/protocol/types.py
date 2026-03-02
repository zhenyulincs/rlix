from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any, Dict, Optional

# Ray namespace and actor name protocol constants shared across schedrl modules.
SCHEDRL_NAMESPACE: str = "schedrl"
SCHEDULER_ACTOR_NAME: str = "schedrl:scheduler"
ORCHESTRATOR_ACTOR_NAME: str = "schedrl:orchestrator"
RESOURCE_MANAGER_ACTOR_NAME: str = "schedrl:resource_manager"
# Prefix for per-pipeline coordinator actors: full name = f"{COORDINATOR_ACTOR_NAME_PREFIX}{pipeline_id}"
COORDINATOR_ACTOR_NAME_PREFIX: str = "schedrl:coordinator:"
# Prefix for per-pipeline coordinator actors: full name = f"{PIPELINE_ACTOR_NAME_PREFIX}{pipeline_id}"
PIPELINE_ACTOR_NAME_PREFIX: str = "schedrl:pipeline:"
# Name for the ROLL-specific ResourceManager singleton actor (used when SCHEDRL_CONTROL_PLANE=schedrl)
ROLL_RESOURCE_MANAGER_ACTOR_NAME: str = "schedrl:roll_resource_manager"


@dataclass(frozen=True, slots=True)
class ActionResponse:
    success: bool
    error: Optional[str] = None


class Priority(enum.IntEnum):
    """7-tier priority system for GPU allocation (lower numeric value = higher priority)."""

    INITIALIZATION = 0
    ACTOR_TRAINING = 1
    CRITIC_TRAINING = 2
    OLD_LOG_PROBS = 3
    REF_LOG_PROBS = 4
    VALUE_COMPUTE = 5
    GENERATION = 6


@dataclass(frozen=True, slots=True)
class ProgressReport:
    pipeline_id: str
    queued_trajectories: int
    inflight_trajectories: int
    step_target_trajectories: int
    percent_completed: float = 0.0
    oldest_unfinished_creation_ts: Optional[float] = None
    active_base_version: int = 0
    fifo_timestamp: Optional[float] = None
    metrics: Optional[Dict[str, Any]] = None
