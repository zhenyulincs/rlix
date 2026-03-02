from __future__ import annotations

REQUEST_ID_DELIMITER = ":"


def validate_pipeline_id(pipeline_id: str) -> None:
    if not isinstance(pipeline_id, str):
        raise ValueError(f"pipeline_id must be str, got {type(pipeline_id).__name__}")
    if pipeline_id == "":
        raise ValueError("pipeline_id must be non-empty")
    if REQUEST_ID_DELIMITER in pipeline_id:
        raise ValueError(f"pipeline_id must not contain {REQUEST_ID_DELIMITER!r}: {pipeline_id!r}")


def _validate_traj_id(traj_id: str) -> None:
    if not isinstance(traj_id, str):
        raise ValueError(f"traj_id must be str, got {type(traj_id).__name__}")
    if traj_id == "":
        raise ValueError("traj_id must be non-empty")
    if REQUEST_ID_DELIMITER in traj_id:
        raise ValueError(f"traj_id must not contain {REQUEST_ID_DELIMITER!r}: {traj_id!r}")


def build_request_id(pipeline_id: str, traj_id: str, turn_id: int, attempt: int) -> str:
    validate_pipeline_id(pipeline_id)
    _validate_traj_id(traj_id)
    if not isinstance(turn_id, int) or turn_id < 0:
        raise ValueError(f"turn_id must be int >= 0, got {turn_id!r}")
    if not isinstance(attempt, int) or attempt < 0:
        raise ValueError(f"attempt must be int >= 0, got {attempt!r}")
    return f"{pipeline_id}{REQUEST_ID_DELIMITER}{traj_id}{REQUEST_ID_DELIMITER}{turn_id}{REQUEST_ID_DELIMITER}{attempt}"
