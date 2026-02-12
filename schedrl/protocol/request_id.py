from __future__ import annotations

from dataclasses import dataclass

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


@dataclass(frozen=True, slots=True)
class ParsedRequestId:
    pipeline_id: str
    traj_id: str
    turn_id: int
    attempt: int


def parse_request_id(request_id: str) -> tuple[str, str, int, int]:
    parsed = _parse_request_id_obj(request_id)
    return parsed.pipeline_id, parsed.traj_id, parsed.turn_id, parsed.attempt


def _parse_request_id_obj(request_id: str) -> ParsedRequestId:
    if not isinstance(request_id, str):
        raise ValueError(f"request_id must be str, got {type(request_id).__name__}")
    parts = request_id.split(REQUEST_ID_DELIMITER)
    if len(parts) != 4:
        raise ValueError(
            f"request_id must have 4 parts separated by {REQUEST_ID_DELIMITER!r}, got {len(parts)}: {request_id!r}"
        )
    pipeline_id, traj_id, turn_raw, attempt_raw = parts
    validate_pipeline_id(pipeline_id)
    _validate_traj_id(traj_id)
    try:
        turn_id = int(turn_raw)
    except ValueError as e:
        raise ValueError(f"request_id turn_id must be int, got {turn_raw!r}: {request_id!r}") from e
    try:
        attempt = int(attempt_raw)
    except ValueError as e:
        raise ValueError(f"request_id attempt must be int, got {attempt_raw!r}: {request_id!r}") from e
    if turn_id < 0:
        raise ValueError(f"request_id turn_id must be >= 0, got {turn_id}: {request_id!r}")
    if attempt < 0:
        raise ValueError(f"request_id attempt must be >= 0, got {attempt}: {request_id!r}")
    return ParsedRequestId(pipeline_id=pipeline_id, traj_id=traj_id, turn_id=turn_id, attempt=attempt)


def validate_request_id(request_id: str) -> None:
    _ = _parse_request_id_obj(request_id)

