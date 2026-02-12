from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

from schedrl.protocol.request_id import validate_pipeline_id, validate_request_id


@dataclass(frozen=True, slots=True)
class RegisterValidationInput:
    pipeline_id: str
    total_gpus: int
    gpu_ids: List[int]


def validate_register_pipeline(inp: RegisterValidationInput) -> None:
    validate_pipeline_id(inp.pipeline_id)
    if not isinstance(inp.total_gpus, int) or inp.total_gpus <= 0:
        raise ValueError(f"total_gpus must be int > 0, got {inp.total_gpus!r}")
    if not isinstance(inp.gpu_ids, list) or not inp.gpu_ids:
        raise ValueError("gpu_ids must be a non-empty list[int]")
    for gpu_id in inp.gpu_ids:
        if not isinstance(gpu_id, int):
            raise ValueError(f"gpu_id must be int, got {gpu_id!r}")
        if gpu_id < 0:
            raise ValueError(f"gpu_id must be >= 0, got {gpu_id!r}")
        if gpu_id >= inp.total_gpus:
            raise ValueError(
                f"gpu_id {gpu_id} must be within [0, total_gpus), total_gpus={inp.total_gpus}"
            )


def validate_request_ids(request_ids: Iterable[str]) -> None:
    for rid in request_ids:
        validate_request_id(rid)


def validate_optional_timeout_s(timeout_s: Optional[float]) -> None:
    if timeout_s is None:
        return
    if not isinstance(timeout_s, (int, float)):
        raise ValueError(f"timeout_s must be float|None, got {type(timeout_s).__name__}")
    if timeout_s <= 0:
        raise ValueError(f"timeout_s must be > 0, got {timeout_s!r}")

