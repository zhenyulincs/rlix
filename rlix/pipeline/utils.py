
from __future__ import annotations
import os
from typing import List


def parse_env_timeout_s(env_key: str, default_s: float) -> float:
    """Read a timeout in seconds from an env var; fail-fast on invalid values.

    Returns default_s if the env var is not set.
    Raises RuntimeError if the value is not a valid positive number.
    """
    raw = os.environ.get(env_key)
    if raw is None:
        return default_s
    try:
        value = float(raw)
    except ValueError as exc:
        raise RuntimeError(f"{env_key} must be a number, got: {raw!r}") from exc
    if value <= 0:
        raise RuntimeError(f"{env_key} must be > 0, got {value}")
    return value


def validate_resize_params(
    dp_ranks_to_remove: List[int], dp_ranks_to_add: List[int]
) -> None:
    """Validate resize_infer parameters: exactly one list must be non-empty."""
    if not isinstance(dp_ranks_to_remove, list):
        raise ValueError("dp_ranks_to_remove must be list[int]")
    if not isinstance(dp_ranks_to_add, list):
        raise ValueError("dp_ranks_to_add must be list[int]")
    if bool(dp_ranks_to_remove) == bool(dp_ranks_to_add):
        raise ValueError("Exactly one of dp_ranks_to_remove or dp_ranks_to_add must be non-empty")
