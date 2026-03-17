from __future__ import annotations

import os


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
