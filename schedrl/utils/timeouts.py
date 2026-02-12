from __future__ import annotations

import os
import signal
import time
from contextlib import contextmanager
from typing import Any, Optional


def get_env_timeout_s(var_name: str, default_s: float) -> float:
    if not isinstance(var_name, str) or var_name == "":
        raise ValueError("var_name must be non-empty str")
    if not isinstance(default_s, (int, float)) or default_s <= 0:
        raise ValueError(f"default_s must be > 0, got {default_s!r}")

    scale_raw = os.environ.get("ROLL_TIMEOUT_SCALE", "1")
    try:
        scale = float(scale_raw)
    except ValueError:
        scale = 1.0
    if scale <= 0:
        scale = 1.0

    scaled_default_s = float(default_s) * scale
    raw = os.environ.get(var_name)
    if raw is None or raw == "":
        return scaled_default_s
    try:
        value = float(raw)
    except ValueError:
        return scaled_default_s
    if value <= 0:
        return scaled_default_s
    return value


def get_env_timeout_optional_s(var_name: str, default_s: float) -> Optional[float]:
    raw = os.environ.get(var_name)
    if raw is None or raw == "":
        return get_env_timeout_s(var_name, default_s)

    normalized = raw.strip().lower()
    if normalized in {"none", "null", "inf", "infinite", "infinity"}:
        return None

    return get_env_timeout_s(var_name, default_s)


@contextmanager
def timeout_context(seconds: float, operation: str):
    if not isinstance(seconds, (int, float)) or seconds <= 0:
        raise ValueError(f"seconds must be > 0, got {seconds!r}")
    if not isinstance(operation, str) or operation == "":
        raise ValueError("operation must be non-empty str")

    def timeout_handler(signum: int, frame: Any):
        raise TimeoutError(f"Operation {operation!r} timed out after {seconds}s")

    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(int(seconds))
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def get_named_actor_with_timeout(
    actor_name: str,
    *,
    namespace: str,
    timeout_s: float = 60.0,
    poll_interval_s: float = 0.5,
):
    if timeout_s <= 0:
        raise ValueError(f"timeout_s must be > 0, got {timeout_s!r}")
    if poll_interval_s <= 0:
        raise ValueError(f"poll_interval_s must be > 0, got {poll_interval_s!r}")

    try:
        import ray
    except Exception as e:
        raise RuntimeError("ray is required to look up named actors") from e

    deadline = time.monotonic() + float(timeout_s)
    last_error: Optional[BaseException] = None
    while time.monotonic() < deadline:
        try:
            return ray.get_actor(actor_name, namespace=namespace)
        except ValueError as e:
            last_error = e
            time.sleep(float(poll_interval_s))
    raise RuntimeError(
        f"Timed out after {timeout_s:.1f}s waiting for named Ray actor {actor_name!r} "
        f"in namespace={namespace!r}. Last error: {last_error}"
    )

