"""Ray actor resolution utilities for rlix."""
from __future__ import annotations

from typing import Any

import ray


def get_actor_or_raise(name: str, namespace: str, *, error_context: str) -> Any:
    """Get an existing Ray actor by name, raising RuntimeError if not found.

    Used when the caller requires the actor to already exist (e.g., scheduler,
    coordinator) and wants a clear error message on startup ordering problems.
    """
    try:
        return ray.get_actor(name, namespace=namespace)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to resolve actor {name!r} in namespace {namespace!r}. {error_context}"
        ) from exc
