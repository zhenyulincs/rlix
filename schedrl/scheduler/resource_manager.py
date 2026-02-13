from __future__ import annotations

import time
from dataclasses import dataclass

from schedrl.utils.ray_head import head_node_affinity_strategy


def _require_ray():
    try:
        import ray  # noqa: F401
    except Exception as e:
        raise RuntimeError("schedrl.scheduler.resource_manager requires ray") from e


@dataclass(slots=True)
class ResourceManager:
    def __post_init__(self):
        _require_ray()

    def get_num_gpus(self) -> int:
        """Return current Ray cluster GPU count (no waiting / gating)."""
        _require_ray()
        import ray

        cluster_resources = ray.cluster_resources()
        return int(cluster_resources.get("GPU", 0))

    def snapshot(
        self,
        *,
        wait_timeout_s: float = 10.0,
        poll_interval_s: float = 0.2,
        expected_num_gpus: int | None = None,
    ) -> dict:
        if wait_timeout_s <= 0:
            raise ValueError(f"wait_timeout_s must be > 0, got {wait_timeout_s!r}")
        if poll_interval_s <= 0:
            raise ValueError(f"poll_interval_s must be > 0, got {poll_interval_s!r}")
        if expected_num_gpus is not None and expected_num_gpus < 0:
            raise ValueError(f"expected_num_gpus must be >= 0, got {expected_num_gpus!r}")

        _require_ray()
        import ray

        deadline = time.monotonic() + float(wait_timeout_s)
        last_num_gpus = None
        last_alive_nodes = None
        last_cluster_resources = None
        while time.monotonic() < deadline:
            cluster_resources = ray.cluster_resources()
            alive_nodes = [n for n in ray.nodes() if n.get("Alive")]
            num_gpus = int(cluster_resources.get("GPU", 0))

            last_num_gpus = num_gpus
            last_alive_nodes = alive_nodes
            last_cluster_resources = cluster_resources

            if expected_num_gpus is not None:
                if num_gpus >= expected_num_gpus:
                    break
            else:
                if num_gpus > 0:
                    break

            time.sleep(float(poll_interval_s))

        else:
            raise RuntimeError(
                "Timed out waiting for Ray GPU topology to be ready. "
                f"expected_num_gpus={expected_num_gpus!r}, last_num_gpus={last_num_gpus!r}, "
                f"alive_nodes={len(last_alive_nodes or [])}, cluster_resources={last_cluster_resources!r}"
            )

        return {
            "num_gpus": int(last_num_gpus or 0),
            "alive_nodes": [
                {"NodeID": n.get("NodeID"), "NodeManagerAddress": n.get("NodeManagerAddress")} for n in (last_alive_nodes or [])
            ],
            "cluster_resources": dict(last_cluster_resources or {}),
        }


def get_or_create_resource_manager(*, name: str = "schedrl:resource_manager", namespace: str = "schedrl"):
    _require_ray()
    import ray

    try:
        return ray.get_actor(name, namespace=namespace)
    except ValueError:
        pass

    strategy = head_node_affinity_strategy(soft=False)

    @ray.remote(num_cpus=0, max_restarts=0, max_task_retries=0)
    class _ResourceManagerActor(ResourceManager):
        pass

    try:
        return (
            _ResourceManagerActor.options(
                name=name,
                namespace=namespace,
                scheduling_strategy=strategy,
                max_restarts=0,
                max_task_retries=0,
            )
            .remote()
        )
    except Exception:
        return ray.get_actor(name, namespace=namespace)
