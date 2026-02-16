from __future__ import annotations

import shutil
import sys
from pathlib import Path


def ray_cli_path() -> str:
    """Return the ray CLI path for this Python environment.

    Prefer PATH resolution (e.g. /usr/local/bin/ray) because sys.executable may be /usr/bin/python3
    while the ray CLI entrypoint is installed elsewhere.
    """
    ray_path = shutil.which("ray")
    if ray_path:
        return ray_path
    python_bin_dir = Path(sys.executable).parent
    return str(python_bin_dir / "ray")


def get_head_node_id() -> str:
    try:
        from ray.util.state import list_nodes
    except Exception as e:
        raise RuntimeError("ray.util.state.list_nodes is required to identify the head node") from e

    # ray.util.state list_* APIs only support "=" / "!=" predicates (not "==").
    nodes = list_nodes(filters=[("is_head_node", "=", "True")])
    if not nodes:
        raise RuntimeError("Could not identify head node via ray.util.state.list_nodes")
    node_id = nodes[0].get("node_id")
    if not node_id:
        raise RuntimeError(f"Head node record missing node_id: {nodes[0]!r}")
    return node_id


def head_node_affinity_strategy(*, soft: bool = False):
    try:
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
    except Exception as e:
        raise RuntimeError("ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy is required") from e

    return NodeAffinitySchedulingStrategy(node_id=get_head_node_id(), soft=soft)
