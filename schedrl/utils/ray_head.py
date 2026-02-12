from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class HeadNodeInfo:
    node_id: str


def get_head_node_id() -> str:
    try:
        from ray.util.state import list_nodes
    except Exception as e:
        raise RuntimeError("ray.util.state.list_nodes is required to identify the head node") from e

    nodes = list_nodes(filters=[("is_head_node", "==", "True")])
    if not nodes:
        raise RuntimeError("Could not identify head node via ray.util.state.list_nodes")
    node_id = nodes[0].get("node_id")
    if not node_id:
        raise RuntimeError(f"Head node record missing node_id: {nodes[0]!r}")
    return node_id


def get_head_node_info() -> HeadNodeInfo:
    return HeadNodeInfo(node_id=get_head_node_id())


def head_node_affinity_strategy(*, soft: bool = False):
    try:
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
    except Exception as e:
        raise RuntimeError("ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy is required") from e

    return NodeAffinitySchedulingStrategy(node_id=get_head_node_id(), soft=soft)

