from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

REQUEST_ID_DELIMITER = ":"


def validate_pipeline_id(pipeline_id: str) -> None:
    if not isinstance(pipeline_id, str):
        raise ValueError(f"pipeline_id must be str, got {type(pipeline_id).__name__}")
    if pipeline_id == "":
        raise ValueError("pipeline_id must be non-empty")
    if REQUEST_ID_DELIMITER in pipeline_id:
        raise ValueError(f"pipeline_id must not contain {REQUEST_ID_DELIMITER!r}: {pipeline_id!r}")


@dataclass(frozen=True, slots=True)
class RegisterValidationInput:
    pipeline_id: str
    ray_namespace: str
    cluster_tp_configs: Dict[str, int]
    cluster_device_mappings: Dict[str, List[int]]


def validate_register_pipeline(inp: RegisterValidationInput) -> None:
    validate_pipeline_id(inp.pipeline_id)
    if not isinstance(inp.ray_namespace, str) or inp.ray_namespace == "":
        raise ValueError("ray_namespace must be non-empty str")
    if not isinstance(inp.cluster_tp_configs, dict) or not inp.cluster_tp_configs:
        raise ValueError("cluster_tp_configs must be non-empty dict[str,int]")
    if not isinstance(inp.cluster_device_mappings, dict) or not inp.cluster_device_mappings:
        raise ValueError("cluster_device_mappings must be non-empty dict[str,list[int]]")
    if set(inp.cluster_tp_configs.keys()) != set(inp.cluster_device_mappings.keys()):
        missing_tp = sorted(set(inp.cluster_device_mappings.keys()) - set(inp.cluster_tp_configs.keys()))
        missing_map = sorted(set(inp.cluster_tp_configs.keys()) - set(inp.cluster_device_mappings.keys()))
        raise ValueError(f"cluster config mismatch: missing tp_size for {missing_tp}, missing device_mapping for {missing_map}")

    used_non_infer: set[int] = set()
    for cluster_name, tp_size_raw in inp.cluster_tp_configs.items():
        try:
            tp_size = int(tp_size_raw)
        except Exception as e:
            raise ValueError(f"tp_size must be int for cluster {cluster_name!r}, got {tp_size_raw!r}") from e
        if tp_size <= 0:
            raise ValueError(f"tp_size must be > 0 for cluster {cluster_name!r}, got {tp_size!r}")

        device_mapping = list(inp.cluster_device_mappings.get(cluster_name) or [])
        if not device_mapping and cluster_name != "reward":
            raise ValueError(f"device_mapping must be non-empty for cluster {cluster_name!r}")
        if cluster_name == "reward" and device_mapping:
            # TODO: support GPU reward clusters (currently restricted to CPU-only).
            raise ValueError("reward cluster only supports CPU-only mode: reward.device_mapping must be empty")
        if device_mapping and len(device_mapping) != len(set(device_mapping)):
            raise ValueError(f"device_mapping has duplicates for cluster {cluster_name!r}")

        for gpu in device_mapping:
            if not isinstance(gpu, int):
                raise ValueError(f"device_mapping must be list[int] for cluster {cluster_name!r}, got {type(gpu).__name__}")
            # Phase 3 policy: allow `actor_infer` GPU overlap with other clusters (optional colocation).
            # We still disallow overlaps among non-actor_infer clusters to avoid accidental double-mapping.
            if cluster_name != "actor_infer":
                if gpu in used_non_infer:
                    raise ValueError(f"device_mapping overlaps across non-actor_infer clusters within pipeline {inp.pipeline_id!r}: gpu={gpu}")
                used_non_infer.add(gpu)

        if device_mapping and len(device_mapping) % tp_size != 0:
            raise ValueError(
                f"cluster {cluster_name!r} has len(device_mapping)={len(device_mapping)} not divisible by tp_size={tp_size}"
            )
