"""RLix entry driver for MILES fullasync GRPO pipelines.

Registration flow (F8, must run in order):
  1. allocate_pipeline_id("ft")
  2. register_pipeline(pipeline_id, namespace, cluster_tp_configs, cluster_device_mappings)
  3. admit_pipeline(pipeline_id)
  4. create MilesCoordinator actor
  5. create_pipeline_actor(pipeline_config=miles_args)
  6. pipeline.initialize_pipeline()
  7. pipeline.run()

DO NOT run train_async.py with RLIX_CONTROL_PLANE=rlix — it will fail fast with a clear error.
Use this script instead.

Usage:
    RLIX_CONTROL_PLANE=rlix python examples/rlix/run_miles_rlix.py \
        --model-path Qwen/Qwen2.5-0.5B \
        --actor-num-nodes 1 --actor-num-gpus-per-node 2 \
        --rollout-num-gpus 4 --rollout-num-gpus-per-engine 2 \
        --custom-generate-function-path miles.rollout.generate_hub.multi_turn.generate \
        --max-weight-staleness 2 --offload-train True \
        --model-update-transport cpu_serialize \
        [additional miles args...]
"""
from __future__ import annotations

import logging
import os
import sys

import ray

logger = logging.getLogger(__name__)


def _build_cluster_configs(args: object) -> tuple[dict, dict]:
    """Derive cluster_tp_configs and cluster_device_mappings from MILES args.

    First-build: sorted contiguous mapping only. Validated by F10.
    Non-contiguous mapping needs explicit adapter (F12 follow-up, §Cut 1').
    """
    actor_num_nodes = int(getattr(args, "actor_num_nodes", 1))
    actor_num_gpus_per_node = int(getattr(args, "actor_num_gpus_per_node", 1))
    rollout_num_gpus = int(getattr(args, "rollout_num_gpus", 4))
    rollout_num_gpus_per_engine = int(getattr(args, "rollout_num_gpus_per_engine", 2))

    train_devices = list(range(actor_num_nodes * actor_num_gpus_per_node))
    infer_devices = list(range(rollout_num_gpus))

    cluster_device_mappings = {
        "actor_train": train_devices,
        "actor_infer": infer_devices,
    }
    cluster_tp_configs = {
        "actor_train": 1,                          # Megatron: 1 GPU per worker (TP handled internally)
        "actor_infer": rollout_num_gpus_per_engine, # SGLang TP = gpus per engine
    }
    return cluster_tp_configs, cluster_device_mappings


def main() -> None:
    # Enforce RLix mode: this script must only run with RLIX_CONTROL_PLANE=rlix.
    if os.environ.get("RLIX_CONTROL_PLANE") != "rlix":
        os.environ["RLIX_CONTROL_PLANE"] = "rlix"
        logger.info("[run_miles_rlix] Set RLIX_CONTROL_PLANE=rlix")

    # Parse MILES args.
    try:
        from miles.utils.arguments import parse_args
    except ImportError as exc:
        raise RuntimeError(
            "miles package not found. Ensure external/miles is on PYTHONPATH. "
            f"Error: {exc}"
        ) from exc

    args = parse_args()

    # Ensure Ray is initialized before touching rlix actors.
    if not ray.is_initialized():
        ray.init(address="auto", ignore_reinit_error=True)

    # F8: orchestrator registration (allocate → register → admit).
    from rlix.protocol.types import get_pipeline_namespace, RLIX_NAMESPACE, ORCHESTRATOR_ACTOR_NAME

    try:
        orchestrator = ray.get_actor(ORCHESTRATOR_ACTOR_NAME, namespace=RLIX_NAMESPACE)
    except ValueError as exc:
        raise RuntimeError(
            f"rlix:orchestrator not found in namespace '{RLIX_NAMESPACE}'. "
            "Ensure the RLix orchestrator is running before starting the pipeline. "
            f"Original error: {exc}"
        ) from exc

    cluster_tp_configs, cluster_device_mappings = _build_cluster_configs(args)

    # Step 1: allocate pipeline_id.
    pipeline_id: str = ray.get(orchestrator.allocate_pipeline_id.remote("ft"))
    ray_namespace: str = get_pipeline_namespace(pipeline_id)
    logger.info("[run_miles_rlix] allocated pipeline_id=%s namespace=%s", pipeline_id, ray_namespace)

    # Step 2: register topology.
    ray.get(
        orchestrator.register_pipeline.remote(
            pipeline_id=pipeline_id,
            ray_namespace=ray_namespace,
            cluster_tp_configs=cluster_tp_configs,
            cluster_device_mappings=cluster_device_mappings,
        )
    )

    # Step 3: admit pipeline (scheduler starts allocating GPUs).
    ray.get(orchestrator.admit_pipeline.remote(pipeline_id=pipeline_id))
    logger.info("[run_miles_rlix] pipeline admitted: pipeline_id=%s", pipeline_id)

    # Step 4: create MilesCoordinator actor.
    from rlix.pipeline.miles_coordinator import MilesCoordinator, MILES_COORDINATOR_MAX_CONCURRENCY

    CoordinatorActor = ray.remote(MilesCoordinator)
    coordinator = CoordinatorActor.options(
        name=f"rlix:coordinator:{pipeline_id}",
        namespace=ray_namespace,
        get_if_exists=True,
        max_concurrency=MILES_COORDINATOR_MAX_CONCURRENCY,
        runtime_env={"env_vars": {
            "RLIX_CONTROL_PLANE": "rlix",
            "PIPELINE_ID": pipeline_id,
            "ROLL_RAY_NAMESPACE": ray_namespace,
        }},
    ).remote(pipeline_id=pipeline_id, pipeline_config=args)

    # Step 5: create pipeline actor (keyword-only per Coordinator ABC).
    pipeline = ray.get(coordinator.create_pipeline_actor.remote(pipeline_config=args))

    # Step 6: initialize pipeline (topology validation + cluster allocation + cache bootstrap).
    logger.info("[run_miles_rlix] calling initialize_pipeline...")
    result = ray.get(pipeline.initialize_pipeline.remote())
    if not result.success:
        raise RuntimeError(f"initialize_pipeline failed. pipeline_id={pipeline_id!r}")
    logger.info("[run_miles_rlix] initialize_pipeline succeeded. pipeline_id=%s", pipeline_id)

    # Step 7: run main training loop.
    # Phase A stub: run() raises NotImplementedError until Phase C implements the GRPO loop.
    logger.info("[run_miles_rlix] calling pipeline.run()...")
    try:
        ray.get(pipeline.run.remote())
    except NotImplementedError as exc:
        logger.warning(
            "[run_miles_rlix] pipeline.run() not yet implemented (Phase C stub): %s. "
            "initialize_pipeline() succeeded — Phase A gate complete.",
            exc,
        )
        # Phase A gate: initialize_pipeline succeeded, run() stub is expected.
        # Phase C will implement the actual GRPO training loop.
        return

    # Do NOT call ray.shutdown() — user calls `ray stop` CLI (F11 behavior).
    logger.info("[run_miles_rlix] pipeline.run() returned. Exiting driver.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    main()
