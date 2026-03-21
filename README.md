<div align="center">

<img src="assets/rlix-logo-text-horizontal.svg" width="40%" alt="RLix Logo">

<h3>Run more RL experiments. Wait less for GPUs.</h3>

<p>

  <a href="https://deepwiki.com/rlops/rlix" target="_blank">
    <img src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki">
  </a>
  
  <a href="https://github.com/rlops/rlix/stargazers">
    <img src="https://img.shields.io/github/stars/rlops/rlix?style=social" alt="Repo stars">
  </a>

  <a href="https://github.com/rlops/rlix/issues">
    <img src="https://img.shields.io/github/issues/rlops/rlix" alt="GitHub issues">
  </a>

  <a href="https://github.com/rlops/rlix/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License">
  </a>

</p>

</div>

RL research often means running lots of experiments: trying new ideas, comparing settings, and running ablations. When GPU capacity is tight, promising jobs can spend too long waiting to start. Even worse, in long-horizon agentic RL, such as coding and computer-use agents, a few slow rollouts can hold everything up while many GPUs sit idle.

RLix helps you get more out of the GPUs you already have. It lets multiple RL jobs share GPU capacity more effectively, so you can run more experiments at once, spend less time waiting for GPUs, and improve GPU utilization without changing how each pipeline trains.

## Features

- **Support on-policy and off-policy pipelines**: RLix works with both, while keeping each pipeline within its own staleness bounds.
- **Share GPU capacity across jobs**: Full-finetune pipelines can use idle GPU capacity from other jobs instead of waiting for dedicated resources.
- **Share one base model across LoRA adapters**: Multi-LoRA pipelines train multiple adapters on one shared base model, reducing GPU and memory overhead within a pipeline.
- **Grow and shrink rollouts automatically**: Rollout workers expand when demand grows and shrink when GPUs are needed elsewhere.

## Installation

`setup_env.sh` is for Linux machines with working NVIDIA GPUs and drivers already installed. It installs Miniconda if needed, creates the `rlix` Conda environment, installs Python 3.10 and CUDA 12.4 build dependencies, and installs ROLL and RLix into that environment.

```bash
git clone https://github.com/rlops/rlix.git
cd rlix
bash setup_env.sh
conda activate rlix
```

## Quick Start

The example below shows the smallest RLix setup for launching one pipeline.

Workflow overview:

1. Start RLix.
2. Create a pipeline ID.
3. Tell RLix which GPUs and namespace the pipeline will use.
4. Let RLix manage GPU allocation for the pipeline.
5. Create the pipeline coordinator.
6. Create the pipeline actor and run it.

```python
import ray
import rlix
from rlix.pipeline import PipelineCoordinator
from rlix.protocol.types import COORDINATOR_ACTOR_NAME_PREFIX

# Pipeline-specific configuration object
my_config = ...

# 1. Initialize RLix
orchestrator = rlix.init(create_if_missing=True)

# 2. Allocate a pipeline ID
pipeline_id = ray.get(orchestrator.allocate_pipeline_id.remote("ft"))

# 3. Register the pipeline's GPU topology
ray.get(
    orchestrator.register_pipeline.remote(
        pipeline_id=pipeline_id,
        ray_namespace=f"pipeline_{pipeline_id}_NS",
        cluster_tp_configs={"actor_train": 8, "actor_infer": 8},
        cluster_device_mappings={
            "actor_train": [0, 1, 2, 3, 4, 5, 6, 7],
            "actor_infer": [0, 1, 2, 3, 4, 5, 6, 7],
        },
    )
)

# 4. Admit the pipeline before GPU allocation
ray.get(orchestrator.admit_pipeline.remote(pipeline_id=pipeline_id))

# 5. Create the pipeline coordinator
CoordinatorActor = ray.remote(PipelineCoordinator)
coordinator = CoordinatorActor.options(
    name=f"{COORDINATOR_ACTOR_NAME_PREFIX}{pipeline_id}",
    namespace=f"pipeline_{pipeline_id}_NS",
).remote(pipeline_id=pipeline_id, pipeline_config=my_config)

# 6. Create and run the pipeline
pipeline_actor = ray.get(
    coordinator.create_pipeline_actor.remote(pipeline_config=my_config)
)
ray.get(pipeline_actor.run.remote())
```

## Examples

You can also run the example pipelines in [examples/](examples/) directly:

```bash
# Run a single full-finetune pipeline
conda run -n rlix --no-capture-output python examples/start_multi_pipeline_test.py --config_name full_finetune_pipeline1

# Run two full-finetune pipelines concurrently
conda run -n rlix --no-capture-output python examples/start_multi_pipeline_test.py --config_name full_finetune_pipeline1,full_finetune_pipeline2

# Run one full-finetune pipeline and one multi-LoRA pipeline concurrently
conda run -n rlix --no-capture-output python examples/start_multi_pipeline_test.py --config_name full_finetune_pipeline1,multi_lora_pipeline2
```

See [examples/](examples/) for more multi-pipeline examples and full configuration options.

## Pipeline Types

RLix currently supports two built-in pipeline types:

### Full Finetune Pipeline (`RollFullFinetunePipeline`)

Full-parameter training with elastic GPU expand and shrink. Each job trains all model weights, while idle GPU capacity can still be shared with other jobs.
Choose this when you want the best model quality and have enough GPUs and memory for full finetuning, but still want to share spare GPU capacity across jobs.

### Multi-LoRA Pipeline (`RollMultiLoraPipeline`)

Concurrent training of multiple LoRA adapters on a shared base model, with a separate optimizer for each adapter. Jobs share the base model in GPU memory while keeping adapter weights and optimizer states independent.
Choose this when you want lower GPU and memory usage than full finetuning, or when you want to train multiple adapters on the same base model and increase sharing within one pipeline.

RLix also supports custom pipelines and integrations that follow the RLix interface.

## Architecture

RLix has one shared layer that coordinates GPU allocation across jobs and one coordinator for each pipeline. Each pipeline keeps its own training logic.

```text
┌───────────────────────────────────────────────────────────┐
│                 RLix Shared Job Management Layer          │
├──────────────────┬──────────────────┬─────────────────────┤
│   Orchestrator   │    Scheduler     │  Resource Manager   │
│   (job lifecycle)│ (priorities +    │ (cluster resources) │
│                  │ rollout sharing) │                     │
└────────┬─────────┴────────┬─────────┴─────────┬───────────┘
         │                  │                   │
    ┌────▼─────┐       ┌────▼─────┐        ┌────▼─────┐
    │FullFine- │       │Multi-LoRA│        │Custom /  │
    │tune Job 1│       │  Job 2   │        │External  │
    │          │       │          │        │  Job N   │
    └────┬─────┘       └────┬─────┘        └────┬─────┘
         │                  │                   │
    ┌────▼──────────────────▼───────────────────▼────┐
    │               Shared GPU Capacity              │
    │   [GPU 0] [GPU 1] [GPU 2] [GPU 3] ... [GPU N]  │
    └────────────────────────────────────────────────┘
```

## How GPU Scheduling Works

RLix gives GPUs to higher-priority stages first. Most stages keep their GPUs until they finish. Rollout is the flexible stage: it can use spare GPU capacity when available and give it back when higher-priority work needs it.

Rollout has the lowest priority and is always preemptable, meaning it can give GPUs back when higher-priority work needs them. When multiple jobs are rolling out at the same time, RLix divides the available GPU capacity based on how much rollout work each job still has to do, while still respecting placement constraints. To keep rollout workers lightweight, RLix loads inference weights only while a worker is active and releases them again when the worker shrinks.

From highest to lowest priority:

* **0 Initialization**: Model loading; must complete before scheduling begins.
* **1 Actor Training**: Policy gradient update.
* **2 Critic Training**: Value function update.
* **3 Old-Policy Log Probs**: Log-probability computation under the previous policy.
* **4 Reference-Model Log Probs**: Log-probability computation under the reference model.
* **5 Value Compute**: Value estimation for advantage calculation.
* **6 Rollout**: Trajectory sampling; can give GPUs back when needed.

## Acknowledgements

RLix was developed with extensive AI assistance, with human direction and oversight throughout.

RLix is inspired by **Partial Overlapping** scheduling from [**Alibaba/ROLL**](https://github.com/alibaba/ROLL).
