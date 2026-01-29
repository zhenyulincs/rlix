---
title: "Time-Sharing a GPU Cluster Across RL Pipelines: Decoupling Recipes from GPU Scheduling"
goal: for blog post and research paper
---
## Terminology
todo cross check with nvidia terminology 
w
https://docs.nvidia.com/nemo/gym/latest/about/concepts/key-terminology.html

- `GPU cluster`: physical machines + GPUs.
- `GPU pool`: the set of GPUs managed by the centralized scheduler.
- `Worker cluster`: a logical Ray actor group for a pipeline role (e.g., `actor_infer`, `actor_train`).
- `Pipeline`: one independent RL run (generator + trainer + its worker clusters).
- `Pipeline coordinator`: per-pipeline driver that coordinates generation/training phases and interacts with the centralized scheduler.
- `Centralized scheduler`: global service that allocates/reclaims GPUs across pipelines.
- `Cluster controller`: per-cluster control-plane component that manages DP workers (expand/shrink/preempt/resume/load/offload) and tracks GPU ownership.
- `Training recipe` (informal): the per-pipeline training setup (configs + code). This design does not require changing recipes.
- `Generator`: the generation component that runs rollouts (implemented as `actor_infer` here).
- `Trainer`: the training component that updates weights (implemented as `actor_train` / `critic` here).
- `Generation cluster`: the worker cluster that runs the generator (`actor_infer`).
- `Training cluster`: the worker cluster(s) that run training compute (e.g., `actor_train`, `critic`, `reference`, `rewards`).
- `Generation phase` / `rollout`: sample collection via the generator.
- `Training phase`: model update and supporting compute (train/value/logprob/reference/reward).
- `RolloutRouter` (`RequestScheduler`): per-pipeline generation-side control-plane component that routes rollout requests to active DP workers and coordinates
  shrink/expand/release with the centralized scheduler.
- `RolloutBuffer` (`GroupQueueManager`): per-pipeline buffer that stores pending/completed rollout items and supports progress tracking for scheduling.
- `DP rank`: integer id (0..dp-1). `DP worker`: allocation unit = one DP rank worth of work; consumes `tp_size` GPUs as a bundle.
- `TP size`: GPUs per DP worker.
- Model verbs: `broadcast` (sender sends), `sync` (receiver fetch+apply), `apply` (load into model), `offload` (GPU→CPU keep state), `drop` (free).

## Outline

- 1) Introduction
  - The problem to solve
    - Background: LLM development focus is shifting from pre-training to RL post-training, and RL training is a multi-phase loop.
    - Agentic sampling is becoming longer-horizon and long-tailed:
      - Episodes can span many turns with retries; a few stragglers dominate wall time.
      - Straggler effect: the system often waits for the slowest episode before it can advance the pipeline stage, which wastes GPUs.
    - Concurrency is the default: agentic RL is no longer one job.
      - Research workflow: parallel experiments across configs/algorithms/base models/datasets/hyperparameters.
      - Multi-tenant service: tuning APIs (e.g., Tinker, OpenAI RFT) run many user jobs at once.
    - Shared-GPU reality: the same GPU cluster is shared across these concurrent pipelines.
    - Core requirement: strong programmability/usability for researchers and high hardware utilization.
  - Existing approaches (and why they fall short)
    - Approach 1: fixed GPUs per pipeline
      - Each pipeline gets a fixed GPU budget for its roles.
      - Some systems time-share within a pipeline across phases by running async, but that often changes the algorithm and adds staleness.
      - Problem: phase bubbles and long-tail rollouts make the best GPU split change over time, so fixed budgets waste capacity.
    - Approach 2: two disaggregated fixed pools by phase (RollMux as an example)
      - Keep a rollout pool and a training pool, then time-share across pipelines within each pool.
      - Problem: the two pools still need careful balancing as rollout dynamics change; one pool can overload while the other idles.
    - Approach 3: one shared homogeneous pool with a single global loop (strawman)
      - Put all pipelines and all phases under one monolithic controller with a global view.
      - Pro: good utilization potential and simple for engineers to implement in one place.
      - Con: the controller couples scheduling and execution, becomes a bottleneck, does not scale well, and is hard to maintain and hard for
        researchers to program and evolve.
  - Transition: the more common reality is one shared homogeneous pool
    - Many teams do not have (or do not want) two dedicated rollout/training clusters; they have one shared pool of similar GPUs.
    - In that setting, rollout and training for many pipelines all contend for the same GPUs.
    - The dilemma: GPUs should be shared as much as possible across stages, but each pipeline's training logic is independently managed by
      researchers.
    - The punch line: you need a centralized scheduler, with decentralized per-pipeline training logic.
  - Our solution: one homogeneous pool shared by rollout and training
    - Treat the cluster as a single homogeneous GPU pool; time-share it across phases and pipelines.
    - Decouple per-pipeline logic from GPU scheduling (policy vs execution).
    - Semantics note: time-sharing does not change the training algorithm/semantics; behavior is equivalent to resource-isolated training (exclusive GPUs, time-sliced).
  - Quick comparison (four common approaches)

    | Approach | Hardware utilization | Researcher usability |
    |---|---|---|
    | Fixed GPUs per pipeline, per-pipeline controller | ❌ Low: long-tailed rollout introduces an on-policy (stability) vs utilization tradeoff | ✅ High: isolated, easy to program and evolve per pipeline |
    | Separate rollout/training pools, time-sharing within each pool | ⚠️ Medium: good within each pool, but needs rebalancing across pools | ❓ Implementation dependent |
    | Single global controller for one shared pool | ⚠️ High potential, but the controller limits scalability | ❌ Low: pipeline logic is tightly coupled to centralized scheduling |
    | Our approach: one shared pool, centralized scheduler coordinates multiple pipelines | ✅ High: global sharing across phases and pipelines | ✅ High: pipeline logic stays independent while the scheduler handles sharing |
  - What makes it hard
    - Coordination across pipelines and the scheduler: pipeline coordinators and the centralized scheduler are separate distributed components, so you need a clear
      protocol for GPU ownership, preemption, and release.
    - Scheduling mixed workloads: many tasks (training/value/logprob) are non-preemptible; rollout is preemptible but has dynamic demand and stragglers, and there
      are cross-stage dependencies, which makes it hard to keep the system busy.
    - Model sync and memory pressure: CPU/GPU memory is shared across pipelines and limited, naive syncing/caching can blow up memory easily.

- 2) Architecture: decouple GPU scheduling from pipeline coordination
  - Hierarchical control:
    - Centralized scheduler: decides GPU allocation across pipelines (global policy).
    - Pipeline coordinator (per pipeline): manages phase progression and requests GPUs.
    - Cluster controller (per cluster): turns allocations into DP-worker actions (expand/shrink/preempt/resume) and lifecycle ops (load/offload/run).
    - DP workers: directly own and use TP-sized GPU bundles to execute compute.
  - Execution plane: DP workers do the compute; cluster controllers provide the mechanics (preempt/resume/load/offload/run).
  - Design sketch (top = pipelines, bottom = GPUs):
    ```text
    Pipelines (N independent RL runs)
    ┌───────────────────────────────┐   ┌───────────────────────────────┐
    │ Pipeline A                    │   │ Pipeline B                    │
    │  - Pipeline coordinator       │   │  - Pipeline coordinator       │
    │  - Cluster controller(s)      │   │  - Cluster controller(s)      │
    │  - Training clusters          │   │  - Training clusters          │
    │  - Generation cluster         │   │  - Generation cluster         │
    │  - RolloutRouter              │   │  - RolloutRouter              │
    │  - RolloutBuffer              │   │  - RolloutBuffer              │
    └───────────────┬───────────────┘   └───────────────┬───────────────┘
                    │ request/release GPUs                           │
                    └──────────────────────────┬─────────────────────┘
                                               v
                             ┌─────────────────────────────────┐
                             │       Centralized scheduler     │
                             └──────────────────┬──────────────┘
                                                v
                             ┌─────────────────────────────────┐
                             │            Shared GPU pool      │
                             └─────────────────────────────────┘
    ```
  - Coordination protocol:
    - Training clusters (trainer-side compute: train/value/logprobs): execution is driven by the pipeline coordinator (request → run → `release_gpus()`),
      i.e. the scheduler allocates but does not preempt the computation once granted.
    - Generation cluster (generator-side compute: `actor_infer` rollouts): preemptible DP workers and completion-driven release:
      - Scheduling is driven by the centralized scheduler: it can preempt/resume DP ranks while rollouts run.
      - Allocation is at DP/TP-bundle granularity and may be partial.
      - After each training step, the pipeline coordinator releases (or is preempted from) generation DP workers, so those GPUs can be reclaimed and reused.
      - Workload rebalance on preemption/resume: `shrink_workers()` / `expand_workers()` rebalance routing by aborting in-flight requests and clearing sticky mappings,
        so affected work is retried and naturally re-routed to the remaining/new DP workers (“migration by abort + remap”).
      - Protocol invariants:
        - Ownership: each GPU id is either idle or owned by exactly one cluster allocation (never both).
        - Allocation unit: `DP worker` (one DP rank) is the atomic preemption/activation unit; each DP worker consumes `tp_size` GPUs.
        - Per-training-step preemption: the scheduler can safely preempt generation only at training-step boundaries.
    - Model-sync coordination (selective sync-on-resume):
      - After each training step, the scheduler can suspend/preempt generation DP workers, so they are stopped and safe to sync new weights on resume.
      - Later, when the scheduler resumes `actor_infer`, it integrates model sync into scheduling by calling
        `expand_workers(selective_update=True)`: only the re-activated DP ranks sync the latest weights during resume (no push-to-all).
      - Versioning: keep only the latest weights cache, keyed by `global_step`; resume passes `requested_global_step=latest` and only newly activated DP ranks sync/apply.
  - Control-plane component discovery:
    - Pipeline components and the centralized scheduler are Ray actors, discovered by name within a pipeline-specific Ray namespace
      (centralized scheduler / RolloutRouter / RolloutBuffer; code: scheduler/request-scheduler/group-queue-manager).
  - Workflow diagram
    ```mermaid
    sequenceDiagram
      participant CGS as CentralizedGPUScheduler
      participant RCP as Pipeline coordinator (per pipeline)
      participant W as Cluster controllers + DP workers (incl. RolloutRouter/RequestScheduler)

      RCP->>CGS: request_gpus(training cluster, priority)
      CGS-->>RCP: allocated GPUs
      RCP->>W: run training compute (train/value/logprobs)
      RCP->>CGS: release_gpus(training cluster)

      RCP->>CGS: request_gpus(generation cluster (actor_infer), generation/rollout)
      CGS->>W: resume/preempt DP ranks via expand_workers(...) / shrink_workers(...) (preemptible)
      W->>W: (re)activate DP ranks, run rollout

      RCP->>W: release generation GPUs (blocking)
      W->>CGS: notify_cluster_released() (blocking ACK)
      CGS->>W: preempt to reclaim GPUs
    ```

- 3) Scheduling heterogeneous, dynamic workloads: priority + FIFO + progress-aware allocation
  - Priority tiers are ordered by “closeness to end of a training step” (a topological sort of the per-training-step dependency graph):
    - Higher-priority clusters are closer to producing a completed training step (e.g., training/logprob/value), so scheduling them first reduces training-step tail latency.
    - The generation/rollout cluster is lowest priority and preemptible.
    - Within the same priority tier, we use FIFO (older request first).
    - This approximates shortest-remaining-job-first scheduling.
  - Heartbeat-driven monitoring keeps it lightweight: rollout workers send progress heartbeats (`remaining`, `percent_remaining`, `oldest_unfinished_creation_ts`), so the
    scheduler can plan using backlog signals without per-episode polling, reducing scheduler overhead while staying responsive to workload changes.
    - Metric meanings + reporting cadence:
      - `remaining`: how many rollouts are still missing for the current batch (`remaining = total_required - collected`).
      - `percent_remaining`: remaining ratio in `[0, 1]` (`percent_remaining = remaining / max(total_required, 1)`), not a true percentage.
      - `oldest_unfinished_creation_ts`: timestamp of the oldest unfinished rollout group; used as a FIFO tie-break within a priority tier.
      - Heartbeat cadence is event-driven: send at batch start, and whenever `percent_remaining` crosses a 2% progress band (`floor(percent_remaining * 50)` changes).
      - Under preemption: generation DP workers may be shrunk; in-flight requests on shrinking workers are aborted and remapped to active workers, so progress can pause
        until retries complete (no “fake” progress from aborted work).
  - Planning rule (high-level pseudocode):
    ```text
    # Inputs:
    #   pending_requests: (cluster_id, priority, timestamp)
    #   generation_progress[pipeline]: remaining, percent_remaining, oldest_unfinished_creation_ts
    #   active_allocations for generation: active_dp_workers (DP rank bundles), inactive_dp_workers
    #
    # Step 1: Non-generation planning (non-preemptible)
    for req in sort(pending_non_generation, by=(priority asc, timestamp asc)):
        if enough idle GPUs for req: allocate full device_mapping
        else: shrink generation DP workers (lowest priority) to free DP bundles, then allocate if possible
    #
    # Step 2: Generation planning (preemptible, gap-ratio)
    weight[p] = remaining[p] * tp_size[p]
    target_ratio[p] = weight[p] / sum(weight)
    existing_ratio[p] = active_gpu[p] / gen_budget_gpus
    gap[p] = target_ratio[p] - existing_ratio[p]
    while exists receiver with gap>0 and (idle GPUs or shrinkable donors exist):
        pick receiver with max normalized_gap
        activate one inactive DP worker bundle for receiver
          - if bundle GPUs not idle: pick donor DP bundles from pipelines with gap<0 (within shrink budget)
            prefer donors with larger percent_remaining (least progress) to minimize wasted work
          - plan: shrink donor bundles, expand receiver bundle
    ```
  - Gap-ratio allocation happens at DP-worker granularity: DP workers are fixed `tp_size` GPU bundles with fixed placement, so the scheduler resumes/preempts whole DP
    workers to match backlog (`remaining * tp_size`). When shrinking donors, we preferentially preempt rollouts with the least progress (highest `percent_remaining`).


- 4) Model syncing and memory management optimization: selective + sync-on-resume
  - Problem: broadcasting new weights to all rollout workers right after each training step is unsafe and wasteful. Some rollout GPUs may be busy with other work, so an
    immediate sync can easily trigger OOM; but allocating the full rollout GPUs just to do lightweight weight syncing is slow and overkill.
  - Solution:
    - Keep a single “latest weights” cache (CPU buckets) per pipeline; only designated sender ranks build/own this cache (one cached copy per pipeline).
    - Use bucket-based broadcast on the sender side: keep all trainer states offloaded to CPU, and only stage one small weight bucket on GPU at a time during sync.
    - Sync selectively on demand during scheduling: when `actor_infer` resumes, `expand_workers(selective_update=True)` syncs lastest weights only for the re-activated DP
      ranks, avoiding unnecessary syncs and memory blow-ups on rollout workers.
    - When `actor_infer` is preempted, workers aggressively drop footprint (weights + KV cache) instead of keeping inactive copies around.

----------

- 5) Evaluation
  - Goal: beat static per-pipeline partitioning and single-pipeline-only runs in number of gpus per pipeline while maintaining the comparable training speed. 
  - Workload: Terminal Bench
  - Throughput metric: `training steps / hour / GPU` (steps normalized by total GPUs used by all pipelines).
  - Scale settings:

    | Model | Total GPUs | Pipelines | GPUs / pipeline |
    | --- | ---: | ---: | ---: |
    | Qwen3 14B | 8 | 1 | 8 |
    | Qwen3 14B | 8 | 2 | 4 |
    | Qwen3 14B | 12 | 4 | 3 |
    | Qwen3 30A3 | 64 | 1 | 64 |
    | Qwen3 30A3 | 64 | 2 | 32 |
    | Qwen3 30A3 | 96 | 4 | 24 |

  - Plots: the results show that end-to-end time per training step and rollout time per training step remain comparable across different pipeline counts for the same model, indicating ~3× throughput improvement without slowing training.


30a3 1 2 4 pipelines terminal-bench

rollout time per training step:
setup: grpo 30a3 64gpu 
data:
 time/rollout

![alt text](./fig/30a3-terminal-bench-rollout.png)

end to end time per training step:
data:
 time/per_step_e2e

![alt text](./fig/30a3-terminal-bench-endtoend.png)

14b 1 2 4 pipelines terminal-bench

rollout time per training step:
setup: grpo 14b 8gpu 
data:
 time/rollout

![alt text](./fig/14b-terminal-bench-rollout.png)

end to end time per training step:

data:
 time/per_step_e2e

![alt text](./fig/14b-terminal-bench-endtoend.png)

- 6) What works today, and what’s next
  - Current stable parts and rough edges.
    - Tested mainly with a fixed set of RL jobs (system can support dynamic jobs, but that is not fully tested yet).
    - GPU placement / device_mapping for each task cluster is manually constructed today (not yet auto-placed).
  - Next steps:
    - Optimize task placement policy (GPU bundles / topology-aware placement).
    - Tighter integration with hyperparameter tuning (tuning loop as a first-class workload).
    - Keep evolving the scheduling policy (better priorities, fairness, progress signals, and efficiency).
    - Production deployment as a service for dynamic, user-programmable training pipelines.
    - Improve admission control for dynamic jobs(avoid overload, enforce quotas/SLOs).
    - Cross-framework compatibility (interoperability at the scheduling layer): run pipelines from different RL frameworks( https://www.anyscale.com/blog/open-source-rl-libraries-for-llms) on the same cluster without rewriting them (OS-style time-sharing across processes) 
    - integrate multi-lora adapters per pipeline

## extra content for paper submission 
This is extra content for paper submission beyond the content for blog post

- [ ] measure the model update overhead and compare with the baseline
- [ ] measure the memory overhead in GPU, unoffloaded part
- [ ] Sanity-check quality: compare reward/success curves vs resource-isolated baseline (same configs, fixed seeds if possible).
- [ ] Baselines to report:
  - Static partition (fixed GPUs per pipeline)
  - Sequential pipelines (run N jobs one-by-one)
- [ ] Minimal ablations:
  - No preemption
  - No progress-aware planning
  - Full-sync instead of selective sync


## title candidates
### paper style
Scaling Concurrent RL by Decoupling GPU Scheduling from Training Orchestration

(Scalable is not the major benefits?  but it is a important() benefits for large systems)

Centralized GPU Scheduling with Decentralized Training Orchestration for Scaling Concurrent RL
Decoupling Pipeline Execution from GPU Scheduling in Scalable RL Systems

"Decoupling Hardware Allocation from Decentralized RL Pipeline Execution"


**Option B: Principle-Focused**
"Decoupling RL Pipeline Control from GPU Scheduling in Concurrent RL"

**Option C: With Subtitle (Connects to Blog)**
Title: "Centralized GPU Scheduling with Decentralized Training Pipelines for Concurrent RL"
Subtitle: "Decoupling Hardware Allocation from Training Pipeline"

### blog style

**Recommended:**
"How We Time-Share GPUs Across RL Pipelines Without Changing Training Recipes"

**Alternatives:**
- "Running 4× More RL Experiments on the Same GPU Cluster Without Slowing Down"
- "Centralized Scheduling, Decentralized Recipes: Scaling Concurrent RL Runs"
- "How We Time-Share One GPU Cluster Across Multiple RL Pipelines"


## Backlog (future writing)

----- 

pass 2

- P0 Terminology + structure
  - Issue: mixed taxonomy (`GEN`/`generation`/`ROLLOUT`; recipe/training logic/pipeline loop; cluster meanings; DP worker vs DP rank).
    - Done:
      - Added `## Terminology` after Section 1 and standardized doc language to `generation/training`, `pipeline coordinator`, and `centralized scheduler`.
  - Issue: inconsistent title/subtitle wording (“GPU scheduling” vs “cluster scheduling”).
    - Done:
      - Standardized on `GPU scheduling` in title/subtitle and updated the Section 2 header to match.

- P0 Protocol correctness (what is guaranteed)
  - Issue: rollout preemption/release protocol lacks explicit invariants and failure handling.
    - Done:
      - Added a “Protocol invariants” block under Section 2 generation cluster (ownership, DP-worker atomicity, per-training-step preemption before sync/resume).
  - Issue: model-sync versioning is not stated (how resumed ranks get the latest weights).
    - Done: added a single “Versioning” bullet under Section 2 model-sync coordination; the system keeps only the latest `global_step` cache.

- P1 Scheduling policy clarity (define the policy, not just name it)
  - Issue: “SRJF-like + FIFO + gap-ratio” is asserted without a definition or rationale.
    - Done:
      - Added a “Priority tiers” explanation and simple planning pseudocode in Section 3 (priority+FIFO for non-generation, gap-ratio for generation).
  - Issue: progress signals (`remaining`, `percent_remaining`, `oldest_unfinished_creation_ts`) are not sourced.
    - Done:
      - Added “Metric meanings + reporting cadence” under Section 3, including 2% progress-band reporting and preempt abort/remap behavior.

- P0/P1 Evaluation completeness
  - Issue: workload mismatch (Terminal Bench main text vs Sokoban appendix).
    - Done:
      - Moved Sokoban results to the appendix; keep main Evaluation focused on Terminal Bench.
      - Add a one-line caption per figure: model, total GPUs, pipelines, metric.
  - Issue: throughput claim lacks baselines and causality (which technique causes which gain).
    - Done:
      - Define “throughput” as `training steps / hour / GPU` (at fixed `rollout_batch_size` and comparable configs).
      - Baselines + ablations moved to `## extra content for paper submission`.
  - Issue: quality/convergence not discussed but “without slowing training” can be misread as “no quality impact”.
    - Done:
      - Added a “Semantics note” in Section 1 (under “Our solution”) clarifying the training semantics are unchanged (equivalent to resource-isolated training).
      - Added a paper-only TODO to sanity-check quality vs a resource-isolated baseline.

- P2 Polish
  - Issue: draft markers (`status: draft`, inline TODO comment) and “step” ambiguity.
    - Done:
      - Removed `status: draft` and the `<!-- rvst ... -->` comment.
      - Clarified “step” usages as `training step` in the main outline.

---
pass 1 

## Appendix

experiment results for sokoban

4B 1 2 4 pipelines  sokoban 
![alt text](./fig/4b-sokoban-4gpu.png)
