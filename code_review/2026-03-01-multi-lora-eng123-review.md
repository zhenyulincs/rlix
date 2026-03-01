# Code Review Plan: ENG-123 SchedRL Extraction + Multi-LoRA Pipeline

**Date**: 2026-03-01
**Scope**:
- `schedrl/`: baseline `b0774f3` (inclusive) through HEAD — 31 commits (`b0774f3` is the first new commit; `git rev-list --count b0774f3^..HEAD -- schedrl/` = 31)
- `external/ROLL_schedrl`: baseline `88baa61b` (exclusive) through HEAD — 59 commits (`git rev-list --count 88baa61b..HEAD` = 59)
- **Total review targets**: 90 commits (31 `[S]` + 59 `[R]`)

**Plans being implemented**:
- `thoughts/shared/plans/2026-02-05-ENG-123-roll-multipipeline-extraction.md` — ENG-123: extract shared scheduler core into `schedrl/`
- `thoughts/shared/plans/2026-02-02-schedrl-multi-lora-adapter-extension.md` — port multi-LoRA pipeline from `ROLL_multi_lora` into `ROLL_schedrl`

---

## Background

### What ENG-123 built

Before this work, multi-pipeline time-sharing logic lived in a fork (`ROLL_multi_pipeline`). ENG-123 extracts the scheduler core into a standalone Ray-based library (`schedrl/`) and re-integrates it with upstream ROLL via a thin adapter layer.

Ownership boundary enforced by the plan:
- `schedrl/` owns: protocol/types, client, central scheduler Ray actor, planner, state
- `roll/schedrl_adapter/` owns: ROLL-specific mechanics (DP subset lifecycle, abort+ACK, progress mapping, GPU cluster management)

Key constraints that drive many design decisions:
- **Library Mode only** (job-owned scheduler, no shared Service Mode daemon) for ENG-123
- **`sleep_level=2`**: GPU release only; actors stay alive (no process teardown)
- **Timeouts from env vars only** — no SchedRL config fields for timeouts in ENG-123 (sentinel `-1` if present)
- **Shrink ordering is strict**: `suspend()` before clearing `active_dp_ranks` → drain in-flight → stop/offload → clear routing metadata → return success

### What the Multi-LoRA plan built

`SchedRLMultiLoraPipeline` trains multiple LoRA adapters **sequentially** (one at a time) under the SchedRL scheduler. Key constraints:
- No adapter GC — static VRAM budget, fixed adapter set
- `per_adapter` optimizer mode — dedicated optimizer state per adapter, no cross-adapter contamination
- Weight sync after each adapter's training step is **selective** — only the trained adapter's weights are broadcast to inference workers
- `lora_name` flows end-to-end: env manager → vLLM strategy → training step → model update

---

## Critical Invariants Checklist

These invariants are referenced across multiple groups. Check each once centrally, then spot-check per group.

- **INV-1 Strict shrink ordering**: `suspend()` → clear admission → abort/drain → ACK → offload/stop → clear routing → return. Violating this order causes requests to hit empty workers or GPUs to be reclaimed during weight loads. (Groups 2, 3, 4, 11)
- **INV-2 Abort ACK semantics**: ACK means "not in-flight" (safe for retry), not "successfully completed". (Groups 2, 3)
- **INV-3 Adapter lock ordering**: `_op_lock` serializes `resize_infer` and weight sync. `_resize_sync_lock` serializes resize and sync in the pipeline. These two locks must never be held in incompatible order — verify no call path acquires `_resize_sync_lock` then `_op_lock`. (Groups 7, 11)
- **INV-4 Selective adapter sync only**: After training adapter A, only A's weights are broadcast. Adapter B's resident weights must be untouched and uncorrupted. (Groups 4, 6, 7, 8)
- **INV-5 No mixed-adapter batch**: A single inference batch must never contain requests for two different `lora_name` values. Must be enforced at the vLLM strategy layer. (Groups 6, 8)
- **INV-6 No policy leakage into adapter**: `roll/schedrl_adapter/` must contain zero gap-ratio logic or priority assignments. All scheduling policy lives in `schedrl/scheduler/`. (Groups 2, 3, 7)
- **INV-7 Boundary enforcement**: `schedrl/` must have zero ROLL-specific imports (`from roll` / `import roll`). (All groups touching `schedrl/`)
- **INV-8 Timeouts from env vars only**: All timeouts in `schedrl/` must come from `_get_env_timeout_s`, never hardcoded literals. (Groups 1, 3, 4)

---

## How to Navigate

All commands below assume CWD = repo root (`/workspace/SchedRL`).

```bash
# For schedrl/ commits (main repo)
git show <hash>
git show <hash> --stat

# For ROLL_schedrl commits
git -C external/ROLL_schedrl show <hash>
git -C external/ROLL_schedrl show <hash> --stat
```

Commits within each group are listed **oldest → newest** (natural reading order).
`[S]` = main repo commits touching `schedrl/` — review with `git show <hash>`.
`[R]` = `external/ROLL_schedrl` commits — review with `git -C external/ROLL_schedrl show <hash>`.

---

## Review Workflow

### Step 0 — Workspace isolation
- Each agent runs in its assigned review worktree (detached, pinned to group end-state commit).
- Single-repo groups: one worktree pinned to last commit in `range_commits`.
- Mixed-repo groups: two worktrees — main repo (last `S:` commit) and `external/ROLL_schedrl` (last `R:` commit). Coordinator verifies both HEADs match `range_commits` before spawning agents.
- Main tree read-only. No commits, rebases, merges, or pushes.
- Report directory: `repo_review/output/group_{NN}/` (must be git-ignored; never included in merge-target commits)

### Step 0.1 — Freeze commit range
- Record ordered commit list, pipe-delimited with repo tags. Example: `S:d227f2b|S:e34693f|...|R:cc1a3fb|...|R:d338f06`
- `range_fingerprint = SHA256(canonical commit string)`
- If commits change before finalization → full rerun required.

### Step 1 — Verify commit counts
Run these two commands from repo root to confirm the review range matches the plan:
```bash
git rev-list --count b0774f3^..HEAD -- schedrl/
# Expected: 31

git -C external/ROLL_schedrl rev-list --count 88baa61b..HEAD
# Expected: 59
```

### Step 2 — Review by risk tier (not just chronology)
Groups are ordered by dependency for reading, but prioritize review effort by risk:

- **Tier 1 (highest risk — review first and most thoroughly)**:
  - Group 1 (schedrl/ foundation — core types + planner correctness)
  - Group 2 (shrink-to-zero + adapter bootstrap — INV-1 critical path)
  - Group 4 (NCCL teardown + expand order — correctness/OOM bugs)
  - Group 7 (multi-LoRA core pipeline — lock ordering, sequential guarantee)
  - Group 10 (contains `620bdea` correctness fix alongside tracing)
  - Group 11 (streaming OOM fix + late paired commits)

- **Tier 2 (medium risk)**:
  - Group 3 (API simplification — `CompletionSuspensionOp` removal)
  - Group 5 (multi-stream progress — state shape change)
  - Group 6 (per-adapter training — optimizer isolation)
  - Group 8 (lora_name rollout wiring — end-to-end flow)

- **Tier 3 (lowest risk — skim unless Tier 1/2 raises concerns)**:
  - Group 9 (tests + examples — config-only churn)

### Step 3 — Unit execution (iterative)
- List all units: mapped GATE/CHECK rules + AF + DoD. DEFER/EXCLUDE are excluded.
- Compute `depends_on` per unit using the **dependency mapping** (see below).
- Coordinator computes `topo_level` by topological sort.

**Per-level loop** (for each topo_level, ascending):
1. Run all units at this level whose dependencies are all resolved (terminal_state in PASS/HUMAN_RESOLVED/HUMAN_ACCEPTED_RISK). Skip units with any dependency that is REJECTED or itself dep-skipped (transitive closure — coordinator propagates mechanically). All skipped units are enqueued as DEP_SKIPPED in final review.
2. Wait for all running units to reach **stable verdict** (see Section G). Reruns happen within this step.
3. Coordinator validates each PASS verdict before auto-terminalization: if `trigger_flags` is non-empty but `escalated` is not `true` → override to SKIPPED (skip_reason: BAD_SCHEMA), route to rerun path. Then auto-set terminal for non-escalated PASS verdicts on non-JUDGMENT units. PASS on JUDGMENT units → queued as JUDGMENT_ITEM at checkpoint.
4. If any units remain unresolved after reruns (RERUN_EXHAUSTED, NEEDS_HUMAN, ESCALATED_PASS, JUDGMENT_ITEM) → coordinator queues those for **human checkpoint** (before proceeding to next level):
   - Human sets terminal state for each: HUMAN_RESOLVED, HUMAN_ACCEPTED_RISK, or REJECTED.
   - Must include `human_reason`, `resolved_by`, `resolved_at_utc`, `resolution_phase: checkpoint`.
   - Checkpoint actions are persisted as `phase: checkpoint` entries in `human_review_queue`.
   - Coordinator removes resolved checkpoint items from queue.
5. Once all run units at this level are resolved → proceed to next level. (Dep-skipped units are excluded from level progression; they are enqueued as DEP_SKIPPED in final review.)

- Each agent receives: unit constraint, relevant commits, group context, frozen range, and finalized findings from declared dependencies (read-only; all dependencies are fully resolved before downstream runs).
- All gating/rerun/terminal rules per Section G.

#### Dependency mapping (deterministic)
INV-* labels are not units — they expand to rule IDs:
- INV-1 → `[RULE-4.1, RULE-4.2, RULE-4.3, RULE-4.4, RULE-5.2, RULE-12.1, RULE-13.1]`
- INV-3 → `[RULE-26.1, RULE-26.2, RULE-48.1, RULE-48.2]`
- Other INV-x → rules they reference in code_audit_rules.md
- Duplicate rule families: use `RULE-52a.1`, `RULE-53a.1`, etc. (suffixed form, see Rule-ID Disambiguation block).

Rules:
1. GATE/CHECK rules → `depends_on: []`
2. AF items referencing commits → `depends_on: []`
3. AF items referencing invariants/rules → `depends_on: [expanded rule IDs mapped to this group]`
4. DoD items → `depends_on` referenced rule/AF units; no reference → `depends_on: []`
5. Cross-repo paired checks (mapped as AF units) → `depends_on` from a **frozen pair mapping** computed at run start: for each curated pair (per Section C), coordinator lists all GATE/CHECK rules in the group with `check_timing: DIFF` that touch files changed by either commit in the pair. This mapping is stored in `dependency_graph` and does not change during the run.
6. Fallback → `depends_on: []`
7. Cycle → coordinator fails with explicit error. No execution.

### Step 4 — Coordinator merge
- Collect latest unit findings. Validate:
  - Every run unit has exactly one finding; dep-skipped units (including transitive) have no finding (expected, tracked in `dep_skipped_units`)
  - `missing_units = expected - checked - dep_skipped` is empty
  - All dependencies of run units had `terminal_state` in (PASS, HUMAN_RESOLVED, HUMAN_ACCEPTED_RISK) before unit ran
  - All run units have a stable verdict (agent-side complete)
  - No checkpoint-phase items remain in queue
  - Exactly one `unit_status` row per `expected_units` ID (no duplicates, no unknown IDs)
- Build `human_review_queue` and set `preflight_status` (with `blocking_context` if BLOCKED)
- Produce merged group report

---

## Severity Rubric

Classify every finding using this scale so issues are comparable across reviewers:

- **P0 — Correctness / Safety / Data corruption**: Wrong results, data loss, security holes. Must block merge.
- **P1 — Likely regression or deadlock / race**: Not yet observed but structurally possible under concurrency or specific input. Should block merge unless explicitly accepted with justification.
- **P2 — Maintainability / Performance risk**: Hard-to-maintain patterns, unbounded memory growth, missing guards that could cause issues at scale. Fix before next milestone.
- **P3 — Docs / Nits**: Naming, comments, style, minor readability. Fix at convenience.

---

## Per-Commit Review Template

Copy this template for every reviewed commit. Fill every field — write "None" or "N/A" if nothing applies.

```
Commit: <7-char hash>
Group: <group number>
Intent: <one sentence — what the plan says this commit should accomplish>
What changed: <list each file with +/- line count and one-line description>
Intent vs plan: <does this commit match its stated plan goal? yes/no/partial — explain if not>
Invariants checked:
  - INV-X: pass / fail / N/A  (list each relevant invariant)
Regression risk: <does this commit change existing behavior? what could break?>
Findings:
  - [P0/P1/P2/P3] <concrete description with file:line reference>
Test evidence: <which test covers this? if none, note the gap>
Needed follow-up: <actions required before merge>
```

---

## Agent + Human Review Framework

### Section A: Check Types and Status Definitions

**Check timing**:
- `DIFF`: Check against each commit's diff individually. Can PASS/FAIL per commit.
- `END-STATE`: Check against the final codebase state after all group commits. Only evaluated once per group.

**Rule status**:
- `GATE`: Must pass for merge. FAIL blocks merge.
- `CHECK`: Informational. Agent reports, human decides severity.
- `DEFER`: Not applicable in this group (code doesn't exist yet). Agent skips.
- `UPDATED`: Constraint text revised — agent uses revised text. This is a modifier, not a standalone status. The `rule_status` remains GATE or CHECK; set `rule_updated: true` to indicate revised text.
- `EXCLUDE`: Obsolete or out-of-scope. Agent skips.

**Agent verdicts**:
- `PASS`: Check satisfied. Evidence provided.
- `FAIL`: Check violated. Evidence + severity provided.
- `NEEDS_HUMAN`: Agent cannot determine — requires human judgment.
- `SKIPPED`: Agent could not evaluate (e.g., file too large, code path unclear). Must include reason. Never silently omit a check.

**Item type** (for Audit Focus and DoD):
- `MECHANICAL`: Factual, verifiable by reading code or running commands. Agent handles.
- `JUDGMENT`: Requires design reasoning, plan alignment, or risk assessment. Agent flags as NEEDS_HUMAN.

### Section B: Agent Output Schemas

**Schema A — Per-unit agent output** (`{unit_id}_v{n}.yaml`):

```yaml
unit_id: "RULE-4.1"
unit_type: rule                        # rule | audit_focus | dod
topo_level: 0                         # computed by coordinator
depends_on: []
rerun_round: 0                        # 0 = initial, 1 = rerun
generated_at_utc: "2026-03-01T12:00:00Z"
range_fingerprint: "sha256:ab3f...c7d2"

# Target — one block, keyed by unit_type (no null fields)
target:
  rule_id: "4.1"
  rule_status: GATE                    # GATE | CHECK
  rule_updated: false
  check_timing: DIFF                   # DIFF | END-STATE
  # AF/DoD units use: { af_id: "AF-02-001", classification: MECHANICAL }
  # or:               { dod_id: "DOD-02-001", classification: JUDGMENT }

finding:
  verdict: PASS                        # PASS | FAIL | NEEDS_HUMAN | SKIPPED
  confidence: HIGH                     # HIGH | LOW
  anchor: "generate_scheduler.py@500e320:L412"
  trigger_flags: []                    # required list; values: LOW_CONFIDENCE | LOCK_CONCURRENCY | CROSS_REPO_PAIR; empty if none apply
  # --- PASS shape (non-escalated): verdict + confidence + anchor + trigger_flags only. Omit all other fields. ---
  # --- PASS shape (escalated): add escalated + escalation_reason (triggers ESCALATED_PASS in queue). ---
  # escalated: true
  # escalation_reason: "Lock-ordering rule — requires human review"
  # --- SKIPPED shape: add skip_reason (required): ---
  # skip_reason: TIMEOUT | BAD_SCHEMA | NO_ANCHOR | OTHER   # required when verdict: SKIPPED
  # --- Non-PASS shape: add all these fields (required): ---
  # risk_candidate: P0 | P1 | P2 | P3
  # summary: "..."
  # snippet: "..."
  # recommendation: "..."
  # escalated: true | false
  # escalation_reason: "..."           # required when escalated: true

# Required only for commits with findings or behavioral changes; optional for low-risk PASS-only commits
commit_annotations:
  - commit: "500e320"
    note: "7-step shrink ordering — INV-1 critical path"
```

**Schema B — Coordinator merged group report**:

```yaml
group: 2
group_name: "ROLL Adapter Bootstrap"
run_id: "G02-20260301T120000Z"        # unique per group run; inherited by queue items for reconciliation
range_fingerprint: "sha256:ab3f...c7d2"
range_commits: ["R:500e320", "R:33ef906", "R:c379a30", "R:21aad9c", "R:57a9caf", "R:eb70e07", "R:8c7c330"]
generated_at_utc: "2026-03-01T13:00:00Z"
dependency_graph: { ... }              # frozen at run start, stored for auditability

# Run health (top of report)
run_health:
  total_units: 17
  terminal: 15
  blocked: 0
  human_needed: 2

preflight_status: READY_FOR_HUMAN      # READY_FOR_HUMAN | BLOCKED
blocking_reasons: []                  # codes: FINGERPRINT_MISMATCH | MISSING_UNITS | INTEGRITY_VIOLATION | GATE_SKIPPED | RERUN_EXHAUSTED | CHECKPOINT_INCOMPLETE
# When BLOCKED, blocking_context lists affected unit_ids per code:
# blocking_context: { "RERUN_EXHAUSTED": ["RULE-4.1"], "CHECKPOINT_INCOMPLETE": ["AF-02-003"] }

# Single queue with phase tags
# Sort order per phase:
#   checkpoint: RERUN_EXHAUSTED > NEEDS_HUMAN > ESCALATED_PASS > JUDGMENT_ITEM, then risk_candidate, then subject_id
#   final_review: DEP_SKIPPED > COMMIT_GAP, then risk_candidate (P0 > P1 > P2 > P3 > null), then subject_id
human_review_queue:
  - { subject_type: unit, subject_id: "AF-02-005", reason: "DEP_SKIPPED", phase: "final_review", anchor: null, unit_file: null, blocked_by: "RULE-4.1", run_id: "G02-20260301T120000Z" }
  - { subject_type: commit, subject_id: "c379a30", reason: "COMMIT_GAP", phase: "final_review", anchor: null, unit_file: null, run_id: "G02-20260301T120000Z" }
  # phase: checkpoint | final_review
  #   checkpoint reasons: RERUN_EXHAUSTED, NEEDS_HUMAN, ESCALATED_PASS, JUDGMENT_ITEM (during Step 3 per-level loop; removed after resolution)
  #   final_review reasons: DEP_SKIPPED, COMMIT_GAP (after all levels complete)
  # subject_type: unit | commit
  # run_id: inherited from report-level run_id for reconciliation
  # risk_candidate: copied from finding.risk_candidate for non-PASS unit items; null for PASS/commit/dep-skipped items
  # anchor: copied from finding.anchor for unit items; null for commit/dep-skipped items (informational only)
  # unit_file: agent output filename (always latest version); null for dep-skipped/commit items
  # origin_verdict: verdict that triggered escalation (RERUN_EXHAUSTED only: FAIL or SKIPPED)
  # blocked_by: unit_id of REJECTED or dep-skipped dependency (DEP_SKIPPED only; nearest blocking ancestor)

# One status record per unit (merge of old terminal_states + resolutions)
unit_status:
  - unit_id: "RULE-4.1"
    terminal_state: PASS               # PASS | HUMAN_RESOLVED | HUMAN_ACCEPTED_RISK | REJECTED | null
    rerun_round: 0
    queue_reason: null                 # reason unit was escalated to human: RERUN_EXHAUSTED | NEEDS_HUMAN | ESCALATED_PASS | JUDGMENT_ITEM | DEP_SKIPPED; null for auto-terminal PASS
    blocked_by: null                   # unit_id of REJECTED or dep-skipped dependency (nearest blocking ancestor; DEP_SKIPPED only); persisted for audit
    human_reason: null                 # required when HUMAN_RESOLVED, HUMAN_ACCEPTED_RISK, or REJECTED
    resolved_by: null                  # human reviewer identifier; required when terminal_state is human-set
    resolved_at_utc: null              # timestamp; required when terminal_state is human-set
    resolution_phase: null             # checkpoint | final_review; required when terminal_state is human-set

coverage:
  expected_units: ["RULE-4.1", "AF-02-001", "DOD-02-001", "..."]
  checked_units:  ["RULE-4.1", "AF-02-001", "DOD-02-001", "..."]  # units with agent output (finding)
  dep_skipped_units: []               # units skipped due to REJECTED dependency (no finding, tracked via unit_status)
  missing_units:  []                  # = expected - checked - dep_skipped; must be empty for preflight PASS
  deferred_rules: []
  excluded_rules: ["RULE-53b.2"]
  integrity: { no_primary: [], dup_primary: [], dep_violations: [] }

commit_annotations:
  - commit: "500e320"
    notes: ["7-step shrink ordering — INV-1 critical path"]
    coordinator_generated: false

# Durable resolution record for COMMIT_GAP queue items (audit traceability)
commit_resolutions:
  - commit: "c379a30"
    resolution: "REVIEWED"             # REVIEWED | ACCEPTED_RISK
    human_reason: "Commit reviewed manually — no issues found"
    resolved_at_utc: "2026-03-01T14:30:00Z"
    reviewer: "eng123"                 # human reviewer identifier
```

**Commit annotation merge** (coordinator):
- Append-only: collect all notes per commit from all units. No semantic merge.
- Annotations required for commits with non-PASS findings or behavioral changes. Low-risk PASS-only commits: annotation optional — no `COMMIT_GAP` generated for these. High-risk PASS commits (escalated or JUDGMENT) should include annotation for human context, but absence does not trigger `COMMIT_GAP`.
- `COMMIT_GAP` triggered only when a commit has non-PASS findings from any unit but zero annotations. Coordinator generates stub with `coordinator_generated: true`.

### Section C: Escalation Rules

Auto-escalate to human review (regardless of agent verdict):
- Any finding with `confidence: LOW`
- Any finding on lock-ordering or routing-lock rules (26.1-26.2, 48.1-48.2) or INV-3 (adapter lock ordering) in any group
- Any finding touching a curated cross-repo pair (agent must cross-check consistency between both repos). Curated pairs (maintain this list when groups change):
  - Group 3: `d227f2b` [S] + `d338f06` [R] — resize_infer API introduction + ROLL adaptation
  - Group 3: `62bee8b` [S] + `d945a80` [R] — CompletionSuspensionOp removal + concurrent_pipeline refactor
  - Group 10: `4873ee5` [S] + `2da6eff` [R] — env var inheritance + tracing error handling
  - Group 11: `0c66eba` [S] + `a81b69f` [R] — late resize/lock fixes across both repos
- Any finding where `risk_candidate: P0` or `risk_candidate: P1` — even if verdict is PASS. Agent sets `risk_candidate` using the mapping below.

**Rule → Risk Class Mapping** (deterministic — both agents use this table):

P0 rules (data corruption, deadlock, NCCL hang):
- 4.1-4.4 (shrink-to-zero gate)
- 12.1-12.3 (suspend gate)
- 13.1-13.2 (shrink ordering — INV-1)
- 26.1-26.2 (routing lock — INV-3)
- 37.1-37.3 (NCCL destroy before barrier)
- 47.1-47.2 (GPU rollback)
- 48.1-48.2 (lock ordering)
- 57.1-57.2 (dp_rank overlap)

P1 rules (correctness regression, silent wrong results):
- 5.1-5.3 (abort ACK — INV-2)
- 6.1-6.4 (gap-ratio)
- 7.1-7.3 (multi-LoRA aggregation)
- 8.1-8.5 (sequential training / RNG / LoRA ID)
- 17.1-17.3 (allocation state)
- 19.1-19.3 (train/val shrink order)
- 24.1 (mutual exclusivity — UPDATED)
- 28.1-28.3 (phase ordering)
- 30.1-30.2 (commit ordering)
- 38.1-38.3 (forward version forbidden)
- 42.1 (source-type immutability)
- 45.1-45.2 (sender eligibility)

All other rules: `risk_candidate: null` (not a P0/P1 candidate).

Audit Focus and DoD items: agent sets `risk_candidate` based on which invariant they relate to. Items referencing INV-1 through INV-8 inherit the risk class of the corresponding rule family. Items with no invariant link: `risk_candidate: null`.

### Section D: Minimum Gate Set Per Group

Each group has a minimum gate set of its highest-risk GATE rules (up to 10). Groups with fewer GATE rules have proportionally smaller gate sets; Group 9 has no GATE rules (CHECK-only). Agent must check minimum gate set first and with highest rigor.

**Extended checks coverage rule**: Agent MUST check all mapped rules for the group (GATE and CHECK). If the agent cannot complete any rule check, it MUST report `verdict: SKIPPED` with reason — never silently omit.

**SKIPPED gating**:
- Any `SKIPPED` on any GATE rule → blocks group sign-off. Human must evaluate manually.
- Any `SKIPPED` on CHECK / Audit Focus / DoD items → does not block sign-off, but human must explicitly resolve (accept with reason, evaluate manually, or request agent re-run).

- **Group 1**: 6.2 (gap-ratio), 17.1 (allocation state), 28.1 (phase ordering), 30.1 (commit ordering), 47.1 (GPU rollback)
- **Group 2**: 4.1 (shrink-to-zero gate), 5.2 (ACK definition), 12.1 (suspend gate), 13.1 (shrink ordering), 26.1 (routing lock)
- **Group 3**: 24.1-UPDATED (mutual exclusivity)
- **Group 4**: 37.2 (NCCL destroy before barrier), 38.1 (forward version forbidden), 45.1 (sender eligibility)
- **Group 5**: 7.3 (multi-LoRA aggregation), 42.1 (source-type immutability)
- **Group 6**: 8.1 (sequential training), 8.2 (RNG isolation), 8.4 (optimizer validation)
- **Group 7**: 8.1 (sequential), 19.1 (train/val shrink order), 48.1 (lock ordering)
- **Group 8**: 8.5 (LoRA ID consistency), 11.3 (dual-write)
- **Group 9**: (no minimum gate — CHECK only)
- **Group 10**: 16.1 (trace context lifecycle), 57.1 (dp_rank overlap), 24.1-UPDATED
- **Group 11**: 48.1 (lock ordering — _resize_sync_lock vs _op_lock)

### Section E: Agent Prompt Template

```
You are reviewing Group {N}: {group name}.
Repo root: /workspace/SchedRL

## Commits
{commit list from this file}

## Context
{group context from this file}

## Source 1: Curated Audit Rules
Minimum gate set (check first, highest rigor):
  {rule_id}: {one-line constraint}
Extended checks:
  {rule_id}: {one-line constraint}
Full text: read code_review/code_audit_rules.md sections {X,Y,Z}
Note: Rules 52 and 53 have duplicate section numbers. See "Rules Not Applicable" section for line-number anchors to the correct duplicate.

## Source 2: Audit Focus Questions
{Audit Focus section from this file — already tagged MECHANICAL/JUDGMENT}

## Source 3: Definition of Done
{DoD section from this file — already tagged MECHANICAL/JUDGMENT}

## Output
Write `{unit_id}_v{n}.yaml` using Schema A.
- PASS findings: verdict, confidence, anchor, trigger_flags. Omit summary/snippet/recommendation.
- SKIPPED findings: verdict, confidence, anchor, skip_reason (required: TIMEOUT|BAD_SCHEMA|NO_ANCHOR|OTHER).
- Non-PASS findings: all fields required (summary, snippet, anchor, recommendation).
- Commit annotations: required for commits with non-PASS findings or behavioral changes. Optional for PASS-only commits.
- `trigger_flags`: required list. Set `LOW_CONFIDENCE` when confidence: LOW, `LOCK_CONCURRENCY` for lock/concurrency findings, `CROSS_REPO_PAIR` for curated pair findings. Empty list if none apply.
- Escalation: set `escalated: true` with reason when `trigger_flags` is non-empty. Coordinator enforces in Step 3 before auto-terminal: non-empty flags + missing escalation → SKIPPED (BAD_SCHEMA).
```

### Section F: Human Review Protocol

#### Coordinator preflight (automated)
All mechanical checks. If any fail → `preflight_status: BLOCKED`.
- range_fingerprint validation
- missing_units == [] (where missing = expected - checked - dep_skipped)
- exactly one `unit_status` row per `expected_units` ID; no duplicates, no unknown IDs (else INTEGRITY_VIOLATION)
- integrity lists (no_primary, dup_primary, dep_violations) all empty
- all run units have a stable verdict (agent-side complete, all reruns finished)
- all dep-skipped units are enqueued with reason `DEP_SKIPPED` in final_review
- no checkpoint-phase items remain in queue (all resolved during Step 3); if any remain → BLOCKED with code `CHECKPOINT_INCOMPLETE`
- SKIPPED on GATE with `terminal_state: null` → BLOCKED (resolved GATE_SKIPPED does not block)
- rerun budget enforcement (max 1 rerun per unit)
- build human_review_queue, set preflight_status; if BLOCKED, populate `blocking_context` with affected unit_ids per code

#### Final Human Review (queue-driven, after all levels complete)
1. If BLOCKED → resolve by `blocking_reasons` code:
   - `FINGERPRINT_MISMATCH` → verify commit range, trigger full rerun if changed
   - `MISSING_UNITS` → investigate missing agent outputs (dep-skipped units are excluded from this check); trigger rerun for truly missing units
   - `INTEGRITY_VIOLATION` → fix assignment/dependency errors, rerun affected units
   - `GATE_SKIPPED` → investigate root cause, trigger rerun or escalate
   - `RERUN_EXHAUSTED` → resolve units listed in `blocking_context["RERUN_EXHAUSTED"]`; set terminal state for each (safety net — normally resolved at checkpoint)
   - `CHECKPOINT_INCOMPLETE` → resolve units listed in `blocking_context["CHECKPOINT_INCOMPLETE"]`; re-enter checkpoint for these units
2. Walk `human_review_queue` (final_review items only) top-to-bottom. Act by reason code:
   - DEP_SKIPPED: unit never ran because dependency was REJECTED or itself dep-skipped (transitive); set terminal state (typically REJECTED with reason, or HUMAN_ACCEPTED_RISK if acceptable)
   - COMMIT_GAP: manually review commit, record resolution in `commit_resolutions`
   (Note: RERUN_EXHAUSTED/NEEDS_HUMAN/ESCALATED_PASS/JUDGMENT_ITEM are all handled at checkpoint during Step 3.)
3. Resolution by subject type:
   - **unit items**: set `terminal_state`, `human_reason`, `resolved_by`, `resolved_at_utc`, `resolution_phase: final_review` in `unit_status`
   - **commit items** (COMMIT_GAP): record in `commit_resolutions` (resolution + human_reason + resolved_at_utc + reviewer)
   Coordinator removes resolved items from queue.
4. Post-queue check before sign-off:
   - Queue is empty
   - All `unit_status` entries have non-null `terminal_state` (including units skipped due to REJECTED dependencies — human must explicitly set their terminal state)
   - If any `terminal_state: REJECTED` → group cannot be signed off clean; human must fix code + trigger full rerun, or change to `HUMAN_ACCEPTED_RISK`

### Section G: Gating & Rerun Protocol

**Execution**: Coordinator computes topo_level from dependency graph. All units at same level run in parallel. Between levels, human checkpoint resolves all non-PASS verdicts. Next level runs only after all dependencies have `terminal_state` in (PASS, HUMAN_RESOLVED, HUMAN_ACCEPTED_RISK). Units with REJECTED or dep-skipped dependencies are dep-skipped (transitive closure, computed mechanically by coordinator).

**Terminal states**: PASS, HUMAN_RESOLVED, HUMAN_ACCEPTED_RISK, REJECTED.

**Stable verdict**: A verdict becomes stable when no more agent-side changes are possible:
- PASS or NEEDS_HUMAN → stable immediately (no rerun for these)
- FAIL or SKIPPED → stable after rerun completes, or immediately if rerun budget exhausted

**Unblock rule**: Unit runs when all `depends_on` units have `terminal_state` in (PASS, HUMAN_RESOLVED, HUMAN_ACCEPTED_RISK). REJECTED or dep-skipped dependencies → unit is dep-skipped (transitive closure; coordinator propagates mechanically). Dep-skipped units are enqueued as DEP_SKIPPED in final review; human handles at sign-off. No provisional execution — all dependencies are fully resolved before downstream runs.

**Reruns**: Max 1 rerun for FAIL/SKIPPED. `rerun_round: 0` (initial) or `1` (rerun). After rerun, if verdict is still FAIL/SKIPPED → checkpoint with reason `RERUN_EXHAUSTED`. NEEDS_HUMAN → checkpoint directly (no rerun). New version file per rerun (`_v1`, `_v2`). Prior versions immutable. **Queue reason lifecycle**: `RERUN_EXHAUSTED`/`NEEDS_HUMAN`/`ESCALATED_PASS`/`JUDGMENT_ITEM` appear in checkpoint phase (resolved between levels). `DEP_SKIPPED`/`COMMIT_GAP` appear in final_review phase (after all levels complete).

**Mandatory escalation triggers**: Agent must set `escalated: true` when `trigger_flags` is non-empty. `trigger_flags` is a required structured list in Schema A: `LOW_CONFIDENCE` (confidence: LOW), `LOCK_CONCURRENCY` (lock-ordering or concurrency finding), `CROSS_REPO_PAIR` (curated cross-repo pair finding). Coordinator validates **in Step 3 before auto-terminal** (not at merge): if `trigger_flags` is non-empty but `escalated` is not `true` → SKIPPED with skip_reason `BAD_SCHEMA`, consumes rerun. This prevents incorrectly auto-terminalizing a PASS that should be escalated. No free-text detection — enforcement is purely from flags.

**Terminal-state ownership**: Coordinator is purely mechanical — only auto-sets PASS terminal for non-escalated PASS verdicts on non-JUDGMENT units. All other terminal states set by human. Escalated PASS → queued as ESCALATED_PASS at checkpoint. PASS on JUDGMENT-classified unit → queued as JUDGMENT_ITEM at checkpoint. Both pending until human reviews.

**Timeout**: 2700s (45 min). Timeout → SKIPPED, consumes rerun.

**No cascading reruns**: Upstream verdict change during rerun does not trigger downstream rerun — rerun produces the stable verdict. Downstream waits for dependency to reach terminal state (via checkpoint). No provisional execution means no invalidation complexity.

**Assignment**: Coordinator assigns once at run start. Lexicographic sort of unit_ids → monotonic run_seq. No reassignment during reruns.

**Capacity**: One global `max_concurrency` knob. Coordinator batches within levels if needed.

**Default exception paths** (all set `finding.skip_reason` in Schema A):
- Timeout → SKIPPED, skip_reason: TIMEOUT
- Schema validation failure → SKIPPED, skip_reason: BAD_SCHEMA
- Missing anchor → SKIPPED, skip_reason: NO_ANCHOR
- Missing mandatory escalation → SKIPPED, skip_reason: BAD_SCHEMA
All exceptions follow same rerun/escalation path.

### Section H: Pilot Run

Before full rollout, run the complete agent workflow on **Group 2** (Tier-1, 7 commits, ROLL adapter — good mix of GATE rules, mechanical DoD items, and judgment Audit Focus questions).

**Pilot acceptance criteria**:
- Agent produces valid output matching schema
- No false PASS on minimum gate set rules (verified by: human re-checks 100% of minimum-gate findings in pilot group)
- No SKIPPED on any GATE rule
- Executive summary is actionable (human can triage in <15 min)

### Rules Not Applicable to This Review

- 41.2: OUT-OF-SCOPE — SkyRL references
- 53b.2: OBSOLETE — `SchedRLTimeouts` deleted in Group 3
- Duplicate IDs disambiguated (code_audit_rules.md has two sections for each):
  - 52a = "Registration Policy Parity" (code_audit_rules.md line 1023)
  - 52b = "Client Initialization & Race Condition" (code_audit_rules.md line 1181)
  - 53a = "Cluster ID Grammar and Parser Safety" (code_audit_rules.md line 1045)
  - 53b = "Node Affinity & Timeouts" (code_audit_rules.md line 1193)

### Rule Updates

- **24.1**: Both-empty or both-non-empty raises `ValueError` (strict XOR). `SchedRLAdapter.resize_infer` and `SchedRLConcurrentPipeline.resize_infer` both enforce this. Note: `RolloutScheduler.shrink_sampler`/`expand_sampler` still delegate to `shrink_workers.remote()`/`expand_workers.remote()` (Rule 2.1 unchanged).

### Meta-rules

46.1-46.5 (Audit Prompt Rigor) apply to all agent findings.

---

## Group 1 — schedrl/ Library Foundation
**Repo**: `[S]` only | **Commits**: 10 | **All net-new code**

### Context

Everything in this group is written from scratch. There is no prior `schedrl/` implementation — the plan explicitly states "schedrl/ currently has no Python implementation". These commits build the full library skeleton: protocol contract, Ray actor skeleton, scheduler with gap-ratio planner, orchestrator, client, and launcher.

Reading order matters here: `b0774f3` defines the types that every subsequent commit depends on. Read them strictly oldest-to-newest.

### Commits

**`b0774f3`** `feat(schedrl/protocol): add protocol types, actions, validation and request_id helpers`
- `schedrl/protocol/types.py` — `ActionResponse`, `ProgressReport`, `SchedRLConfig`, etc.
- `schedrl/protocol/actions.py` — lifecycle action dataclasses: `RegisterPipeline`, `AdmitPipeline`, etc.
- `schedrl/protocol/adapter.py` — `Adapter` abstract base class
- `schedrl/protocol/request_id.py` — `build_request_id`, `parse_request_id`, `validate_request_id`
- `schedrl/protocol/validation.py` — pipeline config validation entrypoints
- `schedrl/__init__.py`, `schedrl/init.py`

**`dc7a5c0`** `feat(schedrl/client): add client connection logic with get-or-create semantics`
- `schedrl/client/client.py` — `connect(create_if_missing=True)` returning orchestrator handle; Library Mode race handling with backoff

**`c71beda`** `feat(schedrl/orchestrator): add orchestrator actor with RPC surface`
- `schedrl/orchestrator/orchestrator.py` — singleton Ray actor; RPCs: `register_pipeline`, `admit_pipeline`, `get_pipeline_state`, `monitor_pipelines`, `cleanup_pipeline`, `kill_pipeline`, `shutdown`; head-node affinity `soft=False`; zombie prevention

**`64394b9`** `feat(schedrl/scheduler): add scheduler actor skeleton`
- `schedrl/scheduler/scheduler.py` — Scheduler Ray actor, state management, run loop skeleton
- `schedrl/scheduler/state.py` — in-memory per-pipeline state
- `schedrl/scheduler/resource_manager.py` — GPU allocation tracking
- `schedrl/scheduler/executor.py`, `schedrl/scheduler/run.py` — **stubs only** (deleted in Group 3)

**`ab6bab5`** `feat(schedrl): add launcher utility and helper modules`
- `schedrl/launcher/launcher.py` — MPI-style Ray cluster lifecycle (rank 0 = head, others = workers)
- `schedrl/utils/ray_head.py` — head-node pinning strategy
- `schedrl/utils/timeouts.py` — `_get_env_timeout_s`, `timeout_context`, `_get_named_actor_with_timeout`

**`4110097`** `feat(schedrl/scheduler): add scheduler types and validation modules`
- `schedrl/scheduler/types.py` — `ClusterAllocation`, `ExecutionPlan`, `PendingRequest`, `Priority` enum, operation types (`CompletionSuspensionOp`, `SchedGuidedShrinkOp`, etc.)
- `schedrl/scheduler/validation.py` — 11 execution-plan validation conditions

**`f5e5691`** `feat(schedrl/scheduler): implement Phase 2 scheduling with gap-ratio planning`
- `schedrl/scheduler/scheduler.py` (major) — full scheduling cycle: completion → non-gen → gen → validate → execute; gap-ratio fairness algorithm; GPU request/release/release-and-request APIs; progress reporting with monotonic sequence; `notify_ready_to_release`
- `schedrl/scheduler/resource_manager.py`

**`3cb4169`** `feat(schedrl): add Priority enum and enhance resource manager`
- `schedrl/protocol/types.py` — `Priority` enum
- `schedrl/scheduler/resource_manager.py` — snapshot RPC, GPU discovery

**`6a92604`** `feat(schedrl/orchestrator): enhance orchestrator with topology and lifecycle RPCs`
- `schedrl/orchestrator/orchestrator.py` — `register_pipeline_topology`, `get_pipeline_namespace`, enhanced shutdown with source tracking

**`ac16b1f`** `feat(schedrl): harden scheduler integration`
- `schedrl/orchestrator/orchestrator.py`, `schedrl/protocol/validation.py`, `schedrl/scheduler/resource_manager.py`, `schedrl/scheduler/scheduler.py`
- Early `cluster_id` format validation; CPU-only reward device mapping enforcement; `actor_infer` GPU overlap allowed, others forbidden; `get_num_gpus()` for init; improved `kill_pipeline` cleanup logging

### Curated Audit Rules

**Minimum gate set** (check first, highest rigor):
- 6.2 (gap-ratio formula) `GATE DIFF` — gap-ratio must avoid division-by-zero on empty queues
- 17.1 (allocation state) `GATE DIFF` — ClusterAllocation must track GPU IDs atomically
- 28.1 (phase ordering) `GATE DIFF` — completion → non-gen → gen → validate → execute
- 30.1 (commit ordering) `GATE DIFF` — state commit must be atomic per scheduling cycle
- 47.1 (GPU rollback) `GATE DIFF` — failed allocation must roll back fully

**Extended GATE rules**: 1.1-1.3, 6.1, 6.3-6.4, 11.1-11.2, 15.1-15.3, 17.2-17.3, 18.1-18.4, 22.1-22.2, 23.1-23.3, 27.1-27.2, 28.2-28.3, 29.1-29.3, 30.2, 31.1-31.2, 43.1-43.2, 47.2, 48.1-48.2, 49.1-49.2, 50.1-50.2, 52b.1-52b.2, 53b.1, 54.1-54.3, 56.1-56.2, 57.1-57.2, 58.1-58.3, 59.1-59.2 (all `DIFF`)

**CHECK rules**: 33.1-33.2, 34.1-34.3, 35.1-35.3, 36.1, 51.1-51.2, 52a.1-52a.3, 53a.1-53a.3, 55.1-55.3

### Audit Focus

- [JUDGMENT] **`b0774f3` (types)**: Are type names domain-specific and self-explanatory? Is `request_id` deterministic — same inputs always produce same ID? Are any `Any` types present that should be concrete? Are all ENG-123 plan types represented?
- [MECHANICAL] **`f5e5691` (planner — largest commit)**: Does the phase ordering (completion → non-gen → gen → validate → execute) match ENG-123 Phase 2 spec exactly? Is the gap-ratio formula correct — no division-by-zero on empty queues? Is FIFO enforced via monotonic sequence (not `time.time()`)? Are the listed bug fixes (GPU leak, state corruption, duplicate cluster_id, completion/shrink conflict) each addressed with a clear comment explaining what the bug was?
- [MECHANICAL] **`ac16b1f` (hardening)**: Is the `actor_infer` GPU overlap allowance clearly documented with a comment explaining why it is permitted when other overlaps are forbidden?
- [MECHANICAL] **Across all**: Zero ROLL-specific imports anywhere in `schedrl/`. All timeout reads sourced from env vars (`_get_env_timeout_s`), never hardcoded. Singleton enforcement in orchestrator is race-safe (two concurrent callers). `kill_pipeline` leaves no dangling state.

### Definition of Done
- [ ] [MECHANICAL] INV-7 confirmed: `rg --glob '*.py' "from roll|import roll" schedrl/` returns zero matches
- [ ] [MECHANICAL] INV-8 confirmed: `rg -n --glob '*.py' "timeout.*=\s*[0-9]+" schedrl/` reviewed — all timeouts sourced from env vars
- [ ] [MECHANICAL] Gap-ratio formula in `f5e5691` checked for division-by-zero on empty queues
- [ ] [MECHANICAL] All 10 commits reviewed with per-commit template filled
- [ ] [MECHANICAL] Findings logged with P0-P3 severity

---

## Group 2 — ROLL Adapter Bootstrap
**Repo**: `[R]` only | **Commits**: 7

### Context

The first ROLL-side wiring to `schedrl/`. This group introduces three major things simultaneously:
1. **Shrink-to-zero** in ROLL's `generate_scheduler` — a previously unsupported lifecycle state (`active_dp_ranks = {}`)
2. The `SchedRLAdapter` Ray actor and `SchedRLConcurrentPipeline` — the first concrete implementation of the `Adapter` contract
3. **Pipeline-scoped isolation** — each pipeline gets its own Ray namespace, named actor prefix, and port space

The shrink-to-zero ordering is the most safety-critical part of this group. The ENG-123 plan mandates a strict 7-step sequence:
1. `suspend()` (sets `need_suspend=True`)
2. Clear admission (block new requests)
3. Abort/drain in-flight requests
4. Wait for ACK
5. Offload/stop servers for all DP ranks
6. Clear routing metadata (`active_dp_ranks`, `src_rank2_dp_rank`)
7. Return success with `shrunk_to_zero` signal

Violating this order can cause new requests to hit an empty worker set, or GPUs to be reclaimed while weights are still loaded.

### Commits

**`500e320`** `feat(multipipeline): progress + shrink-to-zero`
- `roll/distributed/scheduler/generate_scheduler.py` (+228) — atomic shrink/expand sequencing with locks; allows `active_dp_ranks={}`; shrink-to-zero path
- `roll/distributed/scheduler/rollout_scheduler.py` (+174) — emits `ProgressReport` from `GroupQueueManager.put()` (train path only)
- `roll/distributed/strategy/vllm_strategy.py` — removes colocated gating for vLLM offload
- `roll/pipeline/agentic/env_manager/traj_env_manager.py`, `vl_traj_env_manager.py` — preserves canonical request metadata
- `roll/utils/context_managers.py`, `roll/utils/functionals.py`

**`33ef906`** `feat(schedrl): add ROLL adapter entrypoint`
- `roll/schedrl_adapter/adapter.py` (new, +313) — `SchedRLAdapter` Ray actor; driver scripts call this to register/admit pipelines
- `roll/schedrl_adapter/concurrent_pipeline.py` (new, +471) — `SchedRLConcurrentPipeline` running under per-pipeline `runtime_env`

**`c379a30`** `fix(config): avoid eval in worker config`
- `roll/configs/worker_config.py` — replaces `eval()` with `ast.literal_eval()` for structured config parsing

**`21aad9c`** `feat(roll): implement resize_infer and multi-pipeline support`
- `roll/distributed/scheduler/generate_scheduler.py` (+113) — unified `resize_infer(dp_ranks_to_remove, dp_ranks_to_add)` replaces `shrink_workers`/`expand_workers`; abort+retry semantics with proper ACK handling
- `roll/distributed/executor/worker.py` (+32) — topology validation hooks
- `roll/distributed/scheduler/async_generate_scheduler.py`, `roll/distributed/scheduler/initialize.py` — pipeline-scoped actor naming with `PIPELINE_ID` prefix; per-pipeline namespace via `ROLL_RAY_NAMESPACE`; `SCHEDRL_CONTROL_PLANE` guard prevents `ray.shutdown()` in library mode
- New example files: `examples/multi_pipeline/`

**`57a9caf`** `refactor(schedrl_adapter): simplify adapter API and static cluster GPU mgmt`
- `roll/schedrl_adapter/adapter.py` — removes `_require_ray()` pattern; merges `ensure_coordinator + start_pipeline` into `create_coordinator`; `resize_infer` returns `ActionResponse`; sets `max_concurrency=1000`
- `roll/schedrl_adapter/concurrent_pipeline.py` — renames `_SchedRLAgenticPipeline` to `SchedRLConcurrentPipeline`; adds GPU request/release for static clusters (`actor_train`, `actor_infer`, `critic`, `reference`); reference model offload during shrink/expand

**`eb70e07`** `feat(roll): propagate SchedRL env vars via runtime_env for Ray actors`
- `roll/utils/constants.py` (+25) — `schedrl_env_vars()` helper
- `roll/distributed/scheduler/generate_scheduler.py`, `async_generate_scheduler.py`, `log_monitor.py`, `rollout_scheduler.py` — pass `runtime_env` with SchedRL env vars to all named actors
- `roll/pipeline/agentic/agentic_pipeline.py`, `env/deepeyes/env.py`, `env/gem/math_env.py` — `ROLL_RAY_NAMESPACE` from env (not hardcoded)

**`8c7c330`** `fix(roll): resource manager GPU placement and CPU platform compatibility`
- `roll/distributed/scheduler/resource_manager.py` — uses Ray `'GPU'` resource key when `num_gpus_per_node > 0`; consistent `ray_device_key` in placement group bundles
- `roll/platforms/cpu.py` — adds `CUDA_VISIBLE_DEVICES` and `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES` to `CpuPlatform`
- `roll/utils/env_action_limiter.py` — passes `runtime_env` to `GlobalLimiter`

### Curated Audit Rules

**Minimum gate set** (check first, highest rigor):
- 4.1 (shrink-to-zero gate) `GATE DIFF` — active_dp_ranks={} must be legal
- 5.2 (ACK definition) `GATE DIFF` — ACK means "not in-flight", not "completed"
- 12.1 (suspend gate) `GATE DIFF` — suspend() before clearing routing metadata
- 13.1 (shrink ordering) `GATE DIFF` — 7-step shrink sequence must be exact
- 26.1 (routing lock) `GATE DIFF` — atomic lock covers entire shrink sequence

**Extended GATE rules**: 2.1↑, 2.2, 4.2-4.4, 5.1, 5.3, 9.1-9.3, 12.2-12.3, 13.2, 14.1-14.2, 24.2, 26.2 (all `DIFF`)

**CHECK rules**: 10.1-10.3, 20.1-20.3

(↑ = UPDATED rule text, see Rule Updates section above)

### Audit Focus

- [MECHANICAL] **`500e320` (shrink-to-zero — highest scrutiny)**: Verify the 7-step shrink ordering is implemented exactly. `suspend()` (setting `need_suspend=True`) must be called **before** any routing metadata is cleared. Confirm `active_dp_ranks={}` no longer raises. Verify the atomic lock covers the entire shrink sequence — no partial shrink is visible to incoming requests.
- [JUDGMENT] **`33ef906` (adapter entrypoint)**: Does `SchedRLAdapter` contain only framework mechanics — zero scheduler policy? Does `SchedRLConcurrentPipeline` correctly acquire/release GPU clusters for all four cluster types (`actor_train`, `actor_infer`, `critic`, `reference`)? Are there races between acquire and release across pipelines?
- [MECHANICAL] **`c379a30`**: Confirm no remaining `eval()` on user-controlled input anywhere in `worker_config.py`.
- [JUDGMENT] **`21aad9c` (resize_infer)**: Is the abort ACK semantics correct — ACK must mean "not in-flight" (safe for retry) rather than "successfully completed"?
- [MECHANICAL] **`21aad9c` (resize_infer)**: Is `PIPELINE_ID` prefix applied consistently to **all** named Ray actors (not just some)?
- [JUDGMENT] **`57a9caf`**: `max_concurrency=1000` is set here — is this justified? (It is reduced to 32 in `d945a80`, Group 3.) Does static cluster GPU management account for all clusters holding GPU memory?
- [MECHANICAL] **`eb70e07`**: Does `schedrl_env_vars()` capture all required vars (`PIPELINE_ID`, `SCHEDRL_CONTROL_PLANE`, `ROLL_RAY_NAMESPACE`, etc.)? Are there any actors that still use hardcoded values after this commit?
- [JUDGMENT] **`8c7c330`**: Does the `CpuPlatform` env var addition risk masking GPU errors on nodes that do have GPUs?

### Definition of Done
- [ ] [MECHANICAL] INV-1 confirmed: 7-step shrink ordering verified in `500e320` (`generate_scheduler.py`)
- [ ] [MECHANICAL] INV-2 confirmed: abort ACK semantics verified in `21aad9c` — ACK means "not in-flight"
- [ ] [MECHANICAL] INV-6 confirmed: `SchedRLAdapter` contains zero scheduler policy (no gap-ratio, no Priority)
- [ ] [MECHANICAL] `PIPELINE_ID` prefix applied to all named Ray actors (grep `21aad9c` for consistency)
- [ ] [MECHANICAL] No remaining `eval()` on user input in `worker_config.py` after `c379a30`
- [ ] [MECHANICAL] All 7 commits reviewed with per-commit template filled
- [ ] [MECHANICAL] Findings logged with P0-P3 severity

---

## Group 3 — API Simplification + First Integration Cleanup
**Repo**: `[S]` 7 commits + `[R]` 4 commits | **Total**: 11

### Context

After Group 1-2 landed, the Adapter contract was still broad (many abstract methods). This group simplifies it down to a single `resize_infer(dp_ranks_to_remove, dp_ranks_to_add)` RPC, removing all other abstract methods and associated types. The rationale: admission gating, progress reporting, abort, and release are all handled internally by the ROLL adapter — the scheduler only needs to say "change your DP ranks".

The most significant design change in this group is **`62bee8b`**: removal of completion-driven suspension. Previously the scheduler had two shrink mechanisms (`CompletionSuspensionOp` and `SchedGuidedShrinkOp`). After this commit, `sched_guided_shrink_ops` is the sole mechanism. This simplifies the state machine but requires careful verification that no safety guarantees were dropped.

The `[R]` side (`d945a80`) is the largest single change in this group — a major `concurrent_pipeline` refactor that also fixes vLLM v1 compatibility and reduces `max_concurrency` from 1000 to 32.

### Commits

**`[S] d227f2b`** `feat(schedrl): add topology validation and resize_infer API`
- `schedrl/orchestrator/orchestrator.py` — `_validate_and_canonicalize_device_mapping()` with TP group contiguity and node boundary checks
- `schedrl/scheduler/resource_manager.py` — `init_topology()` for `required_gpus_per_node` enforcement
- `schedrl/scheduler/scheduler.py` — replaces `shrink_workers/expand_workers` with `resize_infer()`; fork-aligned demand via `metrics[remaining]`; `min_bundles=0` for shrink-to-zero

**`[S] e34693f`** `refactor(schedrl/protocol): simplify Adapter interface to resize_infer`
- `schedrl/protocol/adapter.py` (shrinks from 62 to ~15 lines) — removes `close_admission`, `open_admission`, `shrink_workers`, `expand_workers`
- Deletes `schedrl/scheduler/executor.py`, `schedrl/scheduler/run.py` (were stubs from `64394b9`)

**`[S] 4ac8c65`** `refactor(schedrl/protocol): simplify Adapter interface to resize_infer only`
- `schedrl/protocol/adapter.py` — removes remaining abstract methods: `get_pipeline_id`, `report_progress`, `abort_requests`, `wait_abort_ack`, `release_gpus`, `request_gpus`, `get_state_snapshot`
- `schedrl/protocol/request_id.py` — removes `ParsedRequestId`, returns tuple directly
- `schedrl/protocol/types.py` — removes `ModelMode`, `PipelineId`, `ClusterId`, `AdapterId`, `PlatformConfig`, `ReleaseReport`, `ReleaseAck`, `SchedRLTimeouts`, `SchedRLConfig`, `RayNamespaceContract`
- `schedrl/protocol/validation.py` — removes `validate_request_ids`, `validate_optional_timeout_s`

**`[S] bd80618`** `refactor(schedrl/scheduler): remove _require_ray pattern and cleanup types`
- `schedrl/scheduler/resource_manager.py`, `scheduler.py`, `types.py` — removes `_require_ray()`, unused types (`SimulatedState`, `has_operations`); simplifies `ValidationError`; allows `percent_completed > 1.0` (overshoot tolerated)

**`[S] 5f6a1ab`** `refactor(schedrl): remove _require_ray pattern from client, orchestrator, utils`
- `schedrl/client/client.py`, `orchestrator/orchestrator.py`, `utils/ray_head.py`, `utils/timeouts.py` — removes `_require_ray()` everywhere; removes `HeadNodeInfo` and `get_head_node_info`; adds FIXME comment about brittle Ray internal API

**`[S] a82f7e7`** `refactor(schedrl): improve ray CLI resolution and make GPU topology optional`
- `schedrl/launcher/launcher.py` — `ray_cli_path()` prefers PATH over `sys.executable`; fixes Ray filter predicate syntax (`=` not `==`)
- `schedrl/orchestrator/orchestrator.py` — `SCHEDRL_REQUIRED_GPUS_PER_NODE` now optional with auto-detection fallback
- `schedrl/utils/ray_head.py`

**`[S] 62bee8b`** `refactor(schedrl): remove completion-driven suspension and simplify notify_ready_to_release`
- `schedrl/scheduler/scheduler.py` — removes `notify_completion()`, `PendingCompletionRequest`, `CompletionSuspensionOp` from `ExecutionPlan`; `notify_ready_to_release` derives `dp_ranks_to_remove` from active allocation state (no longer takes `planned_release_gpu_ids`); fixes shutdown to use `ray.shutdown()` instead of subprocess

**`[R] cc1a3fb`** `feat(roll): improve bucket cache for multiprocess-safe selective sync`
- `roll/distributed/strategy/megatron_strategy.py` — caches bucket as raw bytes + metadata (not pickled tensors) to avoid torch multiprocessing reduction issues with vLLM v1 workers; promotes active checkpoint after building bucket cache
- `roll/third_party/megatron/model_update.py`

**`[R] d945a80`** `feat(roll): major SchedRL concurrent_pipeline refactor and vLLM compatibility`
- `roll/schedrl_adapter/concurrent_pipeline.py` (+360) — explicit `initialize_pipeline()` for lazy init; `build_latest_bucket_cache` wrapper; reduces `max_concurrency` 1000 → 32
- `roll/schedrl_adapter/adapter.py` — adds `PYTHONPATH` to pipeline env vars
- `roll/distributed/executor/cluster.py`, `worker.py` — thread limit env vars
- `roll/third_party/vllm/vllm_0_8_4/__init__.py` (+63) — patches vLLM v1 `_dummy_run` for `numpy.int64` tensor indexing; adds `bucket_bytes` format support in `update_parameter_in_bucket`

**`[R] 5788b2f`** `chore(roll): update example configs and requirements for smoke testing`
- Example YAMLs, `start_multi_pipeline_test.py`, `requirements_common.txt`, `requirements_torch260_vllm.txt` — reduces GPU/TP sizes for single-node; adds `VLLM_USE_V1=1`; relaxes Ray version pin

**`[R] d338f06`** `feat(roll): adapt to simplified SchedRL API with state verification`
- `roll/schedrl_adapter/concurrent_pipeline.py`, `adapter.py` — removes `planned_release_gpu_ids` from `notify_ready_to_release`; adds `release_and_request_static_cluster` for atomic train→critic GPU handoff; adds `RollResourceManagerProxy` singleton for shared placement groups; adds state verification after shrink/expand; `get_active_dp_ranks()` for post-shrink verification; coordinator scheduled in node-0 PG bundle with `num_gpus=0.01`

### Curated Audit Rules

**Minimum gate set** (check first, highest rigor):
- 24.1↑ (mutual exclusivity) `GATE END-STATE` — same dp_rank in both removes+adds → RuntimeError; different dp_ranks is legal

**Extended GATE rules**: (none beyond minimum gate)

**CHECK rules**: 29.1-29.3, 32.1-32.2, 52a.1-52a.3

**EXCLUDE**: 53b.2 (OBSOLETE — `SchedRLTimeouts` deleted in this group)

(↑ = UPDATED rule text, see Rule Updates section above)

### Audit Focus

- [MECHANICAL] **`4ac8c65` (breaking removal)**: Are any of the removed types still imported anywhere in `schedrl/` or `ROLL_schedrl`? Grep for `ModelMode`, `PipelineId`, `ReleaseReport` etc. to confirm clean removal.
- [JUDGMENT] **`62bee8b` (design change — high scrutiny)**: What safety guarantee did `CompletionSuspensionOp` provide? `sched_guided_shrink_ops` must cover all cases the old completion path handled. Is the simplified `notify_ready_to_release` (releases all active dp_ranks) safe — can it ever release ranks that another pipeline's scheduler assigned?
- [JUDGMENT] **`bd80618`**: Why is `percent_completed > 1.0` tolerated (overshoot)? Is this a design decision about progress semantics, or a workaround for a reporting inaccuracy? Should it log a warning?
- [JUDGMENT] **`5f6a1ab`**: What is the brittle Ray internal API mentioned in the FIXME? Is it documented clearly enough for a future maintainer to know what to fix?
- [JUDGMENT] **`d945a80` (major refactor)**: Why was `max_concurrency=1000` too high? Is 32 correct or a conservative guess? Does lazy `initialize_pipeline()` guarantee no requests arrive before initialization completes?
- [JUDGMENT] **`d338f06`**: Is `RollResourceManagerProxy` singleton safe when two pipelines initialize concurrently? Does the `num_gpus=0.01` trick for coordinator placement have any side effects on GPU scheduling?
- [JUDGMENT] **`cc1a3fb`**: Is the checkpoint promotion (promote before next expand/broadcast) ordering correct — can a stale checkpoint be promoted if the current one hasn't been committed?

### Definition of Done
- [ ] [MECHANICAL] Removed types (`ModelMode`, `PipelineId`, `ReleaseReport`, etc.) confirmed absent from all `schedrl/` and `ROLL_schedrl` imports
- [ ] [JUDGMENT] `CompletionSuspensionOp` removal in `62bee8b` verified — `sched_guided_shrink_ops` covers all prior safety guarantees
- [ ] [JUDGMENT] `RollResourceManagerProxy` singleton verified thread-safe under concurrent pipeline init
- [ ] [MECHANICAL] All 11 commits reviewed with per-commit template filled
- [ ] [MECHANICAL] Findings logged with P0-P3 severity

---

## Group 4 — Model Update: Selective Sync + Expand Order Fix
**Repo**: `[R]` only | **Commits**: 7

### Context

This group fixes two major correctness/performance issues discovered during integration testing:

1. **Selective NCCL sync** (`451bd31`): Instead of broadcasting all model weights after every training step, the model update service now uses a `comm_plan` to sync only the changed adapter's weights. It allocates a dedicated NCCL group per PP rank, and routes colocated targets (same physical GPU) through IPC instead. The NCCL group is destroyed **inside** the sync call before `dist.barrier()` — this is the teardown bug fix, avoiding a race where the group was destroyed while barrier was still running.

2. **Wrong expand step ordering** (`4d8e4b4`): Previously `load_states_partial` ran before `sync_selected_workers`, which held KV cache allocations during weight sync and caused transient OOM. The fix reorders to: `sync_selected_workers → process_weights_after_loading → load_states_partial`. Non-SchedRL path is unchanged.

### Commits

**`95e521c`** `feat(collective): add timeout_s, fail-fast KeyError, and teardown helper`
- `roll/utils/collective/collective.py` — `init_collective_group` and `create_collective_group` accept `timeout_s`; `get_group_by_name`/`destroy_collective_group` raise `KeyError` instead of logging silently; adds `teardown_collective_groups()` to `InferenceStrategy`

**`451bd31`** `feat(model_update): comm_plan-based selective sync with NCCL teardown fix`
- `roll/schedrl_adapter/model_update_service.py` (major rewrite) — `sync_selected_workers` selects one sender per PP rank (`dp_rank==0, tp_rank==0, cp_rank==0`) via `_select_sender_ranks_by_pp()`; `_build_comm_plan_for_sender()` allocates dedicated NCCL group per PP rank excluding colocated targets; groups destroyed inside `selective_sync_active_cache` **before** `dist.barrier()`
- `roll/third_party/megatron/model_update.py`

**`6670bab`** `feat(cluster): add resolve_topology flag to skip blocking ray.get in async actors`
- `roll/distributed/executor/cluster.py` — `Cluster.__init__` accepts `resolve_topology=True` (default); `False` skips all `ray.get()` calls and topology resolution; required for `RolloutScheduler` (async Ray actor) to avoid blocking event loop during `__init__`

**`4d8e4b4`** `feat(scheduler): non-blocking init, local PG allocation, SchedRL expand order`
- `roll/distributed/scheduler/generate_scheduler.py` — reorders expand steps in SchedRL mode: `sync_selected_workers → process_weights_after_loading → load_states_partial`; adds per-request dispatch logging; adds slow-request warning (≥30s)
- `roll/distributed/scheduler/resource_manager.py` — `RollResourceManagerProxy` local PG allocation

**`f4a24cd`** `fix(pipeline): re-offload actor_train after checkpoint to prevent GPU residual OOM`
- `roll/schedrl_adapter/concurrent_pipeline.py`
- Root cause: `megatron_strategy.save_checkpoint()` calls `load_states()` internally but never calls `offload_states()`. Fix: `defer_actor_train_release_for_checkpoint` ensures `do_checkpoint()` runs first, then `offload_states()`, then GPU release.

**`09841f3`** `fix(misc): sync resize_infer, asyncio fixes, request tracing logs, config updates`
- `roll/schedrl_adapter/adapter.py` — `resize_infer` changed from `async` to `sync` (`ray.get` instead of `asyncio.wrap_future`)
- `roll/pipeline/agentic/environment_worker.py` — uses `asyncio.get_running_loop()` instead of deprecated `get_event_loop()`; guards `ThreadPoolExecutor` against `max_workers=0`; `pool.shutdown(wait=False)`
- `roll/distributed/scheduler/generate_scheduler.py`, `rollout_scheduler.py` — request tracing logs
- `roll/pipeline/agentic/agentic_pipeline.py` — config updates

**`880493a`** `refactor(schedrl): move notify_ready_to_release to end of pipeline loop`
- `roll/schedrl_adapter/concurrent_pipeline.py` — removes per-step `notify_ready_to_release`; performs single final release at end of pipeline loop (aligns with `ROLL_multi_pipeline` pattern)

### Curated Audit Rules

**Minimum gate set** (check first, highest rigor):
- 37.2 (NCCL destroy before barrier) `GATE DIFF` — NCCL group destroyed inside sync, before dist.barrier()
- 38.1 (forward version forbidden) `GATE DIFF` — no forward-incompatible changes to NCCL group semantics
- 45.1 (sender eligibility) `GATE DIFF` — sender must be dp_rank==0, tp_rank==0, cp_rank==0

**Extended GATE rules**: 3.1-3.3, 37.1, 37.3, 38.2-38.3, 39.1-39.3, 40.1-40.2, 41.1, 45.2 (all `DIFF`)

**CHECK rules**: 44.1, 58.1-58.3

### Audit Focus

- [JUDGMENT] **`451bd31` (NCCL teardown fix — high scrutiny)**: What was the original bug with `finally`-block teardown? The old code destroyed NCCL groups in a `finally` block, which ran while `dist.barrier()` might still be executing on other ranks, causing undefined behavior. Verify the new teardown (inside `selective_sync_active_cache`, before `dist.barrier()`) cannot race with any other group operation. Does colocated target exclusion (IPC path) correctly identify when two ranks share a physical GPU?
- [MECHANICAL] **`95e521c`**: All timeout reads must come from env vars — verify `timeout_s` in `init_collective_group` is read via `_get_env_timeout_s` (not hardcoded). Does the fail-fast `KeyError` on missing group break any caller that previously silently tolerated a missing group?
- [JUDGMENT] **`6670bab`**: Is `resolve_topology=False` safe for `RolloutScheduler` — does topology info ever get populated later, or is it genuinely not needed?
- [MECHANICAL] **`4d8e4b4` (expand ordering — high scrutiny)**: Confirm the non-SchedRL single-pipeline path is unchanged (only the `if SCHEDRL_CONTROL_PLANE == 'schedrl':` branch is changed). Is the OOM root cause fully addressed — are there other code paths that call `load_states_partial` before sync?
- [JUDGMENT] **`f4a24cd`**: Compare with `10ec933` in Group 11 — both fix checkpoint-related OOM. Are they complementary (different code paths) or redundant (one supersedes the other)?
- [JUDGMENT] **`880493a`**: Does the scheduler correctly handle a pipeline that holds GPUs for an entire training loop and only calls `notify_ready_to_release` once at the end? Was the per-step release causing any correctness issue, or just extra overhead?

### Definition of Done
- [ ] [MECHANICAL] INV-4 confirmed: NCCL teardown in `451bd31` happens inside `selective_sync_active_cache` before `dist.barrier()`
- [ ] [MECHANICAL] Expand order in `4d8e4b4` verified: `sync_selected_workers → process_weights_after_loading → load_states_partial`; non-SchedRL path unchanged
- [ ] [MECHANICAL] INV-8 confirmed: `timeout_s` in `init_collective_group` sourced from env vars
- [ ] [JUDGMENT] `f4a24cd` and `10ec933` (Group 11) confirmed as addressing different code paths (no conflict/redundancy)
- [ ] [MECHANICAL] All 7 commits reviewed with per-commit template filled
- [ ] [MECHANICAL] Findings logged with P0-P3 severity

---

## Group 5 — Multi-stream Progress Tracking
**Repo**: `[S]` only | **Commits**: 2

### Context

The gap-ratio planner in Group 1 only tracked one progress stream per pipeline. For multi-LoRA, a pipeline trains multiple adapters sequentially — each adapter is a separate "stream" with its own `remaining` / `required` metrics. Without per-stream tracking, the planner would see incorrect aggregate progress and make wrong scheduling decisions.

These two commits extend the scheduler state to support nested progress tracking and add `lora_name` to allocation types as groundwork for GPU trace labels (used in Group 10).

### Commits

**`8250c77`** `feat(scheduler): multi-stream progress tracking and background rebalance`
- `schedrl/scheduler/state.py` — changes `latest_progress_by_pipeline` from flat dict to nested `[pipeline_id][mode][stream_key]`
- `schedrl/scheduler/scheduler.py` — `report_progress` routes into nested store keyed by `mode + stream_key`; enforces source-type invariant (full-finetune vs adapter cannot mix per pipeline); adds `_iter_pipeline_reports_locked`, `_pipeline_progress_totals_locked`, `_has_waiting_requests_locked`

**`0fabbd3`** `feat(scheduler): add lora_name field to ClusterAllocation and PendingRequest for GPU tracing`
- `schedrl/scheduler/types.py` — adds `lora_name: Optional[str] = None` to `ClusterAllocation` and `PendingRequest`

### Curated Audit Rules

**Minimum gate set** (check first, highest rigor):
- 7.3 (multi-LoRA aggregation) `GATE DIFF` — per-stream progress tracking must not double-count across modes
- 42.1 (source-type immutability) `GATE DIFF` — once a pipeline reports adapter progress, switching to full-finetune must fail-fast

**Extended GATE rules**: 7.1-7.2 (all `DIFF`)

**CHECK rules**: 51.1-51.2, 59.1-59.2

### Audit Focus

- [MECHANICAL] **`8250c77`**: Is the source-type invariant enforced with a clear fail-fast error — if a pipeline sends adapter progress then switches to full-finetune progress, does it crash immediately (not silently corrupt)? When a pipeline transitions between adapter rounds, does old stream state get cleared or does it accumulate indefinitely?
- [MECHANICAL] **`0fabbd3`**: Confirm `lora_name` is `Optional[str] = None` so existing single-LoRA pipelines work without any change. This is a pure additive change — no logic should be gated on it here (tracing use comes later in Group 10).

### Definition of Done
- [ ] [MECHANICAL] Nested progress store in `8250c77` verified: `[pipeline_id][mode][stream_key]` keys are all domain-specific names
- [ ] [MECHANICAL] Source-type invariant (full-finetune vs adapter) enforced with fail-fast error, not silent corruption
- [ ] [MECHANICAL] `lora_name` field in `0fabbd3` is `Optional[str] = None` — no downstream logic gated on it yet
- [ ] [MECHANICAL] All 2 commits reviewed with per-commit template filled
- [ ] [MECHANICAL] Findings logged with P0-P3 severity

---

## Group 6 — Multi-LoRA Foundation: Config + Training Layer
**Repo**: `[R]` only | **Commits**: 5

### Context

Foundation layer for multi-LoRA: the config schema, per-adapter Megatron training, the `lora_routing` dispatch utility, and an integration test that verifies per-adapter isolation. This group establishes the core correctness claim: training adapter A must not affect adapter B's weights.

The key new feature in Megatron (`2ca3a86`) is `lora_optimizer_mode = 'per_adapter'`: each adapter gets its own Adam optimizer state. Without this, a single shared optimizer would mix moments across adapters, breaking isolation.

The integration test (`d93ab92`) is the ground-truth for the correctness claim. Read it before reviewing the implementation to understand what "correct" means.

### Commits

**`0346f42`** `feat(lora): add lora_routing utility for multi-LoRA microbatch dispatch`
- `roll/utils/lora_routing.py` (new, +88) — ports routing utilities from `ROLL_multi_lora`; supports both `domain` (ROLL_schedrl convention) and `lora_name` (ROLL_multi_lora convention) for backward compatibility

**`e84b747`** `feat(config): add adapters field for multi-LoRA configuration`
- `roll/configs/model_args.py` — adds `adapters: Dict[str, LoraArguments]` to `ModelArguments`

**`2ca3a86`** `feat(megatron): add per-adapter multi-LoRA training support`
- `roll/distributed/strategy/megatron_strategy.py` (large) — implements `lora_optimizer_mode` (`shared` | `per_adapter`); dedicated optimizer + LR scheduler per adapter in `per_adapter` mode; new methods: `zero_grad()`, `forward_backward_only()`, `optimizer_step_only()`, `train_step_lora()`, `get_lora_tensors()`, `set_lora_tensors()`, `copy_lora_params()`; modifies `load_states/offload_states` for `per_adapter` compatibility

**`ad52ff1`** `feat(sft): add train_step_lora and LoRA weight management methods`
- `roll/pipeline/sft/sft_worker.py` (+39) — exposes `train_step_lora()`, `get_lora_tensors()`, `set_lora_tensors()`, `copy_lora_params()` using `ONE_TO_ALL` dispatch

**`d93ab92`** `test(integration): add per-adapter single LoRA step equivalence test`
- `tests/integration/test_per_adapter_single_lora_step_equivalence.py` (new, +691) — verifies: N mixed-domain microbatches in one call == N separate single-domain calls; tests gradient accumulation, weight tensor equality, adapter isolation via domain routing

### Curated Audit Rules

**Minimum gate set** (check first, highest rigor):
- 8.1 (sequential training) `GATE DIFF` — one adapter trains at a time, never concurrent
- 8.2 (RNG isolation) `GATE DIFF` — per-adapter RNG state saved/restored on switch
- 8.4 (optimizer validation) `GATE DIFF` — per_adapter mode gives each adapter its own optimizer state

**Extended GATE rules**: 8.3, 8.5 (all `DIFF`)

**CHECK rules**: (none)

### Audit Focus

- [JUDGMENT] **`0346f42`**: Does `lora_routing.py` reject a batch that contains two different adapter names (mixed-adapter batch must never happen)? Is there a collision-detection path for two adapters mapping to the same slot? The dual `domain`/`lora_name` convention — does it add ambiguity if a batch has both fields set?
- [MECHANICAL] **`2ca3a86` (high scrutiny)**: Does `optimizer_step_only()` for adapter A not touch adapter B's Adam moments or parameters? Does `train_step_lora()` correctly route the gradient to the right adapter? Does modifying `load_states/offload_states` break the non-multi-LoRA path (when `lora_optimizer_mode='shared'` or no LoRA at all)?
- [JUDGMENT] **`ad52ff1`**: Is `ONE_TO_ALL` the right dispatch mode — should it be ALL_TO_ALL (each rank computes independently) or source-of-truth? Does `copy_lora_params()` do a deep copy or a reference copy?
- [MECHANICAL] **`d93ab92` (read first)**: Does the test assert weight tensor equality (not just "runs without crash")? Are gradients checked per-adapter — not just the final accumulated result? Does the test exercise the `per_adapter` optimizer mode specifically?

### Definition of Done
- [ ] [MECHANICAL] INV-5 confirmed: `lora_routing.py` rejects mixed-adapter batches (two different `lora_name` in one batch)
- [ ] [MECHANICAL] INV-4 confirmed: `optimizer_step_only()` for adapter A does not touch adapter B's Adam moments
- [ ] [MECHANICAL] `per_adapter` optimizer mode tested in `d93ab92` — weight tensor equality asserted, not just "no crash"
- [ ] [MECHANICAL] `load_states/offload_states` in `2ca3a86` verified to not break non-multi-LoRA path (`lora_optimizer_mode='shared'`)
- [ ] [MECHANICAL] All 5 commits reviewed with per-commit template filled
- [ ] [MECHANICAL] Findings logged with P0-P3 severity

---

## Group 7 — Multi-LoRA Core Pipeline Orchestration
**Repo**: `[R]` only | **Commits**: 11

### Context

This is the core of the multi-LoRA plan: `SchedRLMultiLoraPipeline` — the pipeline that trains multiple LoRA adapters sequentially under SchedRL scheduler control.

Key design constraints from the plan being implemented here:
- Adapter training is **sequential** — one adapter trains at a time, never concurrent
- **`sleep_level=2`** — only GPU release (no process teardown between adapters)
- Per-adapter RNG state must be saved and restored on adapter switch (to maintain reproducibility)
- Selective sync — after training adapter A, only A's weights are broadcast to inference workers (adapter B's weights are untouched)
- `_op_lock` serializes `resize_infer` and weight sync — they cannot run concurrently

Read `0ae269b` (the pipeline implementation) before the supporting commits (`d102860` through `d97f9f0`).

### Commits

**`d102860`** `feat(multi-lora): add adapters_to_update parameter to model_update`
- `roll/distributed/executor/model_update_group.py` (+6) — adds `adapters_to_update` filter; `None` means update all (backward compatible)

**`3143d9d`** `feat(multi-lora): add per-adapter checkpoint promotion and selective sync`
- `roll/distributed/executor/worker.py` (+17) — worker supports promoting only the active adapter's checkpoint; selective sync scoped to trained adapter

**`96915f6`** `feat(multi-lora): add model_update_lora_subset helper method`
- `roll/pipeline/base_pipeline.py` (+8) — `model_update_lora_subset()` calls `model_update` scoped to a subset of adapters

**`c46269c`** `feat(multi-lora): add train_step_lora RPC to ActorWorker`
- `roll/pipeline/base_worker.py` (+13) — `train_step_lora` RPC on `ActorWorker` base class, delegates to strategy

**`0ae269b`** `feat(multi-lora): add SchedRLMultiLoraPipeline implementation`
- `roll/schedrl_adapter/multi_lora_pipeline.py` (new, +679) — the core pipeline: sequential per-adapter training loop; `sleep_level=2`; GPU handoff via SchedRL `notify_ready_to_release`; per-adapter step with `train_step_lora` + `promote_active_adapter_checkpoint`

**`02378f4`** `feat(multi-lora): add pipeline registration and shared RequestScheduler support`
- `roll/schedrl_adapter/adapter.py` (+13) — multi-LoRA pipeline registration with SchedRL
- `roll/schedrl_adapter/concurrent_pipeline.py` (+66) — exposes shared `RequestScheduler` instance for `multi_lora_pipeline` to reference

**`507f740`** `feat(multi-lora): add per-adapter cache, RNG state, and selective sync support`
- `roll/distributed/strategy/megatron_strategy.py` (+184) — per-adapter bucket cache; saves/restores RNG state on adapter switch; selective sync scoped to trained adapter; `load_states` optimization

**`5a799e7`** `feat(multi-lora): add _op_lock and notify_adapter_updated for selective sync`
- `roll/distributed/scheduler/generate_scheduler.py` (+176/-86) — `_op_lock` serializes `resize_infer` and weight sync; `notify_adapter_updated` signals rollout scheduler that a specific adapter's weights are fresh
- `roll/distributed/scheduler/rollout_scheduler.py`

**`d1b80bf`** `feat(multi-lora): add adapters_to_sync support in model update service`
- `roll/schedrl_adapter/model_update_service.py` (+3) — filters model update to only sync specified adapters
- `roll/third_party/megatron/model_update.py` (+156/-49) — per-adapter selective sync

**`5c31db9`** `fix(multi-lora): PP support and per-adapter optimizer fixes`
- `roll/distributed/strategy/megatron_strategy.py` — `_safe_dist_barrier` for NCCL compatibility; conditional `input_ids/labels` based on PP stage; sets all adapters trainable before per-adapter optimizer construction; allows Megatron FP32 main params in validation; rebuilds schedulers with DP-adjusted `max_steps`; supports `meta_info['lora_name']` as fallback routing
- `roll/pipeline/sft/sft_worker.py`, `mcore_adapter`

**`d97f9f0`** `feat(multi-lora): add setup_lora_training_from_adapters for multi-adapter setup`
- `roll/models/model_providers.py` (+119) — `_resolve_lora_target_modules` handles `'all-linear'`, `'all-embedding'`, `'all-router'`; `setup_lora_training_from_adapters` sets up multi-adapter PEFT; handles `autocast_adapter_dtype` for non-first adapters; sets all adapters trainable after creation for Megatron grad buffer allocation

### Curated Audit Rules

**Minimum gate set** (check first, highest rigor):
- 8.1 (sequential training) `GATE DIFF` — one adapter trains at a time, never concurrent
- 19.1 (train/val shrink order) `GATE DIFF` — train shrink before val shrink during adapter switch
- 48.1 (lock ordering) `GATE END-STATE` — _op_lock and _resize_sync_lock must never be held in incompatible order

**Extended GATE rules**: 8.2-8.5, 19.2-19.3, 21.1-21.2, 25.1-25.3, 48.2 (mix of `DIFF` and `END-STATE`)

**CHECK rules**: (none)

### Audit Focus

- [MECHANICAL] **`0ae269b` (core pipeline — high scrutiny)**: Verify adapter training is strictly sequential — no concurrent adapters. Verify `sleep_level=2` only (GPU release, no process teardown). What is the per-adapter step termination condition — how does the pipeline know to move to the next adapter? Is there any code path where two adapters could be in training state simultaneously?
- [MECHANICAL] **`507f740` (RNG state)**: Is RNG state save/restore covering all relevant sources (CUDA, CPU, NumPy, Python random)? Is cache invalidation triggered on every weight update — can a stale cache cause model corruption?
- [JUDGMENT] **`5a799e7` (`_op_lock` — high scrutiny)**: Is the `_op_lock` scope correct — does it cover the full resize+sync critical section? Can `notify_adapter_updated` and the next `resize_infer` ever deadlock via `_op_lock` (lock held in two incompatible orderings)?
- [MECHANICAL] **`d1b80bf`**: After selective sync, does the inference worker have a consistent view of all adapters — specifically, does syncing adapter A's weights leave adapter B's resident weights untouched and uncorrupted?
- [JUDGMENT] **`5c31db9` (PP fixes)**: Does `_safe_dist_barrier` correctly handle PP stages where not all ranks participate in NCCL? Is "all adapters trainable before optimizer construction" a Megatron requirement — is it clearly commented?
- [JUDGMENT] **`02378f4`**: Is the shared `RequestScheduler` access thread-safe between `multi_lora_pipeline` and `concurrent_pipeline` accessing it concurrently?
- [MECHANICAL] **`d97f9f0`**: Does PEFT's `autocast_adapter_dtype` override work correctly for adapters 2..N (PEFT only applies it to the first adapter by default)?

### Definition of Done
- [ ] [JUDGMENT] INV-3 confirmed: `_op_lock` in `5a799e7` covers full resize+sync critical section; no deadlock path with `_resize_sync_lock`
- [ ] [MECHANICAL] Sequential adapter training verified in `0ae269b` — no code path allows concurrent adapters
- [ ] [MECHANICAL] `sleep_level=2` enforced in `0ae269b` — GPU release only, no process teardown
- [ ] [MECHANICAL] INV-4 confirmed: selective sync in `d1b80bf` leaves non-trained adapter weights uncorrupted
- [ ] [JUDGMENT] Shared `RequestScheduler` in `02378f4` verified thread-safe
- [ ] [MECHANICAL] All 11 commits reviewed with per-commit template filled
- [ ] [MECHANICAL] Findings logged with P0-P3 severity

---

## Group 8 — Multi-LoRA Rollout Integration
**Repo**: `[R]` only | **Commits**: 7

### Context

Wires `lora_name` through the entire rollout path. The data flow this group implements:

```
Env Manager (injects lora_name into request)
  → policy_proxy / llm_proxy (carries lora_name)
  → vLLM strategy (routes request to correct LoRA slot by lora_name)
  → generate scheduler (dispatches to correct DP rank)
  → inference workers (load/serve correct adapter)
```

The critical invariant enforced here: **a single inference batch must never contain requests for two different `lora_name` values**. Mixed-adapter batches must be detected and rejected.

Also handles legacy config backward compatibility: if only `lora_rank`/`lora_target` are configured (single-LoRA), the `adapters` dict is auto-derived so existing configs work unchanged.

### Commits

**`a727f74`** `feat(lora): add multi-LoRA routing utilities and adapter config normalization`
- `roll/configs/model_args.py` (+48) — `get_lora_name_array()` for strict `lora_name`-only routing; `ensure_lora_name_in_batch()` for single-adapter auto-fill; `adapter_name` field in `LoraArguments`; adapter key normalization with collision fail-fast; auto-derives `adapters` dict from legacy `lora_rank/lora_target` fields

**`202dcad`** `feat(vllm): add multi-LoRA routing support to vLLM strategy`
- `roll/distributed/strategy/vllm_strategy.py` (+410) — per-prompt LoRA request routing via `get_lora_name_array()`; `get_lora_id()`, `list_loras()`, `wait_loras_ready()` async methods; `_normalize_lora_int_ids_loaded()` for cross-rank ID aggregation; enforces `VLLM_USE_V1=1`; fails fast on `load_format='dummy'` in LoRA mode

**`c56444a`** `feat(env): inject lora_name in env managers for multi-LoRA routing`
- `roll/pipeline/agentic/env_manager/traj_env_manager.py` (+85), `agent_native_env_manager.py` (+57), `step_env_manager.py` (+33), `vl_traj_env_manager.py` (+35), `step_concat_env_manager.py` (+16), `base_env_manager.py` (+3) — injects `lora_name` in `format_messages()` (inference path) and `formulate_rollouts()` (training path); validates tag→adapter mapping; auto-uses single-adapter key when only one adapter configured
- `roll/pipeline/agentic/environment_worker.py`, `roll/pipeline/agentic/llm_proxy/policy_proxy.py`

**`7e0147d`** `feat(pipeline): add multi-LoRA integration to workers and schedulers`
- `roll/pipeline/base_worker.py` — `lora_name` auto-fill guard; `get_lora_id/list_loras/wait_loras_ready` wrappers
- `roll/pipeline/sft/sft_worker.py` — docstring updates
- `roll/distributed/scheduler/generate_scheduler.py` — multi-LoRA aware generation scheduling
- `roll/distributed/scheduler/rollout_scheduler.py` — `resume`, `get_inflight_counts`, `offload_dp_ranks` RPCs
- `roll/schedrl_adapter/multi_lora_pipeline.py` — fixes trained-adapter detection to use `lora_name` (was `domain`)
- `roll/pipeline/agentic/agentic_config.py`, `roll/distributed/scheduler/initialize.py`, `roll/pipeline/agentic/env_manager/base_config.py`

**`4eb6706`** `chore(utils): add lora_name support to collective utilities`
- `roll/utils/collective/collective.py` (+7), `roll/utils/send_recv_utils.py` (+7) — `lora_name` passthrough in collective communication helpers (for logging/tracing, not routing)

**`7efa3ba`** `fix(sft): ensure lora_name broadcast before validation in train_step_lora`
- `roll/distributed/strategy/megatron_strategy.py` — moves `_broadcast_non_tensor_batch + get_data_input` before `ensure_lora_name_in_batch` so non-root TP/PP ranks receive `lora_name` via broadcast before validation
- `roll/schedrl_adapter/multi_lora_pipeline.py` — adds `_verify_lora_model_update` call after `expand_sampler`

**`e472375`** `feat(multi-lora): update strategy, workers, and scheduler for multi-LoRA support`
- `roll/distributed/strategy/megatron_strategy.py` — LoRA adapter load/offload; per-adapter weight routing
- `roll/third_party/vllm/worker.py` — propagates `lora_name` through generate requests
- `roll/pipeline/base_worker.py` — routes train/infer steps to correct LoRA adapter per stream
- `roll/distributed/scheduler/generate_scheduler.py` — multi-LoRA aware generation
- `roll/pipeline/agentic/agentic_config.py` — `multi_lora_config` fields
- `roll/third_party/megatron/model_update.py` — per-adapter selective model update
- `roll/schedrl_adapter/concurrent_pipeline.py` — passes `lora_name` through dispatch
- `roll/pipeline/agentic/env/deepeyes/env.py`, `env/gem/math_env.py`

### Curated Audit Rules

**Minimum gate set** (check first, highest rigor):
- 8.5 (LoRA ID consistency) `GATE DIFF` — lora_name flows end-to-end without mutation
- 11.3 (dual-write) `GATE END-STATE` — lora_name set in both inference and training paths

**Extended GATE rules**: (none beyond minimum gate)

**CHECK rules**: 9.1

### Audit Focus

- [MECHANICAL] **`c56444a` (env manager injection)**: Is `lora_name` injected before the request is dispatched to inference — not after? Are all **5** env manager variants consistently updated (`traj`, `step`, `step_concat`, `vl_traj`, `agent_native`)? Does each variant inject at both the inference path (`format_messages`) and training path (`formulate_rollouts`)?
- [MECHANICAL] **`202dcad` (vLLM routing — high scrutiny)**: Is there any code path where a batch contains two different `lora_name` values (mixed-adapter batch)? This must be impossible. Does `wait_loras_ready()` poll or block — can it hang indefinitely if an adapter never loads?
- [MECHANICAL] **`7efa3ba`**: Does the broadcast ordering fix the `RuntimeError` on TC-3 through TC-7 (tp>1, pp>1)? The root cause: non-root TP ranks did not receive `lora_name` via broadcast before `ensure_lora_name_in_batch` validated it. Confirm the fix is complete. Does `_verify_lora_model_update` fail-fast on adapter ID skew?
- [MECHANICAL] **`7e0147d`**: "trained-adapter detection fix" — what was the old `domain`-based bug? Did it cause the wrong adapter to be marked as trained, or just a lookup failure?
- [MECHANICAL] **`e472375` (large integration commit)**: Verify that all changes include `if lora_name is not None:` (or equivalent) guards so the single-adapter / no-LoRA path is unaffected. Does `lora_name` propagate correctly end-to-end for the multi-LoRA case?
- [MECHANICAL] **`4eb6706`**: Confirm `lora_name` in collective utilities is used only for logging/tracing, not for routing decisions. Routing in collectives would be a design violation.

### Definition of Done
- [ ] [MECHANICAL] INV-5 confirmed: `202dcad` prevents mixed-adapter batches in vLLM strategy — run: `rg -n "lora_name" external/ROLL_schedrl/roll/distributed/strategy/vllm_strategy.py | rg -i "assert|raise|check|ensure"`
- [ ] [MECHANICAL] `lora_name` injection verified in all 5 env managers: `rg -l "lora_name" external/ROLL_schedrl/roll/pipeline/agentic/env_manager/` returns `traj`, `step`, `step_concat`, `vl_traj`, `agent_native`
- [ ] [MECHANICAL] `lora_name` broadcast ordering fix in `7efa3ba` verified — non-root TP ranks receive `lora_name` before validation
- [ ] [MECHANICAL] All changes in `e472375` guarded with `if lora_name is not None:` so single-adapter / no-LoRA path unaffected
- [ ] [MECHANICAL] All 7 commits reviewed with per-commit template filled
- [ ] [MECHANICAL] Findings logged with P0-P3 severity

---

## Group 9 — Testing + Example Configs
**Repo**: `[R]` only | **Commits**: 6

### Context

Integration test coverage for the TP/PP/DP combinations that matter for multi-LoRA. The test file grows across multiple commits (`e988e95`, `ec132b4`, `4c87095`) covering:
- TC5: dp=1, tp=1, pp=2
- TC6: tp=2, pp=2
- TC7: dp=2, pp=2

`ec132b4` required `mcore_adapter` changes to pass the TP=2+PP=2 case — it includes fixes to `lora_layer.py` for TP-sharded LoRA weights. It was a forced merge (noted in commit message).

### Commits

**`6c4df08`** `fix: misc robustness improvements for PP and distributed setup`
- `mcore_adapter/src/mcore_adapter/initialize.py` — passes `device_id` to `initialize_process_group`
- `roll/distributed/scheduler/resource_manager.py` — handles `None` placement group name
- `roll/utils/network_utils.py` — catches `PermissionError` in `get_node_ip`
- `roll/configs/worker_config.py` — adds `eval` fallback for `device_mapping` parsing
- `roll/pipeline/sft/sft_worker.py` — moves `data.to(device)` after `get_data_input` for PP
- `roll/distributed/scheduler/decorator.py` — removes `pp_rank` check blocking non-first-stage dispatch

**`e988e95`** `test(multi-lora): add TC5 for PP=2 and improve test robustness`
- `tests/integration/test_per_adapter_single_lora_step_equivalence.py` — TC5 (dp=1, tp=1, pp=2); UUID-based unique cluster names; `lora_name` in `meta_info` for PP routing; `overlap_p2p_comm=False`; kills workers on shutdown

**`ec132b4`** `(multi-lora): passed the tp2 pp2 test case for multi lora`
- `mcore_adapter/src/mcore_adapter/adapters/lora_layer.py` (+166) — TP-sharded LoRA weight handling
- `mcore_adapter/src/mcore_adapter/adapters/utils.py` (+34), `models/model_factory.py` (+15)
- `tests/integration/test_per_adapter_single_lora_step_equivalence.py`, `megatron_strategy.py`, `sft_worker.py`

**`4c87095`** `test(multi-lora): add TC6 tp2pp2 and TC7 dp2pp2`
- `tests/integration/test_per_adapter_single_lora_step_equivalence.py` (+88) — TC6 and TC7

**`8e2be5e`** `feat(examples): add multi-LoRA pipeline and smoke test configs`
- `roll/pipeline/agentic/agentic_multi_lora_pipeline.py` (new, +996) — `AgenticMultiLoraPipeline`
- `design_docs/single_pipeline_multi_lora_plan.md` (new, +1240) — design doc
- `examples/qwen2.5-0.5B-agentic/agentic_val_sokoban_lora.yaml` — smoke test config
- `examples/qwen2.5-0.5B-agentic/agentic_val_sokoban_mulit_lora_partial_overlap.yaml` — multi-LoRA example

**`b0f0228`** `chore(examples): replace sokoban_grpo configs with full_finetune and multi_lora pipeline configs`
- `examples/multi_pipeline/` — renames example configs

### Curated Audit Rules

**Minimum gate set**: (no minimum gate — CHECK only for this group)

**CHECK rules**: 8.1-8.4

### Audit Focus

- [JUDGMENT] **`6c4df08`**: The `pp_rank` check removal — what did it guard before? Is removing it safe for all PP stages, or does it allow non-first-stage ranks to receive data they should not? The `PermissionError` catch in `network_utils` — does this silently hide a legitimate failure on real clusters?
- [MECHANICAL] **`ec132b4` (force-merged — check carefully)**: The commit message contains conflict markers. Confirm no merge artifacts remain in the final state of all touched files. Does `lora_layer.py` correctly handle TP-sharded LoRA weight initialization and forward pass?
- [MECHANICAL] **`e988e95` + `4c87095`**: Do TC5/TC6/TC7 assert weight tensor equality across all ranks, not just "runs without crash"? Is there a DP-only test (no PP) — TC7 (DP=2+PP=2) covers DP but only with PP. Is there a TC covering DP-only (pp=1)?
- [JUDGMENT] **`8e2be5e`**: `agentic_multi_lora_pipeline.py` is nearly 1000 lines — is it a thin orchestration wrapper over `multi_lora_pipeline.py` (Group 7), or does it re-implement core training logic? Any duplicated logic between the two that should be shared?

### Definition of Done
- [ ] [MECHANICAL] `ec132b4` (force-merged) verified: no merge conflict artifacts remain in final file state
- [ ] [MECHANICAL] TP-sharded LoRA in `lora_layer.py` verified for correct weight initialization and forward pass
- [ ] [MECHANICAL] TC5/TC6/TC7 assert weight tensor equality across ranks, not just "runs without crash"
- [ ] [JUDGMENT] `agentic_multi_lora_pipeline.py` confirmed as orchestration wrapper — no duplicated core training logic from `multi_lora_pipeline.py`
- [ ] [MECHANICAL] All 6 commits reviewed with per-commit template filled
- [ ] [MECHANICAL] Findings logged with P0-P3 severity

---

## Group 10 — GPU Timeline Tracing
**Repo**: `[S]` 11 commits + `[R]` 1 commit | **Total**: 12

### Context

Optional Perfetto-based GPU timeline visualization. Tracing is entirely disabled by default — all tracing calls go through `_safe_trace` helpers that no-op when tracing is off. `tg4perfetto` is an optional dependency; missing it raises a clear error if tracing is explicitly requested.

This group also contains **`620bdea`** — a correctness bug fix unrelated to tracing but discovered during this work. It fixes two bugs in the scheduler's `resize_infer` path: an overly strict mutual exclusivity check that blocked valid shrink+expand operations, and a double-count of freed GPUs in the Phase 2 preemption loop. This commit should be reviewed with the same scrutiny as the planner logic in Group 1.

The tracing commits form a natural evolution: core infrastructure (`a20f17d`) → queue visualization (`f54044e`) → active GPU counter (`1ed18cb`) → refactors to separate concerns (`db5a7c3`, `eadea41`, `61c3a27`, `9885dcb`, `41de4dc`) → timing and label improvements (`6a6a431`) → error handling (`4873ee5`).

For the intermediate refactor commits (`db5a7c3` through `41de4dc`), skim for consistency and read the final state in `6a6a431` and `4873ee5`.

### Commits

**`[S] a20f17d`** `feat(tracing): implement GPU timeline tracing for scheduler`
- `schedrl/scheduler/scheduler.py` — conditional `tg4perfetto` import with graceful fallback; `_safe_trace_call`, `_safe_trace`, `_safe_trace_get` helpers; GPU track management; lifecycle `_init_tracing`/`_shutdown_tracing`; 1s throttled `_maybe_flush_trace`

**`[S] 620bdea`** `fix(scheduler): fix resize_infer mutual exclusivity check and double-count GPU bug`
- `schedrl/scheduler/scheduler.py`
- Bug 1: Old check raised `RuntimeError` whenever any pipeline had both removes AND adds in one cycle. Fix: only raise when the **same** `dp_rank` appears in both (true conflict). Different `dp_ranks` on different GPUs are legal.
- Bug 2: Phase 2 preemption loop double-counts a freed GPU in the inner preemption loop.

**`[S] f54044e`** `feat(scheduler): implement queue visualization for GPU tracing`
- `schedrl/scheduler/scheduler.py` — counter tracks for queue depth per priority; slice tracks for individual request wait time; `Queue_TRN`, `Queue_GEN` sub-groups; per-cluster slice tracks (not per-priority) to handle LIFO `close()` correctly

**`[S] 1ed18cb`** `feat(scheduler): add active_gpus counter track for GPU utilization`
- `schedrl/scheduler/scheduler.py` — `CounterTrack` for GPU utilization; integrated at all `idle_gpus` mutation sites

**`[S] db5a7c3`** `refactor(scheduler): remove redundant cycle start marker`
- `schedrl/scheduler/scheduler.py` — removes `_trace_cycle_marker` (metrics now tracked via counter tracks)

**`[S] eadea41`** `feat(scheduler): add instant markers for scheduling events`
- `schedrl/scheduler/scheduler.py` — instant markers for exec cycle, enqueue, release; `_init_gpu_tracks()`; `_apply_plan_and_signal()` now returns operation details for tracing

**`[S] 61c3a27`** `refactor(scheduler): separate instant markers into dedicated tracks`
- `schedrl/scheduler/scheduler.py` — three separate tracks: `exec_markers`, `enqueue_markers`, `release_markers`; `_create_marker_track()` helper

**`[S] 9885dcb`** `refactor(scheduler): eagerly create queue tracks in priority order`
- `schedrl/scheduler/scheduler.py` — `_init_queue_tracks()` creates all queue groups at init time: INIT → TRN → CRT → OLD → REF → VAL → GEN

**`[S] 41de4dc`** `fix(scheduler): use numeric prefixes for Perfetto track ordering`
- `schedrl/scheduler/scheduler.py` — adds numeric prefixes (`01_enqueue_markers`, `02_exec_markers`, etc.) because Perfetto sorts alphabetically

**`[S] 6a6a431`** `refactor(scheduler): fix GPU trace timing and improve trace label readability`
- `schedrl/scheduler/scheduler.py` — moves GPU trace open/close to `_execute_resize_calls` so timestamps reflect actual RPC completion; `_GPUTraceInfo` dataclass; extracts `_collect_shrink/expand_trace_infos_locked`; carries `lora_name` on `SignalPendingAllocationOp` at planning time; removes `lora_name` from `ClusterAllocation`; queue track names include `lora_name` + `pipeline_id`

**`[S] 4873ee5`** `fix(schedrl): inherit driver env vars and improve tracing error handling`
- `schedrl/client/client.py` — passes driver environment to Ray actors; `runtime_env` for scheduler actor
- `schedrl/scheduler/scheduler.py` — raises clear error when tracing requested but `tg4perfetto` not installed
- `schedrl/requirements.txt` — adds `protobuf<3.21.0`
- `external/ROLL_schedrl` (submodule bump to `e703995`)

**`[R] 2da6eff`** `feat(adapter): pass lora_name to scheduler for GPU trace labels`
- `roll/schedrl_adapter/concurrent_pipeline.py` (+9) — adds `lora_name` parameter to `_request_static_cluster()` and `_release_and_request_static_cluster()`
- `roll/schedrl_adapter/multi_lora_pipeline.py` — extracts trained adapters from `batch.non_tensor_batch` as comma-separated string

### Curated Audit Rules

**Minimum gate set** (check first, highest rigor):
- 16.1 (trace context lifecycle) `GATE END-STATE` — tracing must be write-only, never read back by scheduler logic
- 57.1 (dp_rank overlap) `GATE DIFF` — no dp_rank assigned to two pipelines simultaneously
- 24.1↑ (mutual exclusivity) `GATE DIFF` — same dp_rank in both removes+adds → RuntimeError

**Extended GATE rules**: 1.4, 16.2-16.3, 57.2 (mix of `DIFF` and `END-STATE`)

**CHECK rules**: (none)

(↑ = UPDATED rule text, see Rule Updates section above)

### Audit Focus

- [JUDGMENT] **`620bdea` (correctness fix — high scrutiny)**: Is Bug 1 fix correct — can a pipeline legitimately shrink one `dp_rank` and expand a different `dp_rank` simultaneously? What scenario triggers this (e.g., preemption + rebalance in the same cycle)? Is Bug 2 (double-count) fixed completely — are there other places in the planner that iterate freed GPUs and might similarly double-count?
- [MECHANICAL] **`a20f17d`**: Is tracing entirely off the hot path when disabled — no dict lookups, no conditional checks in tight loops? Does `_safe_trace` suppress all exceptions, or only tracing-specific ones? Suppressing general exceptions would hide real bugs.
- [MECHANICAL] **`6a6a431`**: After moving `lora_name` off `ClusterAllocation`, is it still accessible at all places that previously read it from there?
- [JUDGMENT] **`eadea41`**: `_apply_plan_and_signal()` changed from void to returning operation details. Are there callers that ignore the return value — and is that safe?
- [MECHANICAL] **`4873ee5`**: Submodule bump to `e703995` — confirm this is the final HEAD of the ROLL_schedrl review range. Does the error for missing `tg4perfetto` block startup or just log and continue (should block if tracing was explicitly requested)?
- [MECHANICAL] **Across all tracing commits**: Tracing path must never affect scheduler correctness. Verify no tracing state is read back by the scheduler logic (it is write-only).

### Definition of Done
- [ ] [MECHANICAL] `620bdea` Bug 1 fix verified: mutual exclusivity check only blocks same `dp_rank` in both removes and adds (not different dp_ranks)
- [ ] [MECHANICAL] `620bdea` Bug 2 fix verified: no other places in the planner double-count freed GPUs
- [ ] [MECHANICAL] Tracing is write-only — no tracing state read back by scheduler logic (spot-check `scheduler.py` for any `if self._trace...` guarding non-trace behavior)
- [ ] [MECHANICAL] `tg4perfetto` absence raises clear error when tracing explicitly requested, does not silently degrade
- [ ] [MECHANICAL] Submodule bump `4873ee5` → `e703995` confirmed: `git show 4873ee5 -- external/ROLL_schedrl` matches ROLL_schedrl HEAD
- [ ] [MECHANICAL] All 12 commits reviewed with per-commit template filled
- [ ] [MECHANICAL] Findings logged with P0-P3 severity

---

## Group 11 — Late Bug Fixes + Robustness
**Repo**: `[S]` 1 commit + `[R]` 11 commits | **Total**: 12

### Context

Post-integration fixes found during multi-GPU testing. The two most important fixes:

1. **`[S] 0c66eba` + `[R] a81b69f`** (paired, same change): Replaces the single-rollout run loop in `multi_lora_pipeline` with a per-adapter `lora_step` loop using `ray.wait` (first-ready tag wins). The main repo (`[S]`) also contains scheduler-side changes. These two commits must be consistent with each other.

2. **`[R] 458c53a`** (vLLM streaming): Fixes a major OOM in the weight sync path during `actor_infer` expand. Previously all receive buffers were allocated upfront (peak = model + N × buffer). New approach: reload model first, then stream one buffer at a time via generator (peak = model + 1 buffer).

There are also two separate checkpoint OOM fixes (`f4a24cd` in Group 4 and `10ec933` here) — verify they address different code paths and don't conflict.

### Commits

**`[S] 0c66eba`** + **`[R] a81b69f`** `feat(multi-lora): per-adapter run loop, adapter sync, and load_states optimization`
- `[S]`: `schedrl/scheduler/scheduler.py`, `schedrl/scheduler/types.py`, `schedrl/orchestrator/orchestrator.py`
- `[R]`: `roll/schedrl_adapter/multi_lora_pipeline.py`, `roll/schedrl_adapter/adapter.py`, `roll/distributed/strategy/megatron_strategy.py`, `roll/distributed/executor/worker.py`
- Replaces single-rollout `run()` with per-adapter `lora_step` loop; `barrier_mode=False`; `ray.wait` for first-ready tag; Phase 16 uses `train_step_lora + promote_active_adapter_checkpoint` per dirty adapter; `adapter.sync_adapter_weights()` called directly from pipeline; `_resize_sync_lock` serializes resize and sync; CPU bucket cache built while GPU weights resident

**`[R] f7ca74f`** `fix(multi-pipeline): thread limits and barrier_mode removal`
- Example configs, `roll/schedrl_adapter/adapter.py`, `roll/schedrl_adapter/multi_lora_pipeline.py`, `roll/third_party/vllm/worker.py`, `roll/distributed/strategy/megatron_strategy.py`
- Adds thread-limiting env vars (`OMP/MKL/OPENBLAS_NUM_THREADS`, `RAY_grpc_server_thread_pool_size`) to stay within container `pids.max`; removes `barrier_mode` from `AgenticMultiLoraPipeline`

**`[R] 5f5fe22`** `fix(examples): use HuggingFace and set actor_infer lora_rank to 8`
- Example YAML configs — `USE_MODELSCOPE=0`; reduces `actor_infer lora_rank` from 32 to 8 to match `actor_train` rank

**`[R] 458c53a`** `fix(vllm): stream base weights one-at-a-time and free sender GPU bucket`
- `roll/third_party/vllm/worker.py` (major) — streaming receiver: reload model first, then yield one buffer at a time via generator; peak = model + 1 buffer (was model + N buffers)
- `roll/schedrl_adapter/adapter.py` — frees sender GPU bucket after broadcast

**`[R] 447880c`** `fix(adapter): validate offload_nccl and scope LoRA verify to expanded ranks`
- `roll/schedrl_adapter/adapter.py` — `_validate_offload_nccl`: startup check that every active cluster has `offload_nccl=True` when `sleep_level=2` is active (NCCL buffers ~400-500 MB/process accumulate without this)
- `roll/schedrl_adapter/concurrent_pipeline.py`, `roll/schedrl_adapter/multi_lora_pipeline.py` — scopes LoRA adapter verification to only expanded ranks

**`[R] 59d38d1`** `fix(examples): reduce sequence_length and enable dynamic batching`
- Example YAML configs only — `sequence_length: 2048 → 1024`; enables `use_dynamic_batching_in_infer`

**`[R] 3af0811`** `fix(adapter): close HEAD gaps in concurrent_pipeline run()`
- `roll/schedrl_adapter/concurrent_pipeline.py` — adds `_broadcast_non_tensor_batch=True` after rollout; hoists `is_offload_states=True`; adds `shutdown()` for rollout schedulers; adds TODO comments for remaining gaps (Gap A: ref_log_probs, Gap B: batch_balance, Gap D: val())

**`[R] 10ec933`** `fix(pipeline): offload GPU states after checkpoint to prevent OOM on infer expand`
- `roll/pipeline/base_pipeline.py`, `roll/pipeline/base_worker.py`, example YAML configs

**`[R] 7e4c5b3`** `add prefix for tracker`
- `examples/multi_pipeline/start_multi_pipeline_test.py` — adds numeric prefix to tracker names in test script

**`[R] 3826f68`** `rename lora names`
- `examples/multi_pipeline/multi_lora_pipeline1.yaml`, `multi_lora_pipeline2.yaml` — renames adapter names in example configs

**`[R] e703995`** `fix(multi-lora): use deque for fair FIFO wait order in get_batch loop`
- `roll/schedrl_adapter/multi_lora_pipeline.py` (+19/-10) — previous data structure caused starvation; uses `deque` with `rotate()` for fair round-robin across adapters

### Curated Audit Rules

**Minimum gate set** (check first, highest rigor):
- 48.1 (lock ordering) `GATE END-STATE` — _resize_sync_lock vs _op_lock must never deadlock

**Extended GATE rules**: 48.2 (`END-STATE`)

**CHECK rules**: 2.2, 4.1-4.2, 37.1-37.3

### Audit Focus

- [JUDGMENT] **`0c66eba` + `a81b69f` (paired — high scrutiny)**: Are the two repos consistent — does the `[S]` scheduler side correctly support what the `[R]` pipeline side expects? What does "first-ready tag wins" mean for fairness — can one adapter starve others if it always produces rollouts faster? Does `_resize_sync_lock` create a deadlock risk with any other lock (`_op_lock` from `5a799e7`)?
- [JUDGMENT] **`458c53a` (vLLM streaming)**: Is the generator-based streaming correct under all ranks participating in the broadcast — do all receiver ranks complete before the generator is exhausted? Does the sender-side bucket free happen after all receivers confirm receipt, not before?
- [MECHANICAL] **`447880c`**: Does `_validate_offload_nccl` fail loudly at boot rather than silently at OOM time (correct behavior)? Is `offload_nccl=True` a correctness requirement, or only a memory budgeting requirement?
- [JUDGMENT] **`f4a24cd` (Group 4) vs `10ec933` (here)**: Both fix OOM caused by GPU states not being offloaded after checkpoint. Do they address different code paths (`concurrent_pipeline` vs `base_pipeline`) or does one subsume the other? Is there a risk of double-offload?
- [JUDGMENT] **`e703995`**: Was the starvation reproduced in a test, or was it theoretical? Does the `deque.rotate()` approach guarantee strict round-robin, or is it best-effort?
- [MECHANICAL] **`3af0811` TODO comments**: The TODO gaps (Gap A, B, D) document known simplifications vs HEAD. Are they tracked somewhere for follow-up, or only in comments?

### Definition of Done
- [ ] [JUDGMENT] `0c66eba` + `a81b69f` verified consistent: scheduler-side `[S]` and pipeline-side `[R]` agree on per-adapter `lora_step` semantics
- [ ] [JUDGMENT] INV-3 confirmed: `_resize_sync_lock` cannot deadlock with `_op_lock` — verify no call path holds both in opposite order
- [ ] [JUDGMENT] `458c53a` streaming verified: all receiver ranks complete before generator exhaustion; sender frees bucket after all receivers confirm
- [ ] [MECHANICAL] `447880c` `_validate_offload_nccl` verified: fails loudly at boot, not silently at OOM time
- [ ] [JUDGMENT] `f4a24cd` (Group 4) vs `10ec933` (this group) confirmed as different code paths — no double-offload risk
- [ ] [MECHANICAL] `e703995` deque-based fairness verified: `rotate()` provides round-robin, no adapter starvation
- [ ] [MECHANICAL] Gap A/B/D TODOs in `3af0811` tracked for follow-up (not only in code comments)
- [ ] [MECHANICAL] All 12 commits reviewed with per-commit template filled
- [ ] [MECHANICAL] Findings logged with P0-P3 severity

---

## Cross-Cutting Checks

Run these after completing all groups. All commands assume CWD = repo root (`/workspace/SchedRL`).

Each check has a command and an expected result. If actual output differs, log a finding.

**INV-7 Boundary enforcement** — `schedrl/` must have zero ROLL-specific imports:
```bash
rg --glob '*.py' "from roll|import roll" schedrl/
```
Expected: **zero matches**. Any match is P0 — schedrl/ must not depend on ROLL.

**INV-6 No scheduler policy in the adapter** — `roll/schedrl_adapter/` must not contain gap-ratio logic or priority assignments:
```bash
rg --glob '*.py' "gap_ratio|Priority\." external/ROLL_schedrl/roll/schedrl_adapter/
```
Expected: **zero matches**. Any match is P1 — scheduling policy belongs in `schedrl/scheduler/`.

**INV-8 Timeout source** — all timeouts in `schedrl/` must come from env vars:
```bash
rg -n --glob '*.py' "timeout.*=\s*[0-9]+" schedrl/
```
Expected: **zero matches**, or only matches inside `_get_env_timeout_s` default fallbacks. Any hardcoded timeout in business logic is P2.

**No `Any` misuse** — confirm `Any` is only at true dynamic boundaries:
```bash
rg -n --glob '*.py' ": Any|-> Any" schedrl/
```
Expected: review each match. `Any` at Ray RPC boundaries or JSON deserialization is acceptable. `Any` on internal function signatures is P2.

**INV-5 `lora_name` injection coverage** — all 5 env manager variants must inject `lora_name`:
```bash
rg -l "lora_name" external/ROLL_schedrl/roll/pipeline/agentic/env_manager/
```
Expected: **at least 5 files** — `traj_env_manager.py`, `step_env_manager.py`, `step_concat_env_manager.py`, `vl_traj_env_manager.py`, `agent_native_env_manager.py`. Missing any is P1.

**INV-5 No mixed-adapter batch path** — vLLM strategy must reject or prevent mixed-adapter batches:
```bash
rg -n "lora_name" external/ROLL_schedrl/roll/distributed/strategy/vllm_strategy.py | rg -i "assert|raise|check|ensure|must"
```
Expected: **at least one match** showing an assertion or raise. Zero matches means INV-5 is not enforced at the vLLM layer (P0).

**`offload_nccl` validation at boot** — must fail-fast if `offload_nccl=True` is missing when `sleep_level=2`:
```bash
rg -n "_validate_offload_nccl" external/ROLL_schedrl/roll/schedrl_adapter/adapter.py
```
Expected: **at least one match** showing a call during initialization. Zero matches means the check was removed or never wired (P1).

**Submodule bump consistency** — `4873ee5` bumped `ROLL_schedrl` to `e703995`:
```bash
git show 4873ee5 -- external/ROLL_schedrl
```
Expected: output shows `Subproject commit e703995...`. If the hash differs, the submodule bump is stale (P0).
