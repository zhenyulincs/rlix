# Rules for the Agent

- DO NOT add new files, like ".md" ".txt" or test, unless explicitly told or approved by user. 
- DO NOT make use of any new libraries or packages, unless explicitly told or approved by user.
- NEVER response with table, use list or Bullet points instead. 
- Human Understandability First: Code is rejected unless it is immediately maintainable by a teammate; every contribution MUST use intent-based naming, modular single-purpose functions with no magic values, and explicit type contracts that remove ambiguity.
- Typed Python Only: All new or modified Python code MUST annotate public APIs, minimize `Any` (use only for truly dynamic or untyped boundaries), and pass strict type checking (`mypy --strict`).
- ALWAYS comment code changes: MUST add comments for short explanation first before changing existing code.
- Copy-then-revise: When adding more than ~20 lines of new code, you MUST first search the codebase for a similar pattern (like use Grep/Glob for similar function or class). Copy verbatim the closest match then apply revision, noting the source file. Only write from scratch if no similar pattern exists.
- Trace root cause first: NEVER fix an error without identifying the root cause. If unclear, add targeted logs at suspicious spots and key points to observe runtime state before touching any code.     

## 1. Safety (Critical)
- **No Secrets**: Never print, store, or commit tokens or keys.
- **No Destruction**: Do not delete files or force-push without asking first.
- **No Surprises**: Do not change files unrelated to the task.

## 2. Reliability
- **Happy Path First**: Assume perfect conditions. Do not fix edge case like hardware or external service failure unless asked (like network, hardware, server is down). Just document the risk and let the code fail.
- **Fail Fast**: Crash immediately with a clear error message; do not hide errors.
- **No Fault Tolerance**: Assume rare failures cannot happen to simplify implementation, ignore failure handling and fault tolerance cases by failing fast and loudly, then terminating the whole program. Do not implement retry logic, recovery mechanisms, or graceful degradation unless explicitly requested or already exsit in the codebase.
- **No Idempotent Operations**: Assume operations always succeed and execute exactly once. Do not design for idempotency, deduplication, or duplicate detection. Avoid idempotent operation patterns.
- **Check Your Work**: Run the smallest relevant test. If you can't, tell the user what command to run.


## 3. Scope & Quality
- **Small Changes**: Do the smallest change that solves the problem. No extra formatting.
- **Consistent Changes**: Keep the changes logically consistent across all files.
- **Reuse Code**: Don't reinvent the wheel; use existing patterns and libraries.

## 4. Communication
- **Always English**: Always think and respond to user in English.
- **Simple English**: Use short sentences and plain words in respnose. Avoid jargon (or define it in one sentence).
- **Update Docs**: If you change behavior, paths, or commands, update docs in the same commit.
- **Ask First**: If unclear, ask one short question before starting.

## 5. Code Understandability Rules
- **Semantic Intent (Name + Scope)**: Names must communicate domain intent, and each function should do one job; if a name needs `and`, split the function.
- **Flat Logic (Anti-Nesting)**: Prefer guard clauses and early returns so the main path stays obvious; avoid nesting deeper than two levels.
- **Explicit Interfaces (Typed Contracts)**: Public APIs MUST be typed, prefer `T | None`, and minimize `Any` to true dynamic/untyped boundaries.
- **Why Layer (Intent Docs)**: Comments/docstrings explain non-obvious reasons, constraints, and tradeoffs; replace magic values with named constants.
- **Zero Friction (Automated Style)**: Follow local lint/format tools so code reads consistently and reviews focus on behavior.


## 6. IDE Integration

IMPORTANT: Prefer use the `pycharm-index` MCP server when applicable for code navigation and refactoring:

IMPORTANT: When debugging, prefer using pycharm-debugger MCP tools to interact with the IDE debugger.


# Repository Guidelines

This repository is a multi-framework workspace for SchedRL design + integration across several RL/post-training stacks.

default python env dir is /venv/main/bin/python


## Project Structure & Module Organization

- `design_doc/`: SchedRL design docs (protocols, adaptation plans).
- `external/`: third-party repos as git submodules.
- `external/nemo-rl/`, `external/nemo-gym/`: NeMo-RL and environment components.
- `external/ROLL/`: ROLL framework (Ray-based multi-role pipelines).
- `external/miles/`: Miles RL framework + rollout engines.
- `external/SkyRL/`: SkyRL train/agent framework.
- `external/sglang/`, `external/vllm/`: rollout engines.
- Each framework has its own packaging (`pyproject.toml` / `setup.py`) and its own `tests/` folder.

If a submodule folder is missing locally, run `git submodule update --init`.

## Build, Test, and Development Commands

Run commands from the relevant subproject root:

- ROLL: `cd external/ROLL && make test` (pytest) and `make precommit`.
- NeMo-RL (uses `uv`): `cd external/nemo-rl && uv sync` and `uv run --group test pytest -q`.
- Miles: `cd external/miles && pytest -q` (or follow `external/miles/docs/` and examples).
- SkyRL: `cd external/SkyRL &&` follow `external/SkyRL/README.md` and `external/SkyRL/skyrl-train/` examples.

## Coding Style & Naming Conventions

- Python: 4-space indentation; prefer explicit names over abbreviations.
- Follow the tooling and conventions of the subproject you’re changing:
  - `external/nemo-rl/`: `ruff` + `black` configured in `external/nemo-rl/pyproject.toml` (run via `uv`).
  - `external/ROLL/`: `pre-commit` hooks (`make precommit`).
- Keep edits scoped: avoid reformatting unrelated files.

## Testing Guidelines

- Use `pytest`; keep new tests next to the framework they cover under `*/tests/`.
- Prefer the smallest test that reproduces the behavior (unit → integration → e2e).

## Commit & Pull Request Guidelines

- Commit history here uses short, imperative summaries (e.g., `readme`); keep subjects concise.
- PRs should include: what changed, why, and which framework(s) it impacts.
- For protocol changes, update `design_doc/multi-pipeline-adaptation-plan.md` and keep `design_doc/archive/multi-pipeline_roll_old_design.md` as the reference sequence.
