[中文版](./CN/AutoResearch.md)

# AutoResearch Workflow

## 1. Overview

AutoResearch is an agent-driven iterative optimization workflow in AKG Agents. Unlike single-shot code generation (KernelGen) or population-based search (Evolve/AdaptiveSearch), AutoResearch uses a ReAct agent loop where the LLM autonomously plans optimization strategies, edits code, evaluates results, and keeps or discards changes across multiple rounds.

It is registered as a callable workflow (`call_autoresearch_workflow`) and can be invoked by KernelAgent via ToolExecutor, or directly via `LangGraphTask(workflow="autoresearch")`.

### When to Use AutoResearch

| Scenario | Recommended Workflow |
|----------|---------------------|
| Quick first draft, no verification needed | KernelGen |
| Generate + verify a single candidate | KernelGenOnly |
| Explore many variants in parallel | Evolve / AdaptiveSearch |
| **Deep iterative optimization of a single kernel** | **AutoResearch** |

AutoResearch is best suited when:
- An initial kernel exists but needs deep performance tuning
- The optimization space is complex and requires multi-round trial-and-error
- Agent autonomy (strategy planning, read documentation, diagnostic subagent) is more valuable than population diversity

## 2. Architecture

```
KernelAgent (ReAct)
  │
  ├─ call_autoresearch_workflow(op_name, task_desc, dsl, ...)
  │
  ▼
AutoresearchWorkflow (LangGraph single-node)
  │
  ├─ 1. Preflight (fail-closed: no worker → abort)
  │     a) Acquire worker + create KernelVerifier
  │     b) Validate reference: check_task_desc_static + check_task_desc_runtime
  │     c) Resolve seed:
  │        previous_code provided?
  │          → runtime verify (reference + kernel)
  │            → pass: use it
  │            → fail: fall through to KernelGen
  │        KernelGen loop (×3):
  │          → generate → CodeChecker (static) → runtime verify
  │          → fail: error fed back to KernelGen for retry
  │     d) Release preflight worker
  │
  ├─ 2. Knowledge assembly: Skill system + hardware docs + API docs
  │     Layer 0 (system prompt): fundamental/reference skills + hw info
  │     Layer 1 (docs/ directory): guides, examples, API docs (read on demand)
  │
  ├─ 3. scaffold_task_dir: generate task directory + git init
  │
  ├─ 4. AgentLoop: plan-based autonomous optimization
  │     ┌────────────────────────────────────────────────────────┐
  │     │                                                        │
  │     │  ┌─ LLM turn ─────────────────────────────────┐       │
  │     │  │  update_plan → patch_file/write_file        │       │
  │     │  │  read_file   → compact   → finish           │       │
  │     │  └──────────────────────────────────────────────┘       │
  │     │         │                                              │
  │     │    FeedbackBuilder (plan state machine)                │
  │     │    [no_plan] → [active] → [replanning] → ...          │
  │     │         │                                              │
  │     │    ┌─ TurnExecutor (automatic pipeline) ──────┐       │
  │     │    │  edit → quick_check → eval_fn → settle   │       │
  │     │    │           │             │          │      │       │
  │     │    │       syntax-only  KernelVerifier  keep/  │       │
  │     │    │                   verify+profile   discard│       │
  │     │    │                                   (git)   │       │
  │     │    └──────────────────────────────────────────  │       │
  │     │         │                                              │
  │     │    SessionStore (snapshot / resume / heartbeat)         │
  │     │         │                                              │
  │     │    Auto-diagnose (on N consecutive failures)           │
  │     │    → diagnostic subagent → require_replan              │
  │     │         │                                              │
  │     │    Context compression (microcompact + auto_compact)   │
  │     │                                                        │
  │     └────────────────────────────────────────────────────────┘
  │
  └─ 5. Read best result → {coder_code, verifier_result, profile_res}
```

### Module Layout

```
op/autoresearch/              # Autoresearch core (standalone-capable)
  agent/                      # AgentLoop, TurnExecutor, tools, feedback, compress
    loop.py                   # Main agent loop (ReAct: LLM → tools → eval)
    turn.py                   # Single LLM turn executor (edit → quick_check → eval pipeline)
    tools.py                  # Tool definitions + execution + guardrails
    feedback.py               # FeedbackBuilder: plan state machine + eval feedback
    session.py                # SessionStore: snapshot, resume, heartbeat, turn logging
    compress.py               # Context compression (microcompact + auto_compact)
    llm_client.py             # Standalone LLM client (Anthropic/OpenAI, for debug mode)
    file_logger.py            # Stdout tee to agent.log
    prompts/                  # System prompt templates
  framework/                  # Config, evaluation, experiment runner
    config.py                 # TaskConfig, AgentConfig, EvalResult, RoundRecord
    evaluator.py              # Eval executor (subprocess or injected eval_fn)
    runner.py                 # ExperimentRunner: eval → judge → git → log
    logger.py                 # JSONL + Markdown dual-format round logging
    report.py                 # Final report generation (text + optional plots)
  adapters/                   # AKG integration layer (NOT imported by core)
    llm_adapter.py            # AkgLLMAdapter: wraps LLMClient → ConversationAdapter
    task_scaffolder.py        # Generate task directory for AKG workflow mode

op/workflows/
  autoresearch_workflow.py    # LangGraph workflow (AKG glue, single file)
```

### Dependency Direction

```
autoresearch_workflow.py ──imports──→ autoresearch core (agent/, framework/)
autoresearch core ──does NOT import──→ AKG anything

AKG integration via injection:
  llm_adapter  → AgentLoop (llm_adapter= parameter)
  eval_fn      → ExperimentRunner → run_eval_robust (eval_fn= parameter)
```

## 3. Tool Configuration

| Attribute | Value |
|-----------|-------|
| TOOL_NAME | `call_autoresearch_workflow` |
| Use Case | Deep iterative optimization of an existing kernel |
| Output | Optimized kernel code + profile results (latency, speedup) |

### Parameters

| Parameter | Type/Required | Description |
|-----------|--------------|-------------|
| op_name | str (Required) | Operator name |
| task_desc | str (Required) | Framework code (Model/get_inputs/get_init_inputs format) |
| dsl | str (Required) | Target DSL: `triton_cuda`, `triton_ascend`, `torch`, `cuda_c`, `cpp`, `ascendc`, etc. |
| framework | str (Required) | Target framework: `torch` |
| backend | str (Required) | Target backend: `cuda`, `ascend` |
| arch | str (Required) | Target architecture: `a100`, `ascend910b4`, etc. |
| previous_code | str (Optional) | Initial kernel code (if omitted, KernelGen generates seed) |
| max_rounds | int (Optional) | Maximum eval rounds (default: 20) |

## 4. Core Flow

### 4.1 Preflight Validation

The workflow runs a fail-closed preflight stage before AgentLoop starts. If any step fails, the workflow aborts immediately — no eval rounds are wasted.

1. **Acquire worker** (mandatory). No worker available → abort with error.

2. **Validate reference** (`task_desc`):
   - `check_task_desc_static()`: AST-level structure check (class Model, get_inputs, get_init_inputs)
   - `check_task_desc_runtime()`: Actually execute reference on the worker to confirm it runs
   - Either fails → abort. The reference is the ground truth; it must work.

3. **Resolve seed** (initial kernel):
   - **User-supplied kernel** (`previous_code`): Runtime-verified via `KernelVerifier.run()`. If fails, falls through to KernelGen — user's kernel preserved as context.
   - **KernelGen path**: Three-layer validation per attempt: generate → CodeChecker (static) → KernelVerifier (runtime). Errors fed back to KernelGen for retry (up to `gen_retries` times, default 5).

4. **Release worker**. The preflight worker is held for the entire stage and released in a `finally` block.

### 4.2 Knowledge Assembly

DSL knowledge is assembled in two layers:

- **Layer 0 (system prompt, always visible)**: fundamental/reference skills from the Skill system + hardware docs. Kept compact to respect the 40K char context budget.
- **Layer 1 (docs/ directory, agent reads on demand)**: guide skills, example skills, API docs (e.g., triton_ascend API). The agent uses `read_file` to access these as needed.

For DSLs without a skill package (torch, ascendc, swft, tilelang_npuir), falls back to the legacy `_DSL_DOCS_DIR_MAP` docs directory.

### 4.3 Evaluation (eval_fn)

Evaluation uses KernelVerifier as the single source of truth. The workflow constructs an `eval_fn` closure that:

1. Borrows a worker from WorkerManager (per-eval, not held for the whole loop)
2. Reads the current editable file from task_dir
3. Calls `verifier.run()` for correctness verification
4. Calls `verifier.run_profile()` for latency measurement
5. Returns `EvalResult(correctness, metrics, error)`
6. Releases the worker

**Optimizations:**
- **Single eval call per round**: `eval_fn` is called once per round (verify + profile). Profiler's internal `run_times` / `warmup_times` controls repetition.
- **base_time caching**: Reference implementation never changes, so `base_profile` runs once. Subsequent profiles use `use_reference_data + override_base_time_us` to skip redundant baseline measurement.

### 4.4 LLM-Facing Tools

The agent has access to six tools. Note that `quick_check` and `eval` are **not** LLM-facing — they are triggered automatically by TurnExecutor after edits.

| Tool | Description |
|------|-------------|
| `update_plan` | Submit a structured plan with `- [ ]` items. Items are assigned IDs (p1, p2, ...) and the first is activated. |
| `patch_file` | Apply a targeted string replacement to an editable file. Preferred for incremental edits. |
| `write_file` | Full file rewrite. Used for initial generation or major restructuring. |
| `read_file` | Read editable files or docs/ context files. Supports full and range modes. |
| `compact` | Manually trigger context compression when conversation grows too long. |
| `finish` | Explicitly terminate the optimization loop with a summary. |

### 4.5 Plan-Based Execution Model

The agent operates under a **plan state machine** managed by `FeedbackBuilder`. This enforces sequential, accountable execution rather than ad-hoc edits.

**Three phases:**

| Phase | Meaning | Allowed Actions |
|-------|---------|-----------------|
| `no_plan` | Initial state, no plan submitted | Only `update_plan` |
| `active` | One plan item executing | `patch_file`/`write_file` (with matching `plan_item_id`), `read_file` |
| `replanning` | All items settled | `update_plan` (new plan) or `finish` |

**Lifecycle of a plan item:**
1. Agent calls `update_plan(plan="- [ ] Try BLOCK_SIZE=512\n- [ ] Fuse epilogue")` 
2. Items created: `p1` (active), `p2` (pending)
3. Agent edits code with `plan_item_id="p1"` → TurnExecutor runs quick_check → eval
4. Eval result auto-settles p1 (KEEP → `done_ok`, FAIL/DISCARD → `done_fail`)
5. p2 auto-activated → repeat
6. All items settled → phase becomes `replanning` → agent submits new plan or calls `finish`

**Edit blocking**: Edits are rejected unless the agent is in `active` phase with the correct `plan_item_id`. This prevents unstructured edits.

### 4.6 Automatic Post-Edit Pipeline

After any successful edit (patch_file / write_file), TurnExecutor automatically runs:

1. **quick_check** — Syntax validation (py_compile), DSL compliance check, import validation, optional smoke test. Failure → immediate rollback, no eval consumed.
2. **eval** (via `eval_fn`) — Full correctness + profiling. Returns KEEP / DISCARD / FAIL.
3. **auto-settle** — The active plan item is settled based on eval outcome.
4. **feedback injection** — Structured eval results + performance ranking + cumulative diff (on KEEP) injected back into conversation.

### 4.7 Three-State Model

Each eval round produces one of three outcomes:

| State | Condition | Action |
|-------|-----------|--------|
| **KEEP** | Correctness passes + meets constraints + improves over best | Git commit, update best, reset failure counter |
| **DISCARD** | Correctness passes but no improvement | Git rollback, reset failure counter |
| **FAIL** | Correctness fails or crash | Git rollback, increment failure counter |

### 4.8 Auto-Diagnostic Subagent

When the agent accumulates consecutive failures (every `diagnose_suggest_threshold` failures, default 3), the system automatically:

1. Spawns a **diagnostic subagent** — a fresh LLM context with read-only access to code files
2. The subagent analyzes the error context and current code, returning a structured diagnosis (root cause + fix + patterns to avoid)
3. Calls `require_replan()` — **blocks all edits** until the agent submits a new plan
4. Injects the diagnosis into the conversation as a mandatory direction change

This prevents the agent from repeating the same failing approach indefinitely.

### 4.9 Context Compression

Long optimization sessions can exceed the LLM context window. Two compression mechanisms keep the conversation manageable:

- **microcompact**: Automatically truncates older tool results (below `microcompact_min_chars`), keeping only the most recent `microcompact_keep_recent` results intact.
- **auto_compact**: When estimated tokens exceed `context_limit × compression_threshold`, triggers an LLM-based summarization that replaces old messages with a compact summary.
- **compact tool**: The agent can manually trigger compression via the `compact` tool.

### 4.10 Session Persistence

`SessionStore` provides crash-recovery and observability:

- **Snapshots**: Editable files are snapshotted before each turn for atomic rollback on failure.
- **Session save/load**: Full agent state (plan, eval count, failures, messages) persisted for `--resume`.
- **Heartbeat**: PID lock + status file for monitoring running agents.
- **Turn logging**: Every turn appended to JSONL for post-hoc analysis.

## 5. Configuration

AutoResearch uses the standard AKG config system (`load_config` + `build_langgraph_task_config`). Key config keys:

### Task-Level Configuration

| Key | Default | Description |
|-----|---------|-------------|
| `max_step` | 20 | Maximum eval rounds (overridden by `max_rounds` argument) |
| `agent_model_config.coder` | "standard" | LLM model level for the agent |
| `eval_timeout` | 120 | Eval timeout in seconds (per-round) |
| `workflow_timeout` | dynamic | Total workflow timeout. Auto-computed at workflow init as `max(1800, max_rounds * (eval_timeout + 60) + 300)` to ensure the loop has enough time to complete. Can be overridden explicitly in config. |
| `profile_settings.run_times` | 50 | Profiling repetition count |
| `profile_settings.warmup_times` | 5 | Profiling warmup count |

### AgentConfig (Framework Behavior)

Override via `agent.config` in `task.yaml`. Most tasks use defaults.

| Key | Default | Description |
|-----|---------|-------------|
| **Experiment control** | | |
| `max_consecutive_failures` | 10 | Abort after N consecutive failures |
| `max_no_edit_turns` | 3 | Nudge agent after N turns without edits |
| `max_turns_multiplier` | 8 | Safety cap: max LLM calls = max_rounds × this |
| **LLM** | | |
| `llm_max_tokens` | 8192 | LLM response max tokens |
| `thinking_budget` | 8000 | Extended thinking budget (0 = disabled) |
| `call_timeout` | 120.0 | API call timeout (seconds) |
| **Context compression** | | |
| `context_limit` | None | Model context window (None = disable auto-compress) |
| `compression_threshold` | 0.75 | Trigger auto_compact at this fraction of context_limit |
| `microcompact_keep_recent` | 3 | Keep N most recent tool results intact |
| **Diagnostics** | | |
| `diagnose_suggest_threshold` | 3 | Trigger diagnostic subagent every N consecutive failures |
| `subagent_max_iterations` | 15 | Max iterations for diagnostic subagent |
| **Truncation** | | |
| `editable_file_truncate` | 8000 | Editable file content truncation in system prompt |
| `system_context_total_truncate` | 40000 | Total system context char budget |
| `raw_output_tail` | 2048 | Eval raw output tail length in feedback |
| `cumulative_diff_truncate` | 10000 | Cumulative diff length on KEEP |

## 6. Usage

### Via KernelAgent (Production Path)

KernelAgent automatically selects AutoResearch when the user requests deep iterative optimization. The tool call flows through ToolExecutor → `prepare_config()` → `build_initial_state()` → workflow execution.

### Via LangGraphTask (Direct Path)

```python
from akg_agents.op.langgraph_op.task import LangGraphTask
from akg_agents.op.config.config_validator import load_config
from akg_agents.core.worker.manager import register_local_worker
from akg_agents.utils.task_label import resolve_task_label

await register_local_worker([0], backend="cuda", arch="a100")

config = load_config(dsl="triton_cuda", backend="cuda")
config["task_label"] = resolve_task_label(op_name="my_op", parallel_index=1)
config["max_step"] = 20

task = LangGraphTask(
    op_name="my_op",
    task_desc=open("reference.py").read(),
    task_id="my_op_001",
    backend="cuda", arch="a100",
    dsl="triton_cuda", framework="torch",
    config=config,
    workflow="autoresearch",
)

op_name, success, final_state = await task.run()
```

### Via CLI Script

```bash
# Natural language (LLM generates reference, KernelGen generates initial kernel)
python scripts/run_autoresearch.py --desc "fused ReLU + LayerNorm, input shape (32, 1024), fp16" --backend cuda

# Natural language + initial kernel (LLM generates reference, skip KernelGen)
python scripts/run_autoresearch.py --desc "fused ReLU + LayerNorm, input shape (32, 1024), fp16" --kernel path/to/kernel.py --backend cuda

# Reference file (KernelGen generates initial kernel)
python scripts/run_autoresearch.py --ref path/to/reference.py --backend cuda

# Reference + initial kernel (skip all generation)
python scripts/run_autoresearch.py --ref path/to/reference.py --kernel path/to/kernel.py --backend cuda
```

## 7. Comparison with Other Workflows

| Feature | KernelGenOnly | Evolve | AdaptiveSearch | **AutoResearch** |
|---------|--------------|--------|----------------|-----------------|
| Strategy | Single-shot | Population-based | UCB bandit | Agent-driven ReAct |
| Verification | Once | Per-candidate | Per-candidate | Per-round |
| Iterations | 1 | Many (parallel) | Many (parallel) | Many (sequential) |
| Autonomy | None | Template-driven | Template-driven | Full (agent plans, diagnoses, reads docs) |
| Best for | Quick draft | Wide exploration | Strategy selection | Deep optimization |
| DSL knowledge | Static injection | Static injection | Static injection | On-demand (read_file) |
| Failure recovery | None | Population absorbs | Population absorbs | Diagnostic subagent + forced replan |

## 8. Testing

| Test | Path | Coverage |
|------|------|----------|
| LangGraphTask direct | `tests/op/st/test_autoresearch.py` | Ascend: seed → scaffold → AgentLoop → eval_fn → KernelVerifier |
| LangGraphTask direct | `tests/op/st/test_autoresearch_cuda.py` | CUDA variant |
| ToolExecutor production | `tests/op/st/test_autoresearch_toolexecutor.py` | KernelAgent → ToolExecutor → prepare_config → build_initial_state → workflow + config isolation |
| ToolExecutor production | `tests/op/st/test_autoresearch_toolexecutor_cuda.py` | CUDA variant |
