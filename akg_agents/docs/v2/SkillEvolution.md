[中文版](./CN/SkillEvolution.md)

# Skill Evolution

## 1. Overview

The Skill Evolution system automatically extracts optimization experience and generates reusable `SKILL.md` documents. It runs as a SubAgent registered as the `call_skill_evolution` tool, callable by `KernelAgent`.

**Two modes**:
- **search_log**: Extract evolution chain diffs from search logs — automated optimization patterns
- **expert_tuning**: Extract human tuning experience from conversation history — "user advice → code change → performance delta" causal chains

**Goal**: Close the loop of "optimize → summarize → reuse" — turn both automated search logs and human expertise into structured knowledge for future kernel generation.

**Architecture**: `SkillEvolutionBase` (`core_v2/agents/`) provides shared capabilities (workspace management, logging utilities). `SkillEvolutionAgent` (`op/agents/`) inherits from the base class and implements the two operator-specific modes.

## 2. search_log Mode

### 2.1 Data Sources

The system reads exactly 3 files from the `logs/` directory:

| File | Content | Key Fields |
|------|---------|------------|
| `verification_results.jsonl` | Verification records | `task_id`, `passed`, `verify_dir`, `dsl`, `backend`, `arch` |
| `{op}/profiling/speed_up_record.txt` | Performance records | `task_id`, `generation_time`, `speedup` |
| `{op}_lineage_graph.md` | Evolution tree table | `task_id`, `parent_id`, `generation` |

For each passed task, the actual implementation code is read from `verify_dir/*_impl.py`.

### 2.2 Pipeline

```
1. collect   — Parse 3 files + read impl code → List[TaskRecord]
2. compress  — Build evolution tree → monotonic stack per path → strip comments → diff
3. LLM      — Best code + evolution diffs → generate SKILL.md body
4. Writer   — YAML frontmatter + body → write SKILL.md
```

### 2.3 Collector

`collect(log_dir, op_name) -> (records, metadata)`

- Parses `verification_results.jsonl` for passed tasks and their `verify_dir`
- Parses `speed_up_record.txt` for `gen_time` and `speedup`
- Parses `lineage_graph.md` table for `parent_id` and `generation`
- Reads `*_impl.py` from each task's `verify_dir` as the code
- Returns a flat list of `TaskRecord` and environment metadata (`dsl`, `backend`, `arch`)

### 2.4 Compressor

`compress(records, metadata) -> CompressedData`

**Best record**: The record with the smallest `gen_time` (fastest execution). Its full code is included in the prompt.

**Monotonic stack evolution chains**:

1. Reconstruct the evolution tree from `parent_id` relationships
2. DFS to collect all root-to-leaf paths
3. For each path, maintain a monotonic stack: only keep nodes where `gen_time` strictly decreases (performance strictly improves)
4. For adjacent nodes in the filtered stack, strip comments then generate unified diff
5. Skip pairs where `MIN_GEN_TIME_IMPROVE_PCT < 0.01` (too small to be meaningful)

The comment stripping (`strip_comments`) removes docstrings, pure comment lines, and inline comments before diffing, eliminating noise from comment rewording.

### 2.5 LLM Analysis

The Jinja2 template (`analyze_search_log.j2`) injects:
- Operator info (name, DSL, backend, architecture)
- Best implementation (full code with gen_time and speedup)
- Evolution chain diffs (comment-stripped, monotonic-filtered)
- Performance summary

The LLM generates the SKILL.md body directly in Markdown, with `skill_name` and `description` as the first two lines.

**Generation goal**: Extract transferable, generalized optimization methodology rather than describing operator-specific characteristics. The document structure is: Task characteristics (define the problem class) → Optimization methods (each as an independent section with conditions, approach, and rationale) → Applicability boundaries.

### 2.6 Writer

Assembles YAML frontmatter (name, description, category, backend, dsl, source) + LLM body → writes to `op/resources/skills/{dsl}/cases/{skill_name}/SKILL.md`.

**Naming convention**: `skill_name` follows the `{dsl}-case-{op-category}-{optimization-detail}` format, e.g. `triton-ascend-case-reduction-amin-large`, `triton-ascend-case-elemwise-broadcast-3d`. `category` is `example`, `source` is `search_log`.

### 2.7 Core Algorithm: Monotonic Stack

```
Original path: A(17us) → B(8us) → C(9us) → D(8us)
Monotonic stack: A(17us) → B(8us)
Diff pairs: (A→B) — only pair where performance strictly improved
```

- Comparison uses `gen_time` (lower is better)
- `seen_pairs` set prevents duplicate diffs when paths share common prefixes
- `MIN_GEN_TIME_IMPROVE_PCT = 0.01` filters out negligible improvements

## 3. expert_tuning Mode

Extracts human tuning experience from conversation history to generate SKILL.md.

**Use case**: The user manually guides optimization during a conversation (e.g., "increase BLOCK_SIZE", "add more warps") and wants to distill these insights into a reusable skill.

### 3.1 Data Source

The collector reads `{conversation_dir}/trace.json` to obtain the conversation tree structure, then uses DFS to find all root-to-leaf paths. Each path becomes an independent branch. For each branch, actions are read from the corresponding `actions/action_history_fact.json` files in path order.

Falls back to node-number sorting (node_001, node_002, ...) when `trace.json` is not available.

**Multi-branch handling**: The conversation tree may contain multiple branches (e.g., root→node_001...019 and root→node_020...029). Each branch is output as an independent timeline segment, separated by branch headers. Branches include full shared prefix nodes to ensure complete causal chains. For single-branch scenarios (linear chains), no extra labels are added and behavior is identical to the previous implementation.

```
root → node_001 → ... → node_019   (Branch 1)
     → node_020 → ... → node_029   (Branch 2)

Output:
  ## Branch 1 (19 nodes)
  ### node_001 Turn 1 — ask_user ...
  ...
  ## Branch 2 (10 nodes)
  ### node_020 Turn 1 — ask_user ...
  ...
```

### 3.2 Incremental LLM Compression

The collector formats each action into a section, then builds the timeline incrementally:

```
accumulated = ""
for section in sections:
    if len(accumulated + section) > threshold (60,000 chars):
        accumulated = LLM_compress(accumulated)  // compress accumulated history
    accumulated += section                        // append new section
```

Before adding each new section, the total length is checked. When it exceeds the threshold, the accumulated portion is compressed by LLM first, then the new section is appended. This handles conversations of any length.

**LLM compression retention rules** (enforced via prompt):

| Category | Handling |
|----------|----------|
| User responses (optimization advice) | **Preserve in full** |
| Generated code from code gen tools | **Preserve in full** |
| Performance data (gen_time, speedup, etc.) | **Preserve in full** |
| Tool execution status | **Preserve in full** |
| Agent messages (explanations, confirmations) | May compress or remove |
| Redundant tool parameters | May compress or remove |
| Repeated error messages | May compress or remove |

### 3.3 Pipeline

```
1. collect          — DFS trace.json for branch paths → read actions in path order → format as section list
2. build_timeline   — Incrementally append sections, LLM-compress accumulated portion when threshold exceeded
3. LLM Analysis     — Timeline → self-analyze causal chains → generate SKILL.md body
4. Writer           — YAML frontmatter + body → write SKILL.md
```

### 3.4 Collector and Timeline Builder

`collect(conversation_dir, op_name) -> (sections, metadata)`

**Responsibility**: Read conversation tree structure and format. Reads the tree from `trace.json` and DFS to find all root-to-leaf paths; falls back to node-number sorting when unavailable. Returns a list of sections (including branch headers for multi-branch scenarios), no analysis or compression.

`build_timeline(sections, llm_fn, max_chars=60000, work_dir="") -> str`

**Responsibility**: Incrementally append sections and compress as needed. Returns the final timeline text. Optional `work_dir` outputs intermediate files for debugging.

### 3.5 LLM Prompt

Template `analyze_expert_tuning.j2` injects the timeline and instructs the LLM to:

1. Identify which turns contain substantive optimization advice (ignoring "confirm" messages)
2. Match code versions to performance data (profile_kernel corresponds to the most recent code generation before it)
3. Extract "user advice → code change → performance delta" causal chains
4. Generate SKILL.md body

**Naming convention**: `skill_name` follows the `{dsl}-exp-{op-category}-{tuning-detail}` format. `source` is `expert_tuning`.

## 4. File Structure

```
core_v2/agents/
└── skill_evolution_base.py     — SkillEvolutionBase (workspace management, logging utilities)

op/tools/skill_evolution/
├── common.py                   — Shared types, utilities, LLM output parsing, SKILL.md writer
├── search_log_utils.py         — search_log mode: collect + compress + to_prompt_vars
├── expert_tuning_utils.py      — expert_tuning mode: collect + build_timeline + to_prompt_vars
└── __init__.py

op/agents/skill_evolution_agent.py — SkillEvolutionAgent (inherits base, search_log / expert_tuning dispatch)

op/resources/prompts/skill_evolution/
├── analyze_search_log.j2       — search_log: structured evolution diffs → LLM
└── analyze_expert_tuning.j2    — expert_tuning: action timeline → LLM

tests/op/st/test_skill_evolution.py — Standalone CLI script (no Agent framework dependency)
```

## 5. Standalone CLI Script

`tests/op/st/test_skill_evolution.py` provides an Agent-framework-free entry point.

```bash
# search_log mode
python akg_agents/tests/op/st/test_skill_evolution.py search_log /path/to/logs relu

# expert_tuning mode
python akg_agents/tests/op/st/test_skill_evolution.py expert_tuning ~/.akg/conversations/cli_xxx relu

# With output directory and model level
python akg_agents/tests/op/st/test_skill_evolution.py expert_tuning /path/to/conv relu -o ./output -m complex
```

| Argument | Description |
|----------|-------------|
| `mode` | `search_log` or `expert_tuning` |
| `log_dir` / `conversation_dir` | Log directory (search_log) or conversation directory (expert_tuning) |
| `op_name` | Operator name (e.g. relu, l1norm) |
| `-o / --output-dir` | SKILL.md output directory (defaults to project skills dir) |
| `-m / --model-level` | LLM model level (default: standard) |

## 6. Workspace

Intermediate files are saved to `{cur_path}/logs/skill_evolution/`:

**search_log mode:**

| File | Content |
|------|---------|
| `collected_data.json` | Task records summary (task_id, parent_id, gen_time, speedup, has_code) |
| `compressed_data.json` | Best record + evolution chains |
| `llm_prompt.txt` | Rendered LLM prompt |
| `llm_response.txt` | Raw LLM output |
| `session.log` | Execution log |
| `result.json` | Final result summary |

**expert_tuning mode:**

| File | Content |
|------|---------|
| `action_timeline.md` | Formatted action timeline (may contain compression markers) |
| `llm_prompt.txt` | Rendered LLM prompt (with timeline) |
| `llm_response.txt` | Raw LLM output |
| `session.log` | Execution log |
| `result.json` | Final result summary |
