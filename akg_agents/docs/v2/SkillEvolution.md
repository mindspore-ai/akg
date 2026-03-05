[中文版](./CN/SkillEvolution.md)

# Skill Evolution

## 1. Overview

The Skill Evolution system automatically extracts optimization experience and generates reusable `SKILL.md` documents. It runs as a SubAgent registered as the `call_skill_evolution` tool, callable by `KernelAgent`.

**Two modes**:
- **Mode A (adaptive_search)**: Extract evolution chain diffs from search logs — automated optimization patterns
- **Mode B (feedback)**: Extract human tuning experience from conversation history — "user advice → code change → performance delta" causal chains

**Goal**: Close the loop of "optimize → summarize → reuse" — turn both automated search logs and human expertise into structured knowledge for future kernel generation.

## 2. Mode A: Adaptive Search

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
1. Collector  — Parse 3 files + read impl code → List[TaskRecord]
2. Compressor — Build evolution tree → monotonic stack per path → strip comments → diff
3. LLM        — Best code + evolution diffs → generate SKILL.md body
4. Writer     — YAML frontmatter + body → write SKILL.md
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

The comment stripping (`_strip_comments`) removes docstrings, pure comment lines, and inline comments before diffing, eliminating noise from comment rewording.

### 2.5 LLM Analysis

The Jinja2 template (`analyze.j2`) injects:
- Operator info (name, DSL, backend, architecture)
- Best implementation (full code with gen_time and speedup)
- Evolution chain diffs (comment-stripped, monotonic-filtered)
- Performance summary

The LLM generates the SKILL.md body directly in Markdown, with `skill_name` and `description` as the first two lines.

**Generation goal**: Extract transferable, generalized optimization methodology rather than describing operator-specific characteristics. The document structure is: Task characteristics (define the problem class) → Optimization methods (each as an independent section with conditions, approach, and rationale) → Applicability boundaries.

### 2.6 Writer

Assembles YAML frontmatter (name, description, category, backend, dsl) + LLM body → writes to `op/resources/skills/{dsl}/cases/{skill_name}/SKILL.md`.

**Naming convention**: `skill_name` follows the `{dsl}-case-{op-category}-{optimization-detail}` format, e.g. `triton-ascend-case-reduction-amin-large`, `triton-ascend-case-elemwise-broadcast-3d`, consistent with hand-written cases. `category` is set to `example`.

### 2.7 Core Algorithm: Monotonic Stack

```
Original path: A(17us) → B(8us) → C(9us) → D(8us)
Monotonic stack: A(17us) → B(8us)
Diff pairs: (A→B) — only pair where performance strictly improved
```

- Comparison uses `gen_time` (lower is better)
- `seen_pairs` set prevents duplicate diffs when paths share common prefixes
- `MIN_GEN_TIME_IMPROVE_PCT = 0.01` filters out negligible improvements

## 3. Mode B: Human Experience Feedback

Extracts human tuning experience from conversation history to generate SKILL.md.

**Use case**: The user manually guides optimization during a conversation (e.g., "increase BLOCK_SIZE", "add more warps") and wants to distill these insights into a reusable skill.

### 3.1 Data Source

The collector reads `{conversation_dir}/nodes/*/actions/action_history_fact.json`, sorted by node number (node_001, node_002, ...), collecting actions from all branches.

A conversation may contain multiple branches (e.g., root→node_001...019 and root→node_020...029). All branches are included to fully cover the optimization history.

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
1. FeedbackCollector  — Read all nodes' actions → sort by node number → format as section list
2. build_timeline     — Incrementally append sections, LLM-compress accumulated portion when threshold exceeded
3. LLM Analysis       — Timeline → self-analyze causal chains → generate SKILL.md body
4. Writer             — YAML frontmatter + body → write SKILL.md
```

### 3.4 Collector

`collect_feedback(conversation_dir, op_name) -> (sections, metadata)`

**Responsibility**: Read and format only. Returns a list of sections (one per action), no analysis or compression.

`build_timeline(sections, llm_fn, max_chars=60000, work_dir="") -> str`

**Responsibility**: Incrementally append sections and compress as needed. Returns the final timeline text. Optional `work_dir` outputs intermediate files for debugging.

For each action, generates a summary based on `tool_name`:

| tool_name | Formatted Content |
|-----------|------------------|
| `ask_user` | Agent message summary + full user response |
| `profile_kernel` | Performance data (gen_time_us, base_time_us, speedup) |
| `call_kernelgen_workflow` etc. | Parameters, user requirements, full generated code |
| `history_summary` | Compressed history summary text |
| Others | Status + output summary |

Returns:
- `timeline_text`: Markdown-formatted timeline, injected directly into LLM prompt
- `metadata`: Environment info dict (op_name, dsl, backend, arch)

### 3.5 LLM Prompt

Template `analyze_feedback.j2` injects the timeline and instructs the LLM to:

1. Identify which turns contain substantive optimization advice (ignoring "confirm" messages)
2. Match code versions to performance data (profile_kernel corresponds to the most recent code generation before it)
3. Extract "user advice → code change → performance delta" causal chains
4. Generate SKILL.md body

## 4. File Structure

```
op/tools/skill_evolution/
├── models.py                — TaskRecord, EvolutionStep, CompressedData (Mode A data models)
├── collector.py             — Mode A: collect(log_dir, op_name) → (records, metadata)
├── compressor.py            — Mode A: compress(records, metadata) → CompressedData
├── feedback_collector.py    — Mode B: collect_feedback(conversation_dir, op_name) → (timeline, metadata)
├── analyzer.py              — Prompt variable conversion + LLM output parsing
├── writer.py                — YAML frontmatter + body → SKILL.md (supports dict metadata)
└── __init__.py

op/agents/skill_evolution_agent.py — Agent orchestration (Mode A / Mode B dispatch)

op/resources/prompts/skill_evolution/
├── analyze.j2               — Mode A: structured evolution diffs → LLM
└── analyze_feedback.j2      — Mode B: action timeline → LLM

tests/op/st/test_skill_evolution.py — Standalone CLI script (no Agent framework dependency)
```

## 5. Standalone CLI Script

`tests/op/st/test_skill_evolution.py` provides an Agent-framework-free entry point.

```bash
# Mode A
python akg_agents/tests/op/st/test_skill_evolution.py adaptive_search /path/to/logs relu

# Mode B
python akg_agents/tests/op/st/test_skill_evolution.py feedback ~/.akg/conversations/cli_xxx relu

# With output directory and model level
python akg_agents/tests/op/st/test_skill_evolution.py feedback /path/to/conv relu -o ./output -m complex
```

| Argument | Description |
|----------|-------------|
| `mode` | `adaptive_search` or `feedback` |
| `log_dir` / `conversation_dir` | Log directory (Mode A) or conversation directory (Mode B) |
| `op_name` | Operator name (e.g. relu, l1norm) |
| `-o / --output-dir` | SKILL.md output directory (defaults to project skills dir) |
| `-m / --model-level` | LLM model level (default: standard) |

## 6. Workspace

Intermediate files are saved to `{cur_path}/logs/skill_evolution/`:

**Mode A (adaptive_search):**

| File | Content |
|------|---------|
| `collected_data.json` | Task records summary (task_id, parent_id, gen_time, speedup, has_code) |
| `compressed_data.json` | Best record + evolution chains |
| `llm_prompt.txt` | Rendered LLM prompt |
| `llm_response.txt` | Raw LLM output |
| `session.log` | Execution log |
| `result.json` | Final result summary |

**Mode B (feedback):**

| File | Content |
|------|---------|
| `action_timeline.md` | Formatted action timeline (may contain compression markers) |
| `llm_prompt.txt` | Rendered LLM prompt (with timeline) |
| `llm_response.txt` | Raw LLM output |
| `session.log` | Execution log |
| `result.json` | Final result summary |
