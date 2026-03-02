[中文版](./CN/SkillEvolution.md)

# Skill Evolution

## 1. Overview

The Skill Evolution system automatically extracts optimization experience from `adaptive_search` logs and generates reusable `SKILL.md` documents. It runs as a SubAgent registered as the `call_skill_evolution` tool, callable by `KernelAgent`.

**Goal**: Close the loop of "search → summarize → reuse" — turn search logs into structured knowledge for future kernel generation.

## 2. Data Sources

The system reads exactly 3 files from the `logs/` directory:

| File | Content | Key Fields |
|------|---------|------------|
| `verification_results.jsonl` | Verification records | `task_id`, `passed`, `verify_dir`, `dsl`, `backend`, `arch` |
| `{op}/profiling/speed_up_record.txt` | Performance records | `task_id`, `generation_time`, `speedup` |
| `{op}_lineage_graph.md` | Evolution tree table | `task_id`, `parent_id`, `generation` |

For each passed task, the actual implementation code is read from `verify_dir/*_impl.py`.

## 3. Pipeline

```
1. Collector  — Parse 3 files + read impl code → List[TaskRecord]
2. Compressor — Build evolution tree → monotonic stack per path → strip comments → diff
3. LLM        — Best code + evolution diffs → generate SKILL.md body
4. Writer     — YAML frontmatter + body → write SKILL.md
```

### 3.1 Collector

`collect(log_dir, op_name) -> (records, metadata)`

- Parses `verification_results.jsonl` for passed tasks and their `verify_dir`
- Parses `speed_up_record.txt` for `gen_time` and `speedup`
- Parses `lineage_graph.md` table for `parent_id` and `generation`
- Reads `*_impl.py` from each task's `verify_dir` as the code
- Returns a flat list of `TaskRecord` and environment metadata (`dsl`, `backend`, `arch`)

### 3.2 Compressor

`compress(records, metadata) -> CompressedData`

**Best record**: The record with the smallest `gen_time` (fastest execution). Its full code is included in the prompt.

**Monotonic stack evolution chains**:

1. Reconstruct the evolution tree from `parent_id` relationships
2. DFS to collect all root-to-leaf paths
3. For each path, maintain a monotonic stack: only keep nodes where `gen_time` strictly decreases (performance strictly improves)
4. For adjacent nodes in the filtered stack, strip comments then generate unified diff
5. Skip pairs where `MIN_GEN_TIME_IMPROVE_PCT < 0.01` (too small to be meaningful)

The comment stripping (`_strip_comments`) removes docstrings, pure comment lines, and inline comments before diffing, eliminating noise from comment rewording.

### 3.3 LLM Analysis

The Jinja2 template (`analyze.j2`) injects:
- Operator info (name, DSL, backend, architecture)
- Best implementation (full code with gen_time and speedup)
- Evolution chain diffs (comment-stripped, monotonic-filtered)
- Performance summary

The LLM generates the SKILL.md body directly in Markdown, with `skill_name` and `description` as the first two lines.

**Generation goal**: Extract transferable, generalized optimization methodology rather than describing operator-specific characteristics. The document structure is: Task characteristics (define the problem class) → Optimization methods (each as an independent section with conditions, approach, and rationale) → Applicability boundaries.

### 3.4 Writer

Assembles YAML frontmatter (name, description, category, backend, dsl) + LLM body → writes to `op/resources/skills/{dsl}/cases/{skill_name}/SKILL.md`.

**Naming convention**: `skill_name` follows the `{dsl}-case-{op-category}-{optimization-detail}` format, e.g. `triton-ascend-case-reduction-amin-large`, `triton-ascend-case-elemwise-broadcast-3d`, consistent with hand-written cases. `category` is set to `example`.

## 4. Core Algorithm: Monotonic Stack

```
Original path: A(17us) → B(8us) → C(9us) → D(8us)
Monotonic stack: A(17us) → B(8us)
Diff pairs: (A→B) — only pair where performance strictly improved
```

- Comparison uses `gen_time` (lower is better)
- `seen_pairs` set prevents duplicate diffs when paths share common prefixes
- `MIN_GEN_TIME_IMPROVE_PCT = 0.01` filters out negligible improvements

## 5. File Structure

```
op/tools/skill_evolution/
├── models.py      — TaskRecord, EvolutionStep, CompressedData
├── collector.py   — collect(log_dir, op_name) → (records, metadata)
├── compressor.py  — compress(records, metadata) → CompressedData
├── analyzer.py    — Prompt variable conversion + LLM output parsing
├── writer.py      — YAML frontmatter + body → SKILL.md
└── __init__.py

op/agents/skill_evolution_agent.py — Agent orchestration

op/resources/prompts/skill_evolution/analyze.j2 — LLM prompt template

tests/op/st/test_skill_evolution.py — Standalone CLI script (no Agent framework dependency)
```

## 6. Standalone CLI Script

`tests/op/st/test_skill_evolution.py` provides an Agent-framework-free entry point that directly reuses the collect → compress → LLM → write pipeline.

```bash
# Basic usage
python akg_agents/tests/op/st/test_skill_evolution.py <log_dir> <op_name>

# With output directory and model level
python akg_agents/tests/op/st/test_skill_evolution.py /path/to/logs relu -o ./output -m complex
```

| Argument | Description |
|----------|-------------|
| `log_dir` | adaptive_search log directory (node logs path) |
| `op_name` | Operator name (e.g. relu, l1norm) |
| `-o / --output-dir` | SKILL.md output directory (defaults to project skills dir) |
| `-m / --model-level` | LLM model level (default: standard) |

## 7. Workspace

Intermediate files are saved to `{cur_path}/logs/skill_evolution/`:

| File | Content |
|------|---------|
| `collected_data.json` | Task records summary (task_id, parent_id, gen_time, speedup, has_code) |
| `compressed_data.json` | Best record + evolution chains |
| `llm_prompt.txt` | Rendered LLM prompt |
| `llm_response.txt` | Raw LLM output |
| `session.log` | Execution log |
| `result.json` | Final result summary |
