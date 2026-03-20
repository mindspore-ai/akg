[中文版](./CN/SkillEvolution.md)

# Skill Evolution

## 1. Overview

The Skill Evolution system is a general-purpose experience extraction framework: it automatically distills reusable optimization knowledge from Agent runtime logs and interaction records, generating structured `SKILL.md` documents. This gives Agents the ability to "practice → summarize → reuse" in a closed learning loop.

The current implementation focuses on the operator layer. It runs as a SubAgent registered as the `call_skill_evolution` tool, callable by `KernelAgent` during operator generation/optimization workflows.

**Four modes**:
- **search_log**: Extract evolution chain diffs from search logs — automated optimization patterns
- **expert_tuning**: Extract human tuning experience from conversation history — "user advice → code change → performance delta" causal chains
- **error_fix**: Extract debugging experience from error fix records — "error type → fix strategy"
- **organize**: Consolidate evolved skills under the same DSL by optimization theme — "error type → fix strategy"

**Goal**: Turn automated search logs, human expertise, and debugging experience into structured knowledge for future kernel generation.

**Architecture**: `SkillEvolutionBase` (`core_v2/agents/`) provides shared capabilities (workspace management, logging utilities). `SkillEvolutionAgent` (`op/agents/`) inherits from the base class and implements the four operator-specific modes.

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

Assembles YAML frontmatter (name, description, category, backend, dsl, source) + LLM body → writes to `~/.akg/evolved_skills/{dsl}/evolved-improvement/{skill_name}/SKILL.md`.

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

## 4. error_fix Mode

Extracts "fail → success" error fix records from search logs and accumulates them into a debugging `SKILL.md`.

**Use case**: During the code generation process, code fails multiple times before being successfully fixed. These debugging insights are distilled into reusable skills to help future generation stages avoid similar errors.

### 4.1 Data Source

Shares the same `logs/` directory as `search_log` mode, but focuses on different information:

| File | Content | Key Fields |
|------|---------|------------|
| `verification_results.jsonl` | Verification records (both failures and successes) | `task_id`, `passed`, `verify_dir`, `step` |
| `verify_dir/*_impl.py` | Failed/successful code | Code content |

**Data extraction logic**: For each Task, sort verification records by step and find the first `passed=true` entry. Take the last `passed=false` entry before it as the "failed version". Extract failed code, successful code, and error log.

```
Task verification sequence: step2(fail) → step5(fail) → step8(fail) → step11(pass)
Extraction: failed_code=step8, success_code=step11, error_log=step8's error
```

**Full diff**: Unlike `search_log` mode's 200-line truncation limit, `error_fix` generates untruncated diffs to ensure the LLM sees the complete code change.

**Multi-workflow compatibility**: `error_fix` mode only depends on `verification_results.jsonl` and `verify_dir`. `task_id` is used purely as a grouping key, so it works with adaptive_search (`_Gen1_Task3`), evolve (`1_Island1_Task0`), and kernelgen (`0`) formats alike.

### 4.2 Pipeline

```
1. collect         — Parse verification_results.jsonl → find fail→success pairs per Task
                     → read failed/successful code + error log → full diff (untruncated)
2. LLM Analysis   — Fix cases (error_log + diff) → generate error fix experience
3. LLM Dedup      — If SKILL.md exists, inject existing + new content; LLM outputs only non-redundant items
4. Writer          — First run creates `{dsl}-error-fix/SKILL.md`; later runs append deduplicated incremental content
```

### 4.3 Collector

`collect(log_dir, op_name) -> (records, metadata)`

- Parses `verification_results.jsonl`, groups by `task_id`
- For each Task, sorts by step and finds the last failure before the first success
- Reads failed and successful `*_impl.py` from corresponding `verify_dir`
- Reads the failure step's `error_log` (truncated to the last 1000 chars)
- Generates an untruncated unified diff from failed to successful code
- Returns a list of `SuccessfulFixRecord` and environment metadata

**Data structure**:

```python
@dataclass
class SuccessfulFixRecord:
    task_id: str
    op_name: str
    error_log: str       # Truncated error log
    error_step: int      # Failed step number
    failed_code: str     # Failed version code
    success_code: str    # Successful version code
    diff: str            # Unified diff (untruncated)
    dsl: str
    backend: str
    arch: str
```

### 4.4 LLM Prompt

Template `analyze_error_fix.j2` injects all fix cases (each containing error log and full code diff).

LLM tasks:
1. Classify common errors (with short titles)
2. For each error, provide only **error signature** and **fix method**, with brief code comparisons
3. Merge similar errors and focus on transferable, generalized fix strategies

### 4.5 Dedup and Writer

**Dedup** (`dedup_error_fix.j2`): When `{dsl}-error-fix/SKILL.md` already exists, both the existing body and newly generated content are injected into LLM. The LLM determines which items are new and outputs only the non-redundant incremental content. If everything is duplicate, it outputs "无新增内容" and writing is skipped.

**Writer** (`SkillWriter.write_error_fix`):

- Skill directory name includes DSL prefix: `{dsl}-error-fix` (e.g. `triton-cuda-error-fix`)
- Default output path: `~/.akg/evolved_skills/{dsl}/evolved-fix/{dsl}-error-fix/SKILL.md`
- If `--output-dir DIR` is provided: `DIR/{dsl}-error-fix/SKILL.md`
- If the file does not exist, create it with frontmatter (`name: {dsl}-error-fix`, `description: {dsl}常见错误及修复方法...`, `category: implementation`, `metadata.source: error_fix`)
- If the file already exists, append the deduplicated incremental content (preserving existing frontmatter and body)

```
Run 1: LLM generates → create new SKILL.md
Run 2: LLM generates → compare with existing → append only new items
Run N: Same — continuously accumulate non-redundant debugging experience
```

## 5. organize Mode

Consolidates multiple evolved skills under the same DSL by optimization theme, reducing redundant documents.

**Use case**: After multiple `search_log` and `expert_tuning` runs, `~/.akg/evolved_skills/{dsl}/evolved-improvement/` accumulates many skills. Skills from different operators often contain overlapping optimization techniques. Merging produces fewer, more generalized documents.

### 5.1 Design Constraint

Injecting all skill contents into a single LLM call would overflow the context window. The solution is a **two-phase strategy**:
1. **Summary clustering**: LLM only sees `name + description` (~100 chars each), not full content
2. **Per-cluster merging**: LLM is called per cluster with only the full content of that cluster's skills

### 5.2 Pipeline

```
1. scan            — Scan `evolved-improvement/` under the DSL directory for all SKILL.md (excluding `.archive/`)
2. classify        — Extract name + description → LLM clusters by theme (summaries only, no full content)
3. merge per-cluster — For each cluster with >=2 skills, inject full content for LLM to merge and deduplicate
                       Large clusters (>5) are auto-split into sub-batches with rolling merge
4. archive + write — Archive original skills to .archive/{timestamp}/, write merged SKILL.md
```

### 5.3 Classification Phase

The `classify_skills.j2` template injects only skill `name` and `description`, asking the LLM to group by optimization theme and provide a reason. Output format:

```json
{
  "clusters": [
    {"reason": "These skills all address memory access pattern optimization and bandwidth utilization", "skills": ["skill-a", "skill-c"]},
    {"reason": "These skills focus on compute block size tuning and register pressure control", "skills": ["skill-b"]}
  ]
}
```

### 5.4 Merge Phase

For each cluster containing >= 2 skills:

- `merge_cluster.j2` injects the full content of all cluster skills along with the DSL prefix, instructing the LLM to deduplicate, generalize (remove operator-specific names), unify structure, and produce a `skill_name` (format: `{dsl}-merged-{tuning-feature}`) and `description`
- Large cluster protection: clusters exceeding 5 skills are automatically split into sub-batches. The first 5 are merged, then the result serves as the "existing merged document" for the next batch
- Clusters with only 1 skill are left unchanged

### 5.5 Merged SKILL.md Format

The merged skill name and description are produced by the merge-phase LLM. Example frontmatter:

```yaml
---
name: triton-cuda-merged-memory-access-optimization
description: "Merged optimization methodology..."
category: example
metadata:
  source: merged
  backend: cuda
  dsl: triton-cuda
---
```

- `name` and `description` are generated by the merge-phase LLM
- `source` is `merged`

### 5.6 Incremental Merging

Newly generated skills continue to be written under `~/.akg/evolved_skills/{dsl}/`: `error_fix` outputs to `evolved-fix/`, while `search_log` and `expert_tuning` output to `evolved-improvement/`. The next time `organize` runs:
- Both existing merged skills and newly added individual skills participate in clustering
- If a new skill is clustered with an existing merged skill → incremental merge using the merged skill as base
- If a new skill forms its own cluster → kept as an independent skill

## 6. File Structure

```
core_v2/agents/
└── skill_evolution_base.py     — SkillEvolutionBase (workspace management, logging utilities)

op/tools/skill_evolution/
├── common.py                   — Shared types, utilities, LLM output parsing, SKILL.md writer
├── search_log_utils.py         — search_log mode: collect + compress + to_prompt_vars
├── expert_tuning_utils.py      — expert_tuning mode: collect + build_timeline + to_prompt_vars
├── error_fix_utils.py          — error_fix mode: collect + to_prompt_vars
├── merge_utils.py              — organize mode: scan, classify parsing, archive, merge writing
└── __init__.py


op/agents/skill_evolution_agent.py — SkillEvolutionAgent (inherits base, four-mode dispatch)

op/resources/prompts/skill_evolution/
├── analyze_search_log.j2       — search_log: structured evolution diffs → LLM
├── analyze_expert_tuning.j2    — expert_tuning: action timeline → LLM
├── analyze_error_fix.j2        — error_fix: fix cases → LLM
├── dedup_error_fix.j2          — error_fix: existing + new content → LLM dedup, output increments only
├── classify_skills.j2          — organize: name + description → LLM clustering
└── merge_cluster.j2            — organize: cluster skill contents → LLM merge and dedup

examples/kernel_related/skill_evolution/
├── run_skill_evolution.py      — Standalone CLI script (no Agent framework dependency)
├── run_ab_test.py              — A/B test batch runner
├── ab_test_utils.py            — A/B test utility functions
└── tracking.md                 — Experiment tracking document
```

## 7. Standalone CLI Script

`examples/kernel_related/skill_evolution/run_skill_evolution.py` provides an Agent-framework-free entry point.

```bash
# search_log mode
python examples/kernel_related/skill_evolution/run_skill_evolution.py search_log /path/to/logs relu

# expert_tuning mode
python examples/kernel_related/skill_evolution/run_skill_evolution.py expert_tuning ~/.akg/conversations/cli_xxx relu

# error_fix mode
python examples/kernel_related/skill_evolution/run_skill_evolution.py error_fix /path/to/logs matmul

# organize mode (CLI subcommand: organize)
python examples/kernel_related/skill_evolution/run_skill_evolution.py organize triton_cuda
python examples/kernel_related/skill_evolution/run_skill_evolution.py organize triton_cuda --skills-dir /path/to/evolved -o ./merged

# With output directory and model level
python examples/kernel_related/skill_evolution/run_skill_evolution.py error_fix /path/to/logs matmul -o ./output -m complex
```

| Argument | Description |
|----------|-------------|
| `mode` | `search_log`, `expert_tuning`, `error_fix`, or `organize` (conceptually `organize` in Agent calls and descriptions) |
| `log_dir` / `conversation_dir` | Log directory (search_log / error_fix) or conversation directory (expert_tuning) |
| `op_name` | Operator name (e.g. relu, l1norm, matmul) |
| `-o / --output-dir` | SKILL.md output directory |
| `-m / --model-level` | LLM model level (default: standard) |

## 8. Workspace

In Agent mode, intermediate files are saved to `{cur_path}/logs/skill_evolution/`. In CLI mode, the default location is `~/.akg/skill_evolution/{mode}_{op_name}/` (overridable with `-o`):

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

**error_fix mode:**

| File | Content |
|------|---------|
| `collected_fix_records.json` | Fix records summary (task_id, error_step, has_conductor, diff_lines) |
| `llm_prompt.txt` | Rendered LLM prompt (with fix cases) |
| `llm_response.txt` | Raw LLM output |
| `session.log` | Execution log |
| `result.json` | Final result summary |

**organize mode:**

| File | Content |
|------|--------|
| `skill_summaries.json` | All skills name + description summaries |
| `classify_prompt.txt` | Classification LLM prompt |
| `classify_response.txt` | Classification LLM output |
| `clusters.json` | Parsed clustering result |
| `merge_{theme}_prompt.txt` | Per-cluster merge LLM prompt |
| `merge_{theme}_response.txt` | Per-cluster merge LLM output |
| `result.json` | Final result summary |

## 9. Workflow Compatibility

Different workflows produce logs with different naming conventions:

| Workflow | File Naming Example | Characteristics |
|----------|-------------------|-----------------|
| adaptive_search | `Iteration_Gen1_Task3_Step02_{op}_coder_result.txt` | `Gen` + `Task` hierarchy |
| evolve | `Iteration1_Island0_Task0_Step05_{op}_coder_prompt.txt` | `Island` + `Task` hierarchy |
| kernelgen | `Iteration0_Step01_{op}_kernel_gen_prompt.txt` | No Task/Island hierarchy |

Mode compatibility:

| Mode | adaptive_search | evolve | kernelgen | Notes |
|------|:-:|:-:|:-:|-------|
| **error_fix** | Y | Y | Y | Only depends on `verification_results.jsonl` + `verify_dir`, independent of file naming |
| **search_log** | Y | - | - | Depends on `lineage_graph.md` + `speed_up_record.txt`, currently only produced by adaptive_search |
| **organize** | - | - | - | Does not depend on any logs; processes existing evolved skill files |
| **expert_tuning** | - | - | - | Depends on conversation directory `trace.json` / `action_history_fact.json`|

> **Note**: To extend `search_log` mode to evolve/kernelgen workflows, additional parsing logic for their lineage and performance files would be needed. `error_fix` mode is already natively compatible with any workflow that produces `verification_results.jsonl`.