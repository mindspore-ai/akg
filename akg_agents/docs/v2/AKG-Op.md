# akg-op End-to-End User Guide

## Overview

akg-op is the operator optimization Agent in the AKG Agents + OpenCode integration. It generates high-performance operator code.

| Mode | Trigger | Typical Scenario |
|------|---------|-----------------|
| **Single Operator** | Specify a concrete operator or provide source code | "Generate a relu kernel", "Optimize the layernorm in this file" |

---

## End-to-End Flow

```
┌─────────────────────────────────────────────────────────────┐
│  Switch to the akg-op Agent in opencode CLI                 │
│  (Tab key or /agents command)                               │
└────────────────────┬────────────────────────────────────────┘
                     ▼
          ┌─────────────────────┐
          │  Phase 0  Env Setup  │  Check akg_agents availability
          │  & Param Confirm     │  🛑 Confirm framework/backend/arch/dsl
          └─────────┬───────────┘
                    ▼
  ┌─────────────────────────────┐
  │  Phase 1  Build Task Desc    │  Extract operator logic from code
  │  Auto-validate + 🛑 Confirm  │  → {op_name}.py (KernelBench format)
  └─────────────┬───────────────┘
                ▼
  ┌─────────────────────────────┐
  │  Phase 2  Generate Operator  │  Pick a workflow to run
  │  🛑 User picks workflow      │  (kernelgen / adaptive_search / evolve)
  └─────────────┬───────────────┘
                │
            ┌───┴───┐
            │ Fail?  │──── Yes ──→ Output failure report, task ends
            └───┬───┘
                │ No
                ▼
  ┌─────────────────────────────┐
  │  Phase 3  Confirm Result     │  🛑 Show generated_code.py
  │                              │  Accept / Regenerate
  │  Regenerate → Back to Ph. 2  │
  └─────────────┬───────────────┘
                │ Accept
                ▼
  ┌─────────────────────────────┐
  │  Phase 4  Code Integration   │  Copy generated code to work dir
  │  (if source code provided)   │  Backup + import replacement
  └─────────────┬───────────────┘
                ▼
  ┌─────────────────────────────┐
  │  Phase 5  Output Report      │  report.md
  │  Config, results, changes    │
  └─────────────────────────────┘
```

---

## Usage Examples

### Example 1: Single Operator — Generate Only

```
User> Generate a relu kernel, input shape (128, 4096), fp32, on Ascend 910B2C with Triton

Phase 0: Check env (cache hit, skip install) → Confirm params: framework=torch, backend=ascend, arch=ascend910b2c, dsl=triton_ascend
Phase 1: Generate relu.py (KernelBench format) → Validate → User confirms
Phase 2: User picks kernelgen → Run → Success
Phase 3: Show code → User accepts
Phase 4: Copy relu_generated.py to work dir
Phase 5: Output report
```

### Example 2: Single Operator — Optimize Existing Code

```
User> Optimize the layernorm in /path/to/model.py using Triton

Phase 0: Check env (cache hit) → Confirm params
Phase 1: Extract layernorm from model.py → Generate layernorm.py → Validate → User confirms
Phase 2: User picks adaptive_search → Run (silent mode, ~15 min) → Success
Phase 3: Show code → User accepts
Phase 4: Backup model.py → Add "from layernorm_generated import ModelNew" → Save integrated file
Phase 5: Output report with file change records
```

---

## Working Directory

All artifacts are stored under `~/akg_agents_logs/op_{op_name}_{timestamp}_{random_id}/`:

| File | Description |
|------|-------------|
| `{op_name}.py` | KernelBench format task description (reference implementation + input definitions) |
| `{op_name}_generated.py` | The final generated operator code accepted by user |
| `{model}_generated.py` | Integrated source file with `from {op_name}_generated import ModelNew` |
| `output/{workflow}_{n}/` | Full output of each workflow run (code, summary, logs) |
| `backup/` | Original copies of replaced files (for rollback) |
| `report.md` | Final report |

---

## Available Workflows

| Workflow | Strategy | Typical Duration | Best For |
|----------|----------|-----------------|----------|
| `kernelgen` | Iterative generate → verify → fix (subagent) | 1-5 min | Clear requirements, quick results (**default**) |
| `adaptive_search` | UCB adaptive search (silent mode) | 10-30 min | Higher quality, willing to wait |
| `evolve` | Island-model evolutionary algorithm (silent mode) | 15-60 min | Diversity exploration, multi-device parallel |

If unsatisfied with the result, choose a different workflow to regenerate. Previous results are preserved in separate subdirectories.

---

## Architecture

| Component | Type | Description |
|-----------|------|-------------|
| `akg-op` | Agent | Main orchestrator: phase sequencing, user interaction |
| `akg-env-setup` | Skill | Environment check, hardware/framework/DSL detection, dependency install |
| `op-task-extractor` | Skill | Build KernelBench format task file from code or natural language |
| `kernelgen` | SubAgent | Iterative code generation + verification workflow |
| `kernel-generator` | Skill | DSL-aware kernel code generation (used by kernelgen) |
| `kernel-verifier` | Skill | Multi-framework, multi-backend correctness verification (used by kernelgen) |
| `search-workflow` | Skill | Background execution of adaptive_search / evolve workflows |

---

## Notes

- Every phase includes a human confirmation step; nothing runs end-to-end without approval
- On generation failure, the error is reported immediately with no automatic retry
- Original files are always backed up before replacement; restore from `backup/` if needed
- `adaptive_search` and `evolve` run in silent mode (background) with ~1 min polling interval
