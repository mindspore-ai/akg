# op-optimizer End-to-End User Guide

## Overview

op-optimizer is the operator optimization Agent in the AKG Agents + OpenCode integration. It generates high-performance operator code and supports two modes:

| Mode | Trigger | Typical Scenario |
|------|---------|-----------------|
| **Single Operator** | Specify a concrete operator or provide source code | "Generate a relu kernel", "Optimize the layernorm in this file" |
| **Fusion** | Request model-level fusion analysis | "Analyze Qwen2 for operator fusion opportunities" |

---

## End-to-End Flow

```
┌─────────────────────────────────────────────────────────────┐
│  Switch to the op-optimizer Agent in opencode CLI            │
│  (Tab key or /agents command)                                │
└────────────────────┬────────────────────────────────────────┘
                     ▼
          ┌─────────────────────┐
          │  Phase 0  Env Setup  │  Check akg_agents availability
          │  & Param Confirm     │  🛑 Confirm framework/backend/arch/dsl
          └─────────┬───────────┘
                    ▼
        ┌───────────────────────┐
        │  Fusion analysis       │
        │  needed?               │
        └───┬───────────────┬───┘
            │ Yes           │ No
            ▼               │
   ┌────────────────┐       │
   │  Phase 1       │       │
   │  Fusion        │       │
   │  Analysis      │       │
   │  🛑 User picks │       │
   │  opportunities │       │
   └───────┬────────┘       │
           │                │
           ▼                ▼
  ┌─────────────────────────────┐
  │  Phase 2  Build Task Desc    │  Extract operator logic from code
  │  Auto-validate + 🛑 Confirm  │  → {op_name}.py (KernelBench format)
  └─────────────┬───────────────┘
                ▼
  ┌─────────────────────────────┐
  │  Phase 3  Generate Operator  │  Pick a workflow to run
  │  🛑 User picks workflow      │  (kernelgen / adaptive_search / evolve)
  │  Foreground execution        │  Takes 1-60 minutes
  └─────────────┬───────────────┘
                │
            ┌───┴───┐
            │ Fail?  │──── Yes ──→ Output failure report, task ends
            └───┬───┘
                │ No
                ▼
  ┌─────────────────────────────┐
  │  Phase 4  Confirm Result     │  🛑 Show generated_code.py
  │                              │  User chooses:
  │  Accept ───→ Save + Replace* │    Accept / Regenerate with different workflow
  │  Regenerate → Back to Ph. 3  │
  └─────────────┬───────────────┘
                │ Accept
                ▼
  ┌─────────────────────────────┐
  │  Phase 5  Output Report      │  report.md
  │  Config, results, changes    │
  └─────────────────────────────┘

  (*) If source code was provided, the original file is backed up
      and the operator implementation is replaced automatically.
```

---

## Usage Examples

### Example 1: Single Operator — Generate Only

```
User> Generate a relu kernel, input shape (128, 4096), fp32, on A100 with Triton

Phase 0: Check env (cache hit, skip install) → Confirm params: framework=torch, backend=cuda, arch=a100, dsl=triton_cuda
Phase 2: Generate relu.py (KernelBench format) → User confirms
Phase 3: User picks kernelgen → Run → Success
Phase 4: Show code → User accepts
Phase 5: Output report, generated code saved as relu_generated.py
```

### Example 2: Single Operator — Optimize Existing Code

```
User> Optimize the layernorm in /path/to/model.py using Triton

Phase 0: Check env (cache hit, skip install) → Confirm params: framework=torch, backend=cuda, arch=a100, dsl=triton_cuda
Phase 2: Extract layernorm from model.py → Generate layernorm.py → User confirms
Phase 3: User picks adaptive_search → Run → Success
Phase 4: Show code → User accepts → Backup model.py → Replace layernorm implementation
Phase 5: Output report with file change records
```

### Example 3: Fusion Mode (First Use, No Env Cache)

```
User> Analyze Qwen2 model for operator fusion opportunities

Phase 0: Check env (no cache → detect akg_cli → collect hardware/framework → write cache)
         → Confirm params: framework=torch, backend=ascend, arch=ascend910b4, dsl=triton_ascend
Phase 1: Analyze Qwen2 forward → Found 3 fusion opportunities → User selects 2
  For each selected opportunity:
    Phase 2: Build fused operator task description → User confirms
    Phase 3: Generate fused operator
    Phase 4: Show code → Accept → Backup → Replace model code
Phase 5: Output summary report
```

---

## Working Directory

All artifacts are stored under `~/akg_agents_logs/op_{op_name}_{timestamp}_{random_id}/`:

| File | Description |
|------|-------------|
| `{op_name}.py` | KernelBench format task description (reference implementation + input definitions) |
| `{op_name}_generated.py` | The final generated operator code accepted by user |
| `output/{workflow}_{n}/` | Full output of each workflow run (code, summary, logs) |
| `backup/` | Original copies of replaced files (for rollback) |
| `report.md` | Final report |

---

## Available Workflows

| Workflow | Strategy | Typical Duration | Best For |
|----------|----------|-----------------|----------|
| `kernelgen` | Iterative generate + verify + orchestrate | 1-5 min | Clear requirements, quick results (**default**) |
| `adaptive_search` | UCB adaptive search | 10-30 min | Higher quality, willing to wait |
| `evolve` | Island-model evolutionary algorithm | 15-60 min | Diversity exploration, multi-device parallel |

If unsatisfied with the result, choose a different workflow to regenerate. Previous results are preserved in separate subdirectories.

---

## Notes

- Every phase includes a human confirmation step; nothing runs end-to-end without approval
- On generation failure, the error is reported immediately with no automatic retry
- Original files are always backed up before replacement; restore from `backup/` if needed
- Long-running workflows execute in the foreground; press Esc to interrupt at any time
