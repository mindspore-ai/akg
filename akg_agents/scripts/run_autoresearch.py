#!/usr/bin/env python3
"""Run autoresearch on a custom operator task.

Reference (--desc or --ref) is required.  --kernel is optional — KernelGen
generates the initial kernel from the reference if not provided.

  python scripts/run_autoresearch.py --desc "fused ReLU + LayerNorm, (32,1024), fp16" --backend cuda
  python scripts/run_autoresearch.py --ref reference.py --backend cuda
  python scripts/run_autoresearch.py --ref reference.py --kernel kernel.py --backend cuda
"""

import argparse
import asyncio
import os
import re
import logging

from akg_agents.op.langgraph_op.task import LangGraphTask
from akg_agents.op.config.config_validator import load_config
from akg_agents.core.worker.manager import register_local_worker
from akg_agents.utils.task_label import resolve_task_label

logger = logging.getLogger(__name__)

# ---- Natural language → task_desc generation ----

_GENERATE_PROMPT = """\
You are a PyTorch expert. Given a natural language description of a tensor operation,
generate a complete Python reference implementation following this EXACT format:

```python
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, ...) -> torch.Tensor:
        # implement the operation here
        ...

def get_inputs():
    # create input tensors with the specified shapes, dtypes, and device
    return [...]

def get_init_inputs():
    return []
```

Rules:
- Model.forward() implements the EXACT operation described, nothing more
- get_inputs() creates random tensors matching the specified shapes, dtypes
- Device must be '{device}'
- Output ONLY the Python code block, no explanation
- Do NOT add any optimization — this is the reference (baseline) implementation

Operator description:
{description}
"""


async def generate_task_desc(
    description: str, device: str,
    model_level: str = "standard", gen_retries: int = 5,
) -> str:
    """Generate and validate task_desc from natural language.

    Retries on structural validation failure — the LLM sometimes omits
    required symbols.  Each retry gets a fresh LLM call (no error feedback
    needed; the prompt is deterministic, variance comes from sampling).
    """
    from akg_agents.core_v2.llm.factory import create_llm_client

    llm = create_llm_client(model_level=model_level)
    prompt = _GENERATE_PROMPT.format(description=description, device=device)

    last_error = ""
    for attempt in range(1, gen_retries + 1):
        print(f"[generate_task_desc] Generating reference "
              f"(attempt {attempt}/{gen_retries}) …")
        result = await llm.generate(
            messages=[{"role": "user", "content": prompt}],
            stream=False,
        )
        text = result.get("content", "")

        # Extract code block
        m = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
        code = m.group(1).strip() if m else text.strip()

        # Validate structure (AST-level, same as --ref path)
        try:
            _validate_ref_static(code, "LLM output")
            print(f"[generate_task_desc] Reference validated OK")
            return code
        except ValueError as e:
            last_error = str(e)
            print(f"[generate_task_desc] Validation failed: {last_error}")

    raise ValueError(
        f"Failed to generate valid reference after {gen_retries} attempts "
        f"({last_error}). Please write a reference.py manually and use "
        f"--ref instead of --desc."
    )


def derive_op_name(description: str) -> str:
    """Derive a short op_name from description."""
    # Take first few meaningful words, snake_case
    words = re.findall(r"[a-zA-Z]+", description)[:4]
    name = "_".join(w.lower() for w in words)
    return name or "custom_op"


def _validate_ref_static(code: str, source: str):
    """AST-level validation of reference code structure.

    Same checks as KernelVerifier.check_task_desc_static(): parse the AST
    and verify class Model, get_inputs, get_init_inputs exist as top-level
    definitions.  Runs at CLI layer for early failure before workflow starts.
    """
    import ast as _ast
    try:
        tree = _ast.parse(code)
    except SyntaxError as e:
        raise ValueError(f"Reference from {source} has syntax error: {e}")

    names = {
        node.name for node in tree.body
        if isinstance(node, (_ast.ClassDef, _ast.FunctionDef))
    }
    required = {"Model": "class Model", "get_inputs": "get_inputs()",
                "get_init_inputs": "get_init_inputs()"}
    missing = [label for name, label in required.items() if name not in names]
    if missing:
        raise ValueError(f"Reference from {source} missing: {', '.join(missing)}")


def _read_and_validate_ref(path: str) -> str:
    """Read reference file and validate structure."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Reference file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        code = f.read()
    _validate_ref_static(code, path)
    return code


async def main():
    parser = argparse.ArgumentParser(description="Run autoresearch optimization")

    # Reference source: natural language XOR file
    ref_group = parser.add_mutually_exclusive_group(required=True)
    ref_group.add_argument("--desc", type=str,
                           help="Natural language description → LLM generates reference")
    ref_group.add_argument("--ref", type=str,
                           help="Path to reference.py (Model/get_inputs format)")
    # Optional initial kernel (skips KernelGen if provided)
    parser.add_argument("--kernel", type=str, default=None,
                        help="Path to initial kernel file (skips KernelGen)")

    # Op config
    parser.add_argument("--op-name", type=str, default=None,
                        help="Operator name (auto-derived from --desc if omitted)")
    parser.add_argument("--dsl", type=str, default=None,
                        choices=["triton_ascend", "triton_cuda", "torch",
                                 "cuda_c", "cpp", "ascendc", "tilelang_cuda"])
    parser.add_argument("--backend", type=str, default=None,
                        choices=["ascend", "cuda", "cpu"])
    parser.add_argument("--arch", type=str, default=None)
    parser.add_argument("--framework", type=str, default="torch")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--max-rounds", type=int, default=20)
    parser.add_argument("--gen-retries", type=int, default=5,
                        help="Max retries for code generation (reference and seed)")

    args = parser.parse_args()

    # Backend presets
    _BACKEND_PRESETS = {
        "ascend":  {"dsl": "triton_ascend", "backend": "ascend", "arch": "ascend910b4"},
        "cuda":    {"dsl": "triton_cuda",   "backend": "cuda",   "arch": "a100"},
        "cpu":     {"dsl": "cpp",           "backend": "cpu",    "arch": "x86_64"},
    }
    preset_key = args.backend or (
        "cuda" if args.dsl and "cuda" in args.dsl else
        "cpu" if args.dsl == "cpp" else
        "ascend"
    )
    preset = _BACKEND_PRESETS.get(preset_key, _BACKEND_PRESETS["ascend"])
    if args.dsl is None:
        args.dsl = preset["dsl"]
    if args.backend is None:
        args.backend = preset["backend"]
    if args.arch is None:
        args.arch = preset["arch"]

    device_map = {"ascend": "npu", "cuda": "cuda", "cpu": "cpu"}
    device = device_map.get(args.backend, "npu")

    # ---- Resolve reference (task_desc) ----
    if args.desc:
        print(f"[run_autoresearch] Generating reference from description...")
        print(f"  \"{args.desc}\"")
        task_desc = await generate_task_desc(args.desc, device=device,
                                                gen_retries=args.gen_retries)
        print(f"[run_autoresearch] Reference generated OK")
    else:
        task_desc = _read_and_validate_ref(args.ref)
        print(f"[run_autoresearch] Reference loaded: {args.ref}")

    # ---- Resolve initial kernel (previous_code) ----
    previous_code = ""
    if args.kernel:
        if not os.path.isfile(args.kernel):
            raise FileNotFoundError(f"Kernel file not found: {args.kernel}")
        with open(args.kernel, "r", encoding="utf-8") as f:
            previous_code = f.read()
        print(f"[run_autoresearch] Initial kernel loaded: {args.kernel}")

    op_name = args.op_name or (derive_op_name(args.desc) if args.desc else "custom_op")

    # ---- Run ----
    await register_local_worker([args.device_id], backend=args.backend, arch=args.arch)

    config = load_config(dsl=args.dsl, backend=args.backend)
    config["task_label"] = resolve_task_label(op_name=op_name, parallel_index=1)
    config["max_step"] = args.max_rounds

    task = LangGraphTask(
        op_name=op_name,
        task_desc=task_desc,
        task_id=f"{op_name}_001",
        backend=args.backend,
        arch=args.arch,
        dsl=args.dsl,
        config=config,
        framework=args.framework,
        workflow="autoresearch",
        previous_code=previous_code,
    )

    result_op_name, success, final_state = await task.run()

    print(f"\n{'=' * 60}")
    print(f"Op: {result_op_name}  |  Result: {'SUCCESS' if success else 'FAILED'}")
    if success:
        profile = final_state.get("profile_res", {})
        print(f"gen_time:  {profile.get('gen_time', '?')} us")
        print(f"base_time: {profile.get('base_time', '?')} us")
        print(f"speedup:   {profile.get('speedup', '?')}")
        print(f"\nFinal kernel:\n{final_state.get('coder_code', '')}")
    else:
        print(f"Error: {final_state.get('verifier_error', '')}")


if __name__ == "__main__":
    asyncio.run(main())
