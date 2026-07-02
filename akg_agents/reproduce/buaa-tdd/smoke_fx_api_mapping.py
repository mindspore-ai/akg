#!/usr/bin/env python3
"""Smoke test for FX graph node -> PyTorch API mapping.

This script does not call an LLM or Qdrant. It explains what the current
KernelBench FX metadata path sees, how each node is classified, and which
canonical PyTorch APIs would be used as API-database retrieval sources.
"""

from __future__ import annotations

import argparse
import glob
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set


SCRIPT_DIR = Path(__file__).resolve().parent
AKG_AGENTS_ROOT = SCRIPT_DIR.parent.parent
PYTHON_ROOT = AKG_AGENTS_ROOT / "python"

SAMPLE_TASK = r'''
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(8, 64, 3)
        self.max_pool = nn.MaxPool2d(4)

    def forward(self, x):
        x = self.conv(x)
        x = torch.tanh(x)
        x = x * 2.0
        x = x + 1.0
        x = self.max_pool(x)
        return x

batch_size = 2
in_channels = 8
height = width = 16

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return []
'''

IGNORED_OPS = {"placeholder", "output", "get_attr"}

MODULE_TYPE_TO_API: Dict[str, str] = {
    "Conv1d": "torch.nn.Conv1d",
    "Conv2d": "torch.nn.Conv2d",
    "Conv3d": "torch.nn.Conv3d",
    "ConvTranspose1d": "torch.nn.ConvTranspose1d",
    "ConvTranspose2d": "torch.nn.ConvTranspose2d",
    "ConvTranspose3d": "torch.nn.ConvTranspose3d",
    "Linear": "torch.nn.Linear",
    "MaxPool1d": "torch.nn.MaxPool1d",
    "MaxPool2d": "torch.nn.MaxPool2d",
    "MaxPool3d": "torch.nn.MaxPool3d",
    "AvgPool1d": "torch.nn.AvgPool1d",
    "AvgPool2d": "torch.nn.AvgPool2d",
    "AvgPool3d": "torch.nn.AvgPool3d",
    "AdaptiveAvgPool1d": "torch.nn.AdaptiveAvgPool1d",
    "AdaptiveAvgPool2d": "torch.nn.AdaptiveAvgPool2d",
    "AdaptiveAvgPool3d": "torch.nn.AdaptiveAvgPool3d",
    "BatchNorm1d": "torch.nn.BatchNorm1d",
    "BatchNorm2d": "torch.nn.BatchNorm2d",
    "BatchNorm3d": "torch.nn.BatchNorm3d",
    "LayerNorm": "torch.nn.LayerNorm",
    "GroupNorm": "torch.nn.GroupNorm",
    "ReLU": "torch.nn.ReLU",
    "GELU": "torch.nn.GELU",
    "SiLU": "torch.nn.SiLU",
    "Sigmoid": "torch.nn.Sigmoid",
    "Tanh": "torch.nn.Tanh",
    "Dropout": "torch.nn.Dropout",
}

TORCH_FUNCTION_NAMES = {
    "abs",
    "acos",
    "add",
    "amax",
    "amin",
    "asin",
    "atan",
    "bmm",
    "cat",
    "clamp",
    "cos",
    "div",
    "einsum",
    "erf",
    "exp",
    "gelu",
    "log",
    "matmul",
    "max",
    "mean",
    "min",
    "mul",
    "pow",
    "relu",
    "rsqrt",
    "sigmoid",
    "sin",
    "softmax",
    "sqrt",
    "sub",
    "sum",
    "tanh",
    "truediv",
    "where",
}

BUILTIN_OPERATOR_TO_TORCH = {
    "add": "torch.add",
    "mul": "torch.mul",
    "sub": "torch.sub",
    "truediv": "torch.div",
    "floordiv": "torch.div",
    "matmul": "torch.matmul",
    "pow": "torch.pow",
}

TENSOR_METHODS = {
    "contiguous",
    "expand",
    "flatten",
    "permute",
    "reshape",
    "sum",
    "transpose",
    "unsqueeze",
    "view",
}


@dataclass(frozen=True)
class MappingResult:
    status: str
    api: str = ""
    reason: str = ""


def _resolve_task_file(path_arg: Optional[str]) -> Optional[Path]:
    if not path_arg:
        return None
    raw = Path(path_arg).expanduser()
    if raw.exists():
        return raw
    base = raw if raw.is_absolute() else AKG_AGENTS_ROOT / raw
    if base.exists():
        return base
    matches = sorted(glob.glob(str(base)))
    if not matches:
        matches = sorted(glob.glob(path_arg))
    if not matches:
        raise FileNotFoundError(f"task file not found: {path_arg}")
    return Path(matches[0])


def _load_task(path_arg: Optional[str]) -> str:
    path = _resolve_task_file(path_arg)
    if path is None:
        return SAMPLE_TASK
    text = path.read_text(encoding="utf-8")
    match = re.search(r"```python\s*(.*?)```", text, re.S)
    return match.group(1) if match else text


def _dedup_keep_order(values: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _target_text(node: dict) -> str:
    return " ".join(
        str(node.get(key) or "")
        for key in ("target", "raw_target", "canonical_target")
    ).lower()


def _find_function_name(text: str) -> Optional[str]:
    for name in sorted(TORCH_FUNCTION_NAMES, key=len, reverse=True):
        if re.search(rf"(?<![a-z0-9_]){re.escape(name)}(?![a-z0-9_])", text):
            return name
    return None


def map_fx_node_to_api(node: dict) -> MappingResult:
    op = str(node.get("op") or "")
    if op in IGNORED_OPS:
        return MappingResult("ignored", reason=f"{op} is not a compute API source")

    if op == "call_module":
        module_type = str(node.get("module_type") or "")
        api = MODULE_TYPE_TO_API.get(module_type)
        if api:
            return MappingResult("mapped", api=api, reason=f"module_type={module_type}")
        if module_type:
            return MappingResult("unknown", reason=f"unmapped torch.nn module_type={module_type}")
        return MappingResult("unknown", reason="call_module without module_type")

    if op == "call_function":
        text = _target_text(node)
        for operator_name, api in BUILTIN_OPERATOR_TO_TORCH.items():
            if f"function {operator_name}" in text or f"operator.{operator_name}" in text:
                return MappingResult("mapped", api=api, reason=f"builtin/operator {operator_name}")
        name = _find_function_name(text)
        if name:
            api_name = "div" if name == "truediv" else name
            return MappingResult("mapped", api=f"torch.{api_name}", reason=f"function target contains {name}")
        return MappingResult("unknown", reason=f"unmapped call_function target={node.get('target')}")

    if op == "call_method":
        method = str(node.get("target") or node.get("raw_target") or "")
        if method in TENSOR_METHODS:
            return MappingResult("mapped", api=f"torch.Tensor.{method}", reason=f"tensor method {method}")
        return MappingResult("unknown", reason=f"unmapped tensor method={method}")

    return MappingResult("unknown", reason=f"unmapped FX op={op}")


def _format_node(node: dict) -> str:
    return (
        f"name={node.get('name')} op={node.get('op')} target={node.get('target')} "
        f"canonical={node.get('canonical_target')} module_type={node.get('module_type')} "
        f"shape={node.get('shape')} tags={node.get('semantic_tags')}"
    )


def _print_fx_nodes(nodes: Sequence[dict], results: Sequence[MappingResult]) -> None:
    for node, result in zip(nodes, results):
        suffix = ""
        if result.status == "mapped":
            suffix = f" -> {result.api} ({result.reason})"
        elif result.status == "ignored":
            suffix = f" -> ignored ({result.reason})"
        else:
            suffix = f" -> unknown ({result.reason})"
        print(f"[fx-node] {_format_node(node)}{suffix}")


def _print_compare_ast(task_code: str, fx_apis: Sequence[str]) -> None:
    from akg_agents.database.api.pytorch_api_extract import pytorch_code_to_doc_list

    ast_apis = [doc.get("canonical", "") for doc in pytorch_code_to_doc_list(task_code, only_in_forward=True, class_name="Model")]
    ast_set = set(ast_apis)
    fx_set = set(fx_apis)
    print(f"[ast-api] {ast_apis}")
    print(f"[compare] fx_only={sorted(fx_set - ast_set)}")
    print(f"[compare] ast_only={sorted(ast_set - fx_set)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test FX graph node -> benchmark PyTorch API mapping.")
    parser.add_argument("--task-file", default=None, help="KernelBench file or prompt log containing a python code fence.")
    parser.add_argument("--strict", action="store_true", help="Return non-zero if any FX node is unknown.")
    parser.add_argument("--compare-ast", action="store_true", help="Compare FX-derived APIs with current AST extractor APIs.")
    args = parser.parse_args()

    sys.path.insert(0, str(PYTHON_ROOT))

    try:
        from akg_agents.core.extractor_torch import extract_kernelbench_shapes_dtypes
    except ModuleNotFoundError as exc:
        print(f"[fail] missing dependency while importing akg_agents: {exc}", file=sys.stderr)
        print(f"[hint] run: pip install -r {AKG_AGENTS_ROOT / 'requirements.txt'}", file=sys.stderr)
        return 2

    task_code = _load_task(args.task_file)
    print("[start] running FX extractor", flush=True)
    try:
        meta = extract_kernelbench_shapes_dtypes(task_code, device="cuda")
    except Exception as exc:
        print(f"[fail] FX extractor failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    graph_nodes = list(meta.get("graph_tensors") or [])
    print(f"[fx-summary] graph_tensor_nodes={len(graph_nodes)}", flush=True)
    if not graph_nodes:
        print("[fail] FX extractor produced no graph tensor nodes", file=sys.stderr)
        return 1

    results = [map_fx_node_to_api(node) for node in graph_nodes]
    _print_fx_nodes(graph_nodes, results)

    fx_apis = _dedup_keep_order(result.api for result in results if result.status == "mapped")
    print(f"[fx-api] {fx_apis}", flush=True)

    mapped = sum(1 for result in results if result.status == "mapped")
    ignored = sum(1 for result in results if result.status == "ignored")
    unknown_nodes = [
        (node, result)
        for node, result in zip(graph_nodes, results)
        if result.status == "unknown"
    ]
    print(f"[coverage] mapped={mapped} ignored={ignored} unknown={len(unknown_nodes)}", flush=True)
    for node, result in unknown_nodes:
        print(f"[coverage-unknown] {_format_node(node)} reason={result.reason}")

    if args.compare_ast:
        try:
            _print_compare_ast(task_code, fx_apis)
        except ModuleNotFoundError as exc:
            print(f"[warn] cannot compare AST extractor: missing dependency {exc}", file=sys.stderr)

    if args.strict and unknown_nodes:
        print("[fail] strict mode found unknown FX nodes", file=sys.stderr)
        return 1

    print("[pass] FX API mapping smoke passed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
