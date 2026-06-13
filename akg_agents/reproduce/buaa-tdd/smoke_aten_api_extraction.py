#!/usr/bin/env python3
"""Smoke test for KernelBench ATen dispatcher API extraction.

This script does not call an LLM. By default it only records local ATen ops
triggered by a fake/meta forward. With --with-database it also verifies that
those ATen docs can drive Triton API database recall and prompt rendering.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set


SCRIPT_DIR = Path(__file__).resolve().parent
AKG_AGENTS_ROOT = SCRIPT_DIR.parent.parent
PYTHON_ROOT = AKG_AGENTS_ROOT / "python"
DEFAULT_CONFIG = PYTHON_ROOT / "akg_agents" / "op" / "config" / "default_torch_config.yaml"

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

def get_inputs():
    return [torch.rand(2, 8, 16, 16)]

def get_init_inputs():
    return []
'''

EXPECTED_SAMPLE_ATEN = {
    "torch.ops.aten.convolution.default",
    "torch.ops.aten.tanh.default",
    "torch.ops.aten.mul.Tensor",
    "torch.ops.aten.add.Tensor",
    "torch.ops.aten.max_pool2d_with_indices.default",
}


def _resolve_task_file(path_arg: Optional[str]) -> Optional[Path]:
    if not path_arg:
        return None
    raw = Path(path_arg).expanduser()
    if raw.exists():
        return raw
    base = raw if raw.is_absolute() else AKG_AGENTS_ROOT / raw
    if base.exists():
        return base
    matches = sorted(glob.glob(str(base))) or sorted(glob.glob(path_arg))
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


def _load_yaml(path: Path) -> Dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _resolve_cache_folder(db_cfg: Dict[str, Any]) -> None:
    cache_folder = db_cfg.get("embed_cache_folder")
    if not cache_folder:
        return
    cache_path = Path(str(cache_folder)).expanduser()
    if not cache_path.is_absolute():
        db_cfg["embed_cache_folder"] = str(AKG_AGENTS_ROOT / cache_path)


def _dedup(values: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _print_aten_docs(docs: List[Dict[str, Any]]) -> None:
    for doc in docs:
        print(
            "[aten-op] "
            f"canonical={doc.get('canonical')} "
            f"raw={doc.get('raw')} "
            f"schema={doc.get('redsig')}",
            flush=True,
        )


def _print_ast_compare(task_code: str) -> None:
    from akg_agents.database.api.pytorch_api_extract import pytorch_code_to_doc_list

    docs = pytorch_code_to_doc_list(task_code, only_in_forward=True, class_name="Model")
    print(f"[ast-api] {[doc.get('canonical', '') for doc in docs]}", flush=True)


def _print_fx_compare(task_code: str) -> None:
    from akg_agents.core.extractor_torch import extract_kernelbench_shapes_dtypes
    from smoke_fx_api_mapping import map_fx_node_to_api

    meta = extract_kernelbench_shapes_dtypes(task_code, device="cuda")
    graph_nodes = list(meta.get("graph_tensors") or [])
    fx_apis = _dedup(
        result.api
        for result in (map_fx_node_to_api(node) for node in graph_nodes)
        if result.status == "mapped"
    )
    print(f"[fx-api] {fx_apis}", flush=True)


def _assert_pure_recall_block(recall_block: str) -> int:
    from akg_agents.database.api_helper import verify_triton_api_runtime

    forbidden = {
        "torch base API section": "## 基础 API 文档",
        "torch API entry": "API name: torch.",
        "torch.tanh base doc": "torch.tanh",
        "invalid tl.tanh": "`tl.tanh`",
        "invalid tl.math.tanh": "`tl.math.tanh`",
        "generic libdevice": "tl.extra.libdevice",
        "AMD backend": ".amd.",
        "HIP backend": ".hip.",
        "random helper": "tl.random",
        "internal inline asm": "inline_asm",
        "internal builtin helper": "is_builtin",
    }
    failures = [name for name, snippet in forbidden.items() if snippet in recall_block]
    qualnames = sorted(set(re.findall(r"Triton API:\s*`((?:tl|triton)\.[A-Za-z_][\w.]*)`", recall_block)))
    invalid = [qualname for qualname in qualnames if not verify_triton_api_runtime(qualname)]
    if failures or invalid:
        if failures:
            print(f"[fail] impure recall snippets: {failures}", file=sys.stderr)
        if invalid:
            print(f"[fail] runtime-invalid recall APIs: {invalid}", file=sys.stderr)
        return 1
    print(f"[purity] recall block passed ({len(qualnames)} runtime-valid Triton APIs)", flush=True)
    return 0


def _run_database_smoke(args: argparse.Namespace, task_code: str) -> int:
    from akg_agents.database.api_helper import (
        is_api_database_available,
        render_triton_recall,
        retrieve_and_store_triton_apis,
    )

    config = _load_yaml(args.config)
    db_cfg = dict((config or {}).get("api_database") or {})
    db_cfg["enabled"] = True
    if args.qdrant_host is not None:
        db_cfg["qdrant_host"] = args.qdrant_host
    if args.qdrant_port is not None:
        db_cfg["qdrant_port"] = args.qdrant_port
    if args.force_rebuild:
        db_cfg["force_rebuild"] = True
    if args.force_rebuild_triton:
        db_cfg["force_rebuild_triton"] = True
    if args.force_rebuild_torch:
        db_cfg["force_rebuild_torch"] = True
    _resolve_cache_folder(db_cfg)
    config["api_database"] = db_cfg

    available, reason = is_api_database_available(config)
    if not available:
        print(f"[fail] API database unavailable: {reason}", file=sys.stderr)
        return 1

    state: Dict[str, Any] = {}
    retrieve_and_store_triton_apis(
        task_info=state,
        task_desc=task_code,
        qdrant_host=db_cfg.get("qdrant_host", "localhost"),
        qdrant_port=int(db_cfg.get("qdrant_port", 6333)),
        triton_collection=db_cfg.get("triton_collection", "triton_api"),
        torch_collection=db_cfg.get("torch_collection", "torch_api"),
        embed_model=db_cfg.get("embed_model", "sentence-transformers/all-MiniLM-L6-v2"),
        embed_cache_folder=db_cfg.get("embed_cache_folder"),
        force_rebuild=bool(db_cfg.get("force_rebuild", False)),
        force_rebuild_triton=bool(db_cfg.get("force_rebuild_triton", False)),
        force_rebuild_torch=bool(db_cfg.get("force_rebuild_torch", False)),
        topk_per_query=int(db_cfg.get("topk_per_query", 32)),
        filter_tags=db_cfg.get("filter_tags", ["tl"]),
        target_backend=db_cfg.get("target_backend", "cuda"),
        enable_keyword_recall=bool(db_cfg.get("enable_keyword_recall", True)),
        min_keep=int(db_cfg.get("min_keep", 2)),
        max_keep=int(db_cfg.get("max_keep", 10)),
        elbow_min_gap_ratio=float(db_cfg.get("elbow_min_gap_ratio", 0.15)),
        relative_decay=float(db_cfg.get("relative_decay", 0.90)),
        keyword_fallback_qdrant=bool(db_cfg.get("keyword_fallback_qdrant", True)),
        keyword_fallback_limit_per_kw=int(db_cfg.get("keyword_fallback_limit_per_kw", 4)),
    )
    by_source = state.get("triton_api_recall_by_source") or {}
    for source, entries in sorted(by_source.items()):
        print(f"[retrieve] {source}: {[entry.get('triton_qualname', '') for entry in entries]}", flush=True)
    recall_block = render_triton_recall(state, verify_runtime=True)
    print(
        f"[database] source_kind={state.get('api_database_source_kind')} "
        f"source_apis={state.get('api_database_source_apis')} "
        f"runtime_filtered_len={len(recall_block)}",
        flush=True,
    )
    if not recall_block.strip():
        print("[fail] database recall rendered an empty runtime-filtered block", file=sys.stderr)
        return 1
    if args.assert_pure_recall:
        purity_code = _assert_pure_recall_block(recall_block)
        if purity_code != 0:
            return purity_code
    if args.print_recall_preview:
        print("[recall-preview]", flush=True)
        print("\n".join(recall_block.splitlines()[: args.print_recall_preview]), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test ATen dispatcher API extraction for KernelBench.")
    parser.add_argument("--task-file", default=None, help="KernelBench file or prompt log containing a python code fence.")
    parser.add_argument("--compare-ast", action="store_true")
    parser.add_argument("--compare-fx", action="store_true")
    parser.add_argument("--strict", action="store_true", help="For the default sample, require all expected ATen ops.")
    parser.add_argument("--with-database", action="store_true", help="Also run Triton API database recall/render smoke.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--qdrant-host", default=None)
    parser.add_argument("--qdrant-port", type=int, default=None)
    parser.add_argument("--force-rebuild", action="store_true", help="Rebuild both Triton and Torch collections during --with-database.")
    parser.add_argument("--force-rebuild-triton", action="store_true", help="Rebuild the Triton API collection during --with-database.")
    parser.add_argument("--force-rebuild-torch", action="store_true", help="Rebuild the Torch API collection during --with-database.")
    parser.add_argument("--assert-pure-recall", action="store_true", help="Fail if rendered database recall contains invalid/noisy APIs.")
    parser.add_argument("--print-recall-preview", type=int, default=0, help="Print the first N lines of rendered database recall.")
    parser.add_argument("--online", action="store_true", help="Allow HuggingFace network checks/downloads during --with-database.")
    args = parser.parse_args()

    if not args.online:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    sys.path.insert(0, str(PYTHON_ROOT))
    sys.path.insert(0, str(SCRIPT_DIR))

    from akg_agents.database.api.pytorch_api_extract import aten_dispatch_code_to_doc_list

    task_code = _load_task(args.task_file)
    try:
        docs = aten_dispatch_code_to_doc_list(task_code, class_name="Model")
    except Exception as exc:
        print(f"[fail] ATen dispatch extraction failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    _print_aten_docs(docs)
    canonicals = [doc.get("canonical", "") for doc in docs]
    print(f"[aten-api] {canonicals}", flush=True)
    if not docs:
        print("[fail] no ATen operators captured", file=sys.stderr)
        return 1

    if args.strict and args.task_file is None:
        missing = sorted(EXPECTED_SAMPLE_ATEN - set(canonicals))
        if missing:
            print(f"[fail] missing expected sample ATen ops: {missing}", file=sys.stderr)
            return 1

    if args.compare_ast:
        _print_ast_compare(task_code)
    if args.compare_fx:
        _print_fx_compare(task_code)
    if args.with_database:
        db_code = _run_database_smoke(args, task_code)
        if db_code != 0:
            return db_code

    print("[pass] ATen API extraction smoke passed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
