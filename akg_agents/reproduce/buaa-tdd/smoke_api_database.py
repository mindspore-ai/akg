#!/usr/bin/env python3
"""Smoke test for the Triton API database prompt path.

This script does not call an LLM. It checks the three pieces that must work
before Coder can see database docs:

1. PyTorch API extraction from a KernelBench-style Model.
2. Qdrant retrieval into task_info["triton_api_recall_by_source"].
3. Prompt block rendering after optional Triton runtime filtering.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict


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
        x = self.max_pool(x)
        return x
'''


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise RuntimeError("missing dependency: PyYAML") from exc
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _load_task(path: Path | None) -> str:
    if path is None:
        return SAMPLE_TASK
    text = path.read_text(encoding="utf-8")
    match = re.search(r"```python\s*(.*?)```", text, re.S)
    return match.group(1) if match else text


def _resolve_cache_folder(db_cfg: Dict[str, Any]) -> None:
    cache_folder = db_cfg.get("embed_cache_folder")
    if not cache_folder:
        return
    cache_path = Path(str(cache_folder)).expanduser()
    if not cache_path.is_absolute():
        db_cfg["embed_cache_folder"] = str(AKG_AGENTS_ROOT / cache_path)


def _print_source_summary(by_source: Dict[str, Any]) -> None:
    for source, entries in sorted(by_source.items()):
        qualnames = [entry.get("triton_qualname", "") for entry in entries]
        print(f"[retrieve] {source}: {qualnames}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test Triton API database retrieval and prompt rendering.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--task-file", type=Path, default=None, help="Python file or prompt log containing a python code fence.")
    parser.add_argument("--qdrant-host", default=None)
    parser.add_argument("--qdrant-port", type=int, default=None)
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--online", action="store_true", help="Allow HuggingFace network checks/downloads.")
    parser.add_argument("--require-runtime-render", action="store_true")
    args = parser.parse_args()

    if not args.online:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    sys.path.insert(0, str(PYTHON_ROOT))

    try:
        from akg_agents.database.api.pytorch_api_extract import pytorch_code_to_doc_list
        from akg_agents.database.api_helper import (
            is_api_database_available,
            render_triton_recall,
            retrieve_and_store_triton_apis,
        )
    except ModuleNotFoundError as exc:
        print(f"[fail] missing dependency while importing akg_agents: {exc}", file=sys.stderr)
        print(f"[hint] run: pip install -r {AKG_AGENTS_ROOT / 'requirements.txt'}", file=sys.stderr)
        return 2

    config = _load_yaml(args.config)
    db_cfg = dict((config or {}).get("api_database") or {})
    db_cfg["enabled"] = True
    if args.qdrant_host is not None:
        db_cfg["qdrant_host"] = args.qdrant_host
    if args.qdrant_port is not None:
        db_cfg["qdrant_port"] = args.qdrant_port
    if args.force_rebuild:
        db_cfg["force_rebuild"] = True
    _resolve_cache_folder(db_cfg)
    config["api_database"] = db_cfg

    task_code = _load_task(args.task_file)
    docs = pytorch_code_to_doc_list(task_code, only_in_forward=True, class_name="Model")
    canonicals = [doc.get("canonical", "") for doc in docs]
    print(f"[extract] {canonicals}")
    expected = {"torch.nn.Conv2d", "torch.tanh", "torch.nn.MaxPool2d"}
    missing = sorted(expected - set(canonicals))
    if missing:
        print(f"[fail] PyTorch API extraction missed: {missing}", file=sys.stderr)
        return 1

    available, reason = is_api_database_available(config)
    if not available:
        print(f"[fail] API database unavailable: {reason}", file=sys.stderr)
        print("[hint] check docker container lt_qdrant and run env_build.sh/database init first", file=sys.stderr)
        return 1
    print("[qdrant] available")

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
    if not by_source:
        print("[fail] retrieval returned no source groups", file=sys.stderr)
        return 1
    _print_source_summary(by_source)

    raw_block = render_triton_recall(state, verify_runtime=False)
    runtime_block = render_triton_recall(state, verify_runtime=True)
    print(f"[render] raw_len={len(raw_block)} runtime_filtered_len={len(runtime_block)}")
    if not raw_block.strip():
        print("[fail] recall exists but raw prompt block is empty", file=sys.stderr)
        return 1
    if not runtime_block.strip():
        print("[warn] runtime-filtered prompt block is empty; recalled APIs are not available in this Triton runtime")
        if args.require_runtime_render:
            return 1
    else:
        preview = "\n".join(runtime_block.splitlines()[:20])
        print("[render-preview]")
        print(preview)

    print("[pass] Triton API database smoke passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
