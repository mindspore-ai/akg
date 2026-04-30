from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


_NPUKB_INPUTS_FACTORY = "get_input_groups"


def is_npukb_task_file(path_value: str | Path) -> bool:
    py_path = Path(path_value).expanduser()
    if not py_path.is_file() or py_path.suffix.lower() != ".py":
        return False
    json_path = py_path.with_suffix(".json")
    return json_path.is_file()


def load_npukb_task(py_path: str | Path) -> Tuple[str, str, Dict[str, str], Dict[str, Any]]:
    py_path = Path(py_path).expanduser().resolve()
    if not py_path.is_file() or py_path.suffix.lower() != ".py":
        raise FileNotFoundError(f"NPUKernelBench .py 文件不存在: {py_path}")

    json_path = py_path.with_suffix(".json")
    if not json_path.is_file():
        raise FileNotFoundError(
            f"NPUKernelBench 任务缺少同名 .json 输入清单: {json_path}",
        )

    py_code = py_path.read_text(encoding="utf-8")
    jsonl_content = json_path.read_text(encoding="utf-8")

    # 完整性校验：保证 JSONL 每行都是合法 JSON
    for idx, line in enumerate(jsonl_content.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Invalid JSONL at line {idx} of {json_path}: {e}",
            ) from e

    op_name = py_path.stem
    aux_files: Dict[str, str] = {f"{op_name}.json": jsonl_content}
    factory_names: Dict[str, Any] = {
        "inputs_factory": _NPUKB_INPUTS_FACTORY,
        "is_dynamic_shape": True,
    }
    return op_name, py_code, aux_files, factory_names


def inject_npukb_into_config(
    config: Dict[str, Any],
    aux_files: Dict[str, str],
    factory_names: Dict[str, Any],
) -> None:
    config["framework_aux_files"] = aux_files
    config["framework_factory_names"] = factory_names


def load_npukb_metadata_if_any(
    task_path: str | Path,
) -> Optional[Tuple[Dict[str, str], Dict[str, Any]]]:
    if not is_npukb_task_file(task_path):
        return None
    _op_name, _task_desc, aux_files, factory_names = load_npukb_task(task_path)
    return aux_files, factory_names
