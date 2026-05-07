# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SOL-ExecBench input normalization helpers.

The verifier executes SOL cases from three concrete files:
``definition.json``, ``workload.jsonl`` and ``reference.py``.  In practice
callers may provide that directory directly, an OpTaskBuilder JSON payload
containing those files, or the raw HuggingFace row format where
``reference`` and ``workloads`` are fields on a single record.  This module
normalizes all supported forms into the three-file directory contract.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional


SOL_REQUIRED_FILES = ("definition.json", "workload.jsonl", "reference.py")


def ensure_sol_problem_dir(
    *,
    config: Mapping[str, Any],
    work_dir: str,
    op_name: str,
    task_desc: str = "",
) -> str:
    """Return a directory containing the three required SOL files.

    Supported inputs, in priority order:
    - ``config["sol_problem_dir"]`` pointing at an already materialized case.
    - ``config["sol_problem_dir"]`` pointing at a JSON file/payload source.
    - ``config["sol_problem_json"]``, ``config["sol_problem_data"]`` or
      ``config["sol_task_code"]`` containing either the three-file JSON
      payload or a raw SOL dataset record.
    - ``task_desc`` using either JSON or markdown sections.
    """
    candidate = str(config.get("sol_problem_dir") or "").strip()
    if candidate:
        path = Path(candidate).expanduser()
        if _has_required_files(path):
            return str(path)
        if path.is_file():
            payload = path.read_text(encoding="utf-8")
            return _materialize_payload(payload, work_dir, op_name)

    for key in ("sol_problem_json", "sol_problem_data", "sol_task_code"):
        payload = config.get(key)
        if payload:
            return _materialize_payload(payload, work_dir, op_name)

    if task_desc and task_desc.strip():
        try:
            return _materialize_payload(task_desc, work_dir, op_name)
        except ValueError:
            pass

    detail = (
        "SOL input must be either a directory containing "
        "definition.json/workload.jsonl/reference.py, a JSON payload with "
        "those file names, or a raw SOL-ExecBench record with reference and "
        "workloads fields."
    )
    if candidate:
        raise FileNotFoundError(
            f"Invalid sol_problem_dir: {candidate}. {detail}"
        )
    raise ValueError(f"sol_problem_dir is missing. {detail}")


def _has_required_files(path: Path) -> bool:
    return path.is_dir() and all((path / name).is_file() for name in SOL_REQUIRED_FILES)


def _materialize_payload(payload: Any, work_dir: str, op_name: str) -> str:
    files = _payload_to_files(payload)
    digest = hashlib.sha256(
        json.dumps(files, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    out_dir = Path(work_dir).expanduser() / f"_sol_problem_{_safe_name(op_name)}_{digest}"
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "definition.json").write_text(
        _normalize_json_text(files["definition.json"]),
        encoding="utf-8",
    )
    (out_dir / "workload.jsonl").write_text(
        _normalize_workload_jsonl(files["workload.jsonl"]),
        encoding="utf-8",
    )
    (out_dir / "reference.py").write_text(
        str(files["reference.py"]).rstrip() + "\n",
        encoding="utf-8",
    )
    return str(out_dir)


def _payload_to_files(payload: Any) -> Dict[str, Any]:
    if isinstance(payload, Mapping):
        return _mapping_to_files(dict(payload))

    if isinstance(payload, (bytes, bytearray)):
        payload = payload.decode("utf-8")

    if not isinstance(payload, str):
        raise ValueError(f"Unsupported SOL payload type: {type(payload).__name__}")

    text = payload.strip()
    if not text:
        raise ValueError("SOL payload is empty")

    section_files = _extract_markdown_sections(text)
    if section_files:
        return section_files

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"SOL payload is not valid JSON: {exc}") from exc
    return _mapping_to_files(parsed)


def _mapping_to_files(data: Mapping[str, Any]) -> Dict[str, Any]:
    if all(name in data for name in SOL_REQUIRED_FILES):
        return {
            "definition.json": data["definition.json"],
            "workload.jsonl": data["workload.jsonl"],
            "reference.py": data["reference.py"],
        }

    # Raw HuggingFace row / SOL dataset record.
    if "reference" in data and ("workloads" in data or "workload" in data):
        definition = {
            key: value
            for key, value in data.items()
            if key not in {
                "workloads",
                "workload",
                "workload.jsonl",
                "reference.py",
            }
        }
        reference = data.get("reference.py") or data.get("reference")
        workloads = data.get("workloads", data.get("workload"))
        return {
            "definition.json": definition,
            "workload.jsonl": workloads,
            "reference.py": reference,
        }

    # A slightly nested variant is useful for API callers.
    if "definition" in data and ("workloads" in data or "workload" in data) and "reference" in data:
        definition = data["definition"]
        if isinstance(definition, Mapping) and "reference" not in definition:
            definition = dict(definition)
            definition["reference"] = data["reference"]
        return {
            "definition.json": definition,
            "workload.jsonl": data.get("workloads", data.get("workload")),
            "reference.py": data["reference"],
        }

    raise ValueError(
        "Unsupported SOL payload. Expected three-file JSON payload or raw "
        "record with reference and workloads fields."
    )


def _extract_markdown_sections(text: str) -> Dict[str, str]:
    files: Dict[str, str] = {}
    for file_name in SOL_REQUIRED_FILES:
        pattern = (
            rf"(?:^|\n)#+\s*{re.escape(file_name)}\s*\n"
            rf"```(?:json|python|text)?\s*\n(.*?)```"
        )
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            files[file_name] = match.group(1).strip()
    return files if all(name in files for name in SOL_REQUIRED_FILES) else {}


def _normalize_json_text(value: Any) -> str:
    if isinstance(value, str):
        parsed = json.loads(value)
    else:
        parsed = value
    return json.dumps(parsed, ensure_ascii=False, indent=2) + "\n"


def _normalize_workload_jsonl(value: Any) -> str:
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return ""
        # Already JSONL.
        if "\n" in stripped:
            return "\n".join(line.strip() for line in stripped.splitlines() if line.strip()) + "\n"
        parsed = json.loads(stripped)
    else:
        parsed = value

    if isinstance(parsed, Mapping):
        workloads: Iterable[Any] = [parsed]
    elif isinstance(parsed, list):
        workloads = parsed
    else:
        raise ValueError(
            f"Unsupported SOL workloads type: {type(parsed).__name__}"
        )

    return "".join(
        json.dumps(workload, ensure_ascii=False, separators=(",", ":")) + "\n"
        for workload in workloads
    )


def _safe_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name or "sol")
    return safe.strip("._") or "sol"

