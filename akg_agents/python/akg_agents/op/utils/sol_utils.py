# Copyright 2025-2026 Huawei Technologies Co., Ltd
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

"""
Helpers for loading SOL tasks in single-runner entrypoints.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple


def resolve_path_from_base(path_value: str, base_dir: str | Path) -> str:
    """Resolve a config-relative path to an absolute path."""
    resolved_path = Path(path_value).expanduser()
    if not resolved_path.is_absolute():
        resolved_path = Path(base_dir) / resolved_path
    return str(resolved_path.resolve())


def is_sol_problem_dir(path_value: str | Path) -> bool:
    """Return whether the given path points to a SOL problem directory."""
    case_dir = Path(path_value).expanduser().resolve()
    return case_dir.is_dir() and (case_dir / "definition.json").exists()


def load_sol_task(sol_dir: str | Path) -> Tuple[str, str, str]:
    """Load a SOL dataset and build the task description for runners."""
    case_dir = Path(sol_dir).expanduser().resolve()
    if not (case_dir / "definition.json").exists():
        raise FileNotFoundError(f"SOL dataset missing definition.json: {case_dir}")

    definition_json = (case_dir / "definition.json").read_text(encoding="utf-8")
    definition = json.loads(definition_json)
    op_name = definition.get("name", case_dir.name)

    reference_code = (case_dir / "reference.py").read_text(encoding="utf-8")

    workload_sample = ""
    workload_file = case_dir / "workload.jsonl"
    if workload_file.exists():
        lines = [line.strip() for line in workload_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        if lines:
            first_workload = json.loads(lines[0])
            workload_sample = (
                f"\n\n## workload example ({len(lines)} entries, showing the first one)\n"
                f"```json\n{json.dumps(first_workload, indent=2, ensure_ascii=False)}\n```"
            )

    task_desc = (
        "Please implement a Triton Ascend operator.\n\n"
        f"## definition.json\n```json\n{definition_json}\n```\n\n"
        f"## reference.py\n```python\n{reference_code}\n```"
        f"{workload_sample}\n\n"
        "Note: write the kernel in Triton and wrap it in the forward method of ModelNew."
    )

    return op_name, task_desc, str(case_dir)


def load_task_source(task_path: str | Path) -> Tuple[Optional[str], str, Optional[str]]:
    """Load either a plain task file or a SOL problem directory."""
    resolved_path = Path(task_path).expanduser().resolve()
    if is_sol_problem_dir(resolved_path):
        op_name, task_desc, sol_problem_dir = load_sol_task(resolved_path)
        return op_name, task_desc, sol_problem_dir

    if resolved_path.is_file():
        return None, resolved_path.read_text(encoding="utf-8"), None

    raise FileNotFoundError(f"Task description file or SOL problem dir not found: {resolved_path}")
