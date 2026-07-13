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

"""CANN-Bench task loading and prompt construction utilities.

Pattern follows sol_utils.py: only does path detection and prompt construction.
Input generation and case parsing are handled by CANN-Bench source repo
(DataGenerator, ParamBuilder) imported directly in templates via sys.path.
"""

from __future__ import annotations

import yaml
from pathlib import Path
from typing import Any, Dict, Tuple


def is_cann_task_dir(dir_path: str | Path) -> bool:
    """Check if a directory is a CANN-Bench task directory."""
    p = Path(dir_path).expanduser().resolve()
    return p.is_dir() and (p / "proto.yaml").is_file() and (p / "golden.py").is_file()


def load_cann_task_source(task_path: str | Path) -> Tuple[str, str, str]:
    """Load a CANN-Bench task directory.

    Pattern follows sol_utils.load_sol_task_source.
    Returns (op_name, task_desc, problem_dir_abs).
    """
    resolved_path = Path(task_path).expanduser().resolve()
    if not is_cann_task_dir(resolved_path):
        raise FileNotFoundError(f"CANN task directory not found or invalid: {resolved_path}")
    return load_cann_task_for_runner(resolved_path)


def inject_cann_into_config(
    config: Dict[str, Any],
    problem_dir: str | Path,
) -> None:
    """Inject CANN-Bench configuration into a config dict.

    Injects:
    - cann_problem_dir: task directory path
    - bench_type: "cann"
    """
    config["cann_problem_dir"] = str(Path(problem_dir).expanduser().resolve())
    config["bench_type"] = "cann"


def get_cann_task_desc_for_prompt(problem_dir: str | Path) -> str:
    """Generate structured task description for prompt injection.

    Reads raw files for LLM prompt text only — does not parse cases or inputs.
    Pattern follows sol_utils.py load_sol_task.
    """
    problem_dir = Path(problem_dir).expanduser().resolve()

    proto = load_cann_proto(problem_dir)
    golden_code = load_cann_golden(problem_dir)
    desc = load_cann_desc(problem_dir)

    parts = []
    if desc:
        parts.append(f"## 算子描述\n{desc}")
    parts.append(
        f"## 算子定义 (proto.yaml)\n```yaml\n"
        f"{yaml.dump(proto, allow_unicode=True, default_flow_style=False)}\n```"
    )
    parts.append(f"## Golden 参考实现\n```python\n{golden_code}\n```")

    op = proto.get("operator", {})
    precision_thresholds = op.get("precision_thresholds")
    if precision_thresholds:
        parts.append(f"## 精度阈值覆盖\n{precision_thresholds}")

    return "\n\n".join(parts)


def load_cann_task_for_runner(problem_dir: str | Path) -> Tuple[str, str, str]:
    """Load a CANN-Bench task and build task description for runners.

    Returns (op_name, task_desc, problem_dir_abs).
    """
    problem_dir = Path(problem_dir).expanduser().resolve()
    proto = load_cann_proto(problem_dir)
    op_name = proto.get("operator", {}).get("name", problem_dir.name)
    task_desc = get_cann_task_desc_for_prompt(problem_dir)
    return op_name, task_desc, str(problem_dir)


def load_cann_proto(problem_dir: str | Path) -> Dict[str, Any]:
    """Parse proto.yaml using yaml.safe_load."""
    p = Path(problem_dir).expanduser().resolve() / "proto.yaml"
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_cann_golden(problem_dir: str | Path) -> str:
    """Read golden.py source code."""
    p = Path(problem_dir).expanduser().resolve() / "golden.py"
    return p.read_text(encoding="utf-8")


def load_cann_desc(problem_dir: str | Path) -> str:
    """Read desc.md content (may not exist)."""
    p = Path(problem_dir).expanduser().resolve() / "desc.md"
    if p.is_file():
        return p.read_text(encoding="utf-8")
    return ""