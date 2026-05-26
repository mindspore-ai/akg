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

"""task.yaml loader and the TaskConfig dataclass.

This module's job is parsing — turning the on-disk YAML into a typed
struct. It's the lowest layer in the task_config package: package_builder
and eval_client both depend on TaskConfig but TaskConfig depends on
nothing of ours.

The fields here are the schema. Adding a new task.yaml key means adding
a field on `TaskConfig` and reading it in `load_task_config`. Don't
reach into `raw` dicts elsewhere; route every consumer through the
typed dataclass.
"""
from dataclasses import dataclass, field
from typing import Optional

import os
import yaml


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class TaskConfig:
    """Minimal task configuration parsed from task.yaml."""
    name: str
    description: str = ""

    # Adapter declaration
    dsl: Optional[str] = None
    framework: Optional[str] = None
    backend: Optional[str] = None
    arch: Optional[str] = None

    # Files
    editable_files: list = field(default_factory=list)
    ref_file: str = "reference.py"

    # Eval params
    eval_timeout: int = 600
    # profiler iteration counts. Defaults match the previous hardcoded
    # values in _gen_eval_script (warmup=10, repeats=100). Configurable
    # via task.yaml eval.warmup_times / eval.run_times for ops where
    # the defaults are too noisy or too expensive.
    warmup_times: int = 10
    run_times: int = 100

    # Metric
    primary_metric: str = "score"
    lower_is_better: bool = True
    improvement_threshold: float = 0.0

    # Constraints: {metric_name: (operator_str, threshold)}
    constraints: dict = field(default_factory=dict)

    # Smoke test (optional — quick_check.py runs it before eval when configured)
    smoke_test_script: Optional[str] = None
    smoke_test_timeout: int = 10

    # CodeChecker (static analysis on editable files).
    # Default on; disable per-task via `code_checker.enabled: false` in
    # task.yaml or scaffold's --no-code-checker flag. When off, quick_check
    # and validate_kernel skip the AST/import/DSL pipeline but still reject
    # the scaffold TODO placeholder.
    code_checker_enabled: bool = True

    # Agent budget
    max_rounds: int = 30

    # Remote eval path: ship the package to one of these worker URLs.
    worker_urls: list = field(default_factory=list)

    # Local eval path: run verify/profile as a direct subprocess on this
    # device id (devices[0]). When worker_urls is non-empty it wins.
    devices: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------

def _pos_int(block: dict, key: str, default: int, yaml_path: str) -> int:
    """Read a positive-integer field from a task.yaml block. Profiler
    schedule + warmup loop both go nonsense at <=0 — fail loud at
    config-load time rather than wait for an aclnnArange crash mid-run.
    """
    raw = block.get(key, default)
    try:
        v = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{yaml_path}: eval.{key}={raw!r} is not an integer"
        ) from exc
    if v < 1:
        raise ValueError(f"{yaml_path}: eval.{key}={v} must be >= 1")
    return v


def load_task_config(task_dir: str) -> Optional[TaskConfig]:
    """Load TaskConfig from task_dir/task.yaml. Returns None if not found."""
    yaml_path = os.path.join(task_dir, "task.yaml")
    if not os.path.exists(yaml_path):
        return None

    with open(yaml_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"{yaml_path}: expected YAML dict, got {type(raw).__name__}")

    name = raw.get("name")
    if not name:
        raise ValueError(f"{yaml_path}: 'name' is required")

    eval_block = raw.get("eval", {})
    metric_block = raw.get("metric", {})
    smoke_block = raw.get("smoke_test", {})
    agent_block = raw.get("agent", {})
    code_checker_block = raw.get("code_checker", {})

    # Parse constraints
    constraints = {}
    for metric_name, spec in raw.get("constraints", {}).items():
        if isinstance(spec, dict):
            constraints[metric_name] = (spec["op"], spec["value"])
        elif isinstance(spec, (list, tuple)) and len(spec) == 2:
            constraints[metric_name] = tuple(spec)

    # Parse worker URLs from task.yaml
    worker_block = raw.get("worker", {})
    worker_urls = worker_block.get("urls", [])
    if isinstance(worker_urls, str):
        worker_urls = [u.strip() for u in worker_urls.split(",") if u.strip()]

    # Devices list. Accepts [5] / "5" / "0,1,2".
    devices_raw = raw.get("devices", [])
    if isinstance(devices_raw, int):
        devices = [devices_raw]
    elif isinstance(devices_raw, str):
        devices = [int(d.strip()) for d in devices_raw.split(",") if d.strip()]
    elif isinstance(devices_raw, list):
        devices = [int(d) for d in devices_raw]
    else:
        devices = []

    return TaskConfig(
        name=name,
        description=raw.get("description", ""),
        dsl=raw.get("dsl"),
        framework=raw.get("framework"),
        backend=raw.get("backend"),
        arch=raw.get("arch"),
        editable_files=raw.get("editable_files", []),
        ref_file=agent_block.get("ref_file") or "reference.py",
        eval_timeout=eval_block.get("timeout", 600),
        warmup_times=_pos_int(eval_block, "warmup_times", 10, yaml_path),
        run_times=_pos_int(eval_block, "run_times", 100, yaml_path),
        primary_metric=metric_block.get("primary", "score"),
        lower_is_better=metric_block.get("lower_is_better", True),
        improvement_threshold=metric_block.get("improvement_threshold", 0.0),
        constraints=constraints,
        smoke_test_script=smoke_block.get("script"),
        smoke_test_timeout=smoke_block.get("timeout", 10),
        code_checker_enabled=bool(code_checker_block.get("enabled", True)),
        max_rounds=agent_block.get("max_rounds", 30),
        worker_urls=worker_urls,
        devices=devices,
    )
