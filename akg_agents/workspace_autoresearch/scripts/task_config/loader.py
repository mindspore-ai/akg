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
struct. It's the lowest layer in the task_config package: eval_client
depends on TaskConfig but TaskConfig depends on nothing of ours.

The fields here are the schema. Adding a new task.yaml key means adding
a field on `TaskConfig` and reading it in `load_task_config`. Don't
reach into `raw` dicts elsewhere; route every consumer through the
typed dataclass.
"""
from dataclasses import dataclass, field
from typing import Optional

import os
import yaml


# File-name conventions (REF_FILE_DEFAULT / py_stem / editable-files
# helpers / resolve_kernel_paths_for_op) live in akg_agents.op.utils.
# task_layout — workflow-neutral SoT. Re-exported here for back-compat
# (legacy callers still import from task_config.loader).

from akg_agents.op.utils.dsl_project_config import flatten_task_yaml_dsl_blocks  # noqa: E402
from akg_agents.op.utils.task_layout import REF_FILE_DEFAULT, py_stem  # noqa: E402, F401


def _is_contained(path: str) -> bool:
    """Return True iff `path` is a relative path that doesn't escape its
    parent. False for absolute paths (any platform), drive-letter forms,
    paths containing `..` segments, or paths whose normalised form would
    resolve outside the empty-base join target.

    Used at task.yaml load time to refuse editable_files / data_files /
    ref_file entries that point outside the task_dir. Without this, a
    hand-edited (or hostile) task.yaml could list `../../secret.txt`
    and package_builder would read + tar the file before the remote
    worker's safe_extract had a chance to reject it on extraction —
    the bytes would already have left the client.
    """
    if not path:
        return False
    # Reject any drive letter (Windows) or POSIX absolute path. Also
    # reject leading `/` or `\` on Windows: os.path.isabs returns
    # False for `/foo/bar` on Windows because it lacks a drive letter,
    # so without this extra check a task.yaml shipped from a POSIX dev
    # box could pass containment on a Windows worker (or vice-versa).
    if (os.path.isabs(path)
            or (len(path) >= 2 and path[1] == ":")
            or path.startswith(("/", "\\"))):
        return False
    # Reject any `..` segment, on either separator.
    normalised = path.replace("\\", "/")
    parts = [p for p in normalised.split("/") if p and p != "."]
    if any(p == ".." for p in parts):
        return False
    return True


def _filter_contained(paths: list, field_name: str) -> list:
    """Drop entries that fail `_is_contained` and emit a stderr warning
    naming each rejection — silent drops would have the operator
    chasing 'why is data_files smaller than my task.yaml'."""
    import sys as _sys
    safe: list = []
    for p in paths:
        if _is_contained(str(p)):
            safe.append(p)
        else:
            print(f"[loader] WARNING: dropping {field_name} entry "
                  f"{p!r}: path escapes task_dir (absolute, drive "
                  f"letter, or contains `..`).", file=_sys.stderr)
    return safe


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class TaskConfig:
    """Minimal task configuration parsed from task.yaml.

    The workspace's target triple ``(backend, framework, dsl)`` is pinned
    per repo in ``config.yaml``'s ``defaults`` block (single-target-per-
    repo design); downstream code reads it via ``utils.settings.target_*``
    rather than carrying it on TaskConfig. ``arch`` (e.g. ``ascend910b3``
    / ``a100``) varies per machine and is auto-derived from the picked
    ``--devices`` via the backend-appropriate probe in ``utils.hw_detect``
    (npu-smi for ascend, nvidia-smi for cuda).
    """
    name: str
    description: str = ""

    arch: Optional[str] = None

    # Files
    editable_files: list = field(default_factory=list)
    ref_file: str = REF_FILE_DEFAULT

    # Sibling files the ref module reads at runtime (NPUKernelBench-style
    # `<op>.json` shape lists, sglang-style `ref.pt` output caches,
    # auxiliary `.py` imports, etc.). Listed by basename relative to
    # task_dir. The remote-eval package builder ships them alongside
    # task.yaml + ref + editable; local eval doesn't use the field.
    data_files: list = field(default_factory=list)

    # Eval params
    # Per-SHAPE budget for verify/profile in seconds. eval_client scales it
    # by num_cases (probed from the ref module) before invoking the eval
    # subprocess, so the wall-clock cap is eval_timeout * num_cases.
    # Single-shape refs (num_cases=1) keep the original semantics.
    eval_timeout: int = 600

    # Explicit case-count override (task.yaml `eval.num_cases`). When > 0,
    # eval_request uses it directly instead of importing the ref module to
    # probe get_inputs/get_input_groups — lets dev hosts without torch/CANN
    # scale the eval timeout and sticky fingerprint correctly. 0 = auto.
    num_cases: int = 0

    # Metric
    primary_metric: str = "score"
    lower_is_better: bool = True
    improvement_threshold: float = 0.0

    # Constraints: {metric_name: (operator_str, threshold)}
    constraints: dict = field(default_factory=dict)

    # Smoke test (optional — quick_check.py runs it before eval when configured)
    smoke_test_script: Optional[str] = None
    smoke_test_timeout: int = 10

    # Triton regression check (validate_triton_impl) on editable files.
    # Default on; disable per-task via `code_checker.enabled: false` in
    # task.yaml or scaffold's --no-code-checker flag. The yaml key name
    # is kept as `code_checker.enabled` for back-compat with existing
    # task.yaml files. When off, quick_check and validate_kernel skip
    # the regression check but still reject the scaffold TODO placeholder.
    code_checker_enabled: bool = True

    # Agent budget
    max_rounds: int = 30

    # Local devices
    devices: list = field(default_factory=list)
    """Device IDs for local eval (written by scaffold from --devices). When
    non-empty and no worker_urls, run_eval uses devices[0] as default
    device_id."""

    # Remote workers
    worker_urls: list = field(default_factory=list)
    """HTTP worker URLs (e.g. ["http://127.0.0.1:9111"]) for remote eval.
    When non-empty (or `--worker-url` passed on the CLI), run_eval ships
    the task package via HTTP POST to the first reachable worker. Local
    devices are the fallback."""

    # Per-DSL knobs (e.g. ``catlass.root`` / ``catlass.op_dir`` or
    # ``ascendc.op_dir``). Keys are flat (``catlass_root`` /
    # ``catlass_op_dir`` / ``ascendc_op_dir`` historically); akg_eval
    # forwards them verbatim into the eval ``config_dict`` + ``task_info``
    # so the adapter's ``prepare_config`` consumes them without TaskConfig
    # knowing any DSL.
    # Loader normalizes explicit DSL yaml blocks into this dict; adapter
    # defaults own absent blocks so unrelated DSLs do not receive stale
    # keys.
    dsl_config: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------

def load_task_config(task_dir: str) -> Optional[TaskConfig]:
    """Load TaskConfig from task_dir/task.yaml. Returns None if not found."""
    # Lazy import: callers reach loader via `from task_config import ...`,
    # which guarantees scripts/ is on sys.path by the time this runs.
    from utils.settings import (
        default_max_rounds, default_eval_timeout, default_smoke_test_timeout,
        default_code_checker_enabled, default_metric,
    )
    _metric = default_metric()
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

    # Parse devices list. Accepts [5] / "5" / "0,1,2".
    devices_raw = raw.get("devices", [])
    if isinstance(devices_raw, int):
        devices = [devices_raw]
    elif isinstance(devices_raw, str):
        devices = [int(d.strip()) for d in devices_raw.split(",") if d.strip()]
    elif isinstance(devices_raw, list):
        devices = [int(d) for d in devices_raw]
    else:
        devices = []

    # Parse worker_urls. Accepts "host:port" / "http://host:port" /
    # comma-separated string / list. Empty by default — devices is the
    # default transport unless --worker-url overrides.
    worker_urls_raw = raw.get("worker", {}).get("urls", [])
    if isinstance(worker_urls_raw, str):
        worker_urls = [u.strip() for u in worker_urls_raw.split(",") if u.strip()]
    elif isinstance(worker_urls_raw, list):
        worker_urls = [str(u).strip() for u in worker_urls_raw if str(u).strip()]
    else:
        worker_urls = []

    data_files_raw = raw.get("data_files", [])
    if isinstance(data_files_raw, str):
        data_files = [data_files_raw] if data_files_raw else []
    elif isinstance(data_files_raw, list):
        data_files = [str(f) for f in data_files_raw if f]
    else:
        data_files = []

    # Refuse path-escaping entries before they reach package_builder /
    # eval_runner. The package builder reads the file off disk
    # (task_dir + name); without containment a `../../secret` in
    # task.yaml would be slurped into the tar and shipped to the
    # remote worker even though the worker's safe_extract would reject
    # it on the other side — the bytes have already left.
    raw_editable = raw.get("editable_files") or []
    editable_files = _filter_contained(list(raw_editable), "editable_files")
    data_files = _filter_contained(data_files, "data_files")
    raw_ref = agent_block.get("ref_file") or REF_FILE_DEFAULT
    if not _is_contained(str(raw_ref)):
        import sys as _sys
        print(f"[loader] WARNING: ref_file {raw_ref!r} escapes task_dir; "
              f"falling back to {REF_FILE_DEFAULT}. Hand-edit task.yaml "
              f"if this isn't what you intended.", file=_sys.stderr)
        raw_ref = REF_FILE_DEFAULT

    # Per-DSL blocks are opt-in; adapter defaults handle the common layout.
    dsl_config = flatten_task_yaml_dsl_blocks(raw, yaml_path=yaml_path)

    config = TaskConfig(
        name=name,
        description=raw.get("description", ""),
        arch=raw.get("arch"),
        editable_files=editable_files,
        ref_file=raw_ref,
        data_files=data_files,
        eval_timeout=eval_block.get("timeout", default_eval_timeout()),
        num_cases=int(eval_block.get("num_cases", 0) or 0),
        primary_metric=metric_block.get("primary", _metric["primary"]),
        lower_is_better=metric_block.get(
            "lower_is_better", _metric["lower_is_better"]),
        improvement_threshold=metric_block.get(
            "improvement_threshold", _metric["improvement_threshold"]),
        constraints=constraints,
        smoke_test_script=smoke_block.get("script"),
        smoke_test_timeout=smoke_block.get(
            "timeout", default_smoke_test_timeout()),
        code_checker_enabled=bool(code_checker_block.get(
            "enabled", default_code_checker_enabled())),
        max_rounds=agent_block.get("max_rounds", default_max_rounds()),
        devices=devices,
        worker_urls=worker_urls,
        dsl_config=dsl_config,
    )
    # editable_files drives kernel-file resolution in eval (local + remote).
    # Reject an empty list (e.g. an 'editable_file' typo in task.yaml) at
    # load time — local eval would otherwise crash with an opaque
    # IndexError downstream.
    if not config.editable_files:
        raise ValueError(
            f"{yaml_path}: 'editable_files' must list at least one kernel "
            f"file (got {raw.get('editable_files')!r}) — check for a typo "
            f"such as 'editable_file'.")
    return config
