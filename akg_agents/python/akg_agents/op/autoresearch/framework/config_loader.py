"""
YAML 任务配置加载器 — 从 task.yaml 构造 TaskConfig

task.yaml 示例:
    name: matmul_swiglu
    description: "Fuse Matmul + SwiGLU"
    dsl: triton_cuda
    framework: torch
    backend: cuda
    arch: a100
    editable_files: [kernel.py]
    eval:
      timeout: 120
      repeats: 3
    metric:
      primary: latency_us
      lower_is_better: true
    agent:
      program_file: ../program.md
      ref_file: reference.py
      max_rounds: 20
"""

import os
from typing import Optional

import yaml

from .config import TaskConfig, AgentConfig
from akg_agents.op.utils.dsl_project_config import flatten_task_yaml_dsl_blocks


# ---------------------------------------------------------------------------
# Edit guardrails loader
# ---------------------------------------------------------------------------

_GUARDRAILS_FILE = os.path.join(os.path.dirname(__file__), "edit_guardrails.yaml")
_guardrails_cache: Optional[dict] = None


def _load_edit_guardrails() -> dict:
    """Load and cache ``edit_guardrails.yaml`` from the framework dir."""
    global _guardrails_cache
    if _guardrails_cache is not None:
        return _guardrails_cache
    if not os.path.exists(_GUARDRAILS_FILE):
        _guardrails_cache = {}
        return _guardrails_cache
    with open(_GUARDRAILS_FILE, "r", encoding="utf-8") as f:
        _guardrails_cache = yaml.safe_load(f) or {}
    return _guardrails_cache


def _merge_guardrail_patterns(
    base: dict,
    *overlays: dict,
) -> dict:
    """Merge multiple ``forbidden_patterns`` dicts (union of pattern lists).

    Each dict has keys like ``"content"``, ``"diff"``, ``"diff_any"``
    mapping to lists of regex strings. Lists are concatenated with
    deduplication (preserving order).
    """
    merged: dict = {}
    for d in (base, *overlays):
        if not isinstance(d, dict):
            continue
        for key, patterns in d.items():
            if not isinstance(patterns, list):
                continue
            existing = merged.setdefault(key, [])
            seen = set(existing)
            for p in patterns:
                if isinstance(p, str) and p not in seen:
                    existing.append(p)
                    seen.add(p)
    return merged


def build_forbidden_patterns(
    *,
    dsl: Optional[str] = None,
    backend: Optional[str] = None,
    framework: Optional[str] = None,
    hardware: Optional[str] = None,
    task_override: Optional[dict] = None,
) -> dict:
    """Build the ``forbidden_patterns`` dict for a TaskConfig.

    Merges patterns from ``edit_guardrails.yaml`` by scope (global →
    dsl → hardware → framework) then applies the task-level override
    from ``task.yaml`` on top.

    This is the single construction site for forbidden_patterns —
    ``load_yaml_config`` calls it once and passes the result to
    ``TaskConfig``.
    """
    g = _load_edit_guardrails()

    layers = [g.get("global", {})]

    dsl_section = g.get("dsl", {})
    if dsl and isinstance(dsl_section, dict) and dsl in dsl_section:
        layers.append(dsl_section[dsl])

    hw_section = g.get("hardware", {})
    if hardware and isinstance(hw_section, dict) and hardware in hw_section:
        layers.append(hw_section[hardware])
    # Also try backend as hardware key (common alias).
    if backend and backend != hardware and isinstance(hw_section, dict) and backend in hw_section:
        layers.append(hw_section[backend])

    fw_section = g.get("framework", {})
    if framework and isinstance(fw_section, dict) and framework in fw_section:
        layers.append(fw_section[framework])

    # Task-level override goes last (highest priority).
    if task_override and isinstance(task_override, dict):
        layers.append(task_override)

    return _merge_guardrail_patterns({}, *layers)


def load_yaml_config(task_dir: str) -> Optional[TaskConfig]:
    """从 task_dir/task.yaml 加载 TaskConfig. 文件不存在返回 None."""
    yaml_path = os.path.join(task_dir, "task.yaml")
    if not os.path.exists(yaml_path):
        return None

    with open(yaml_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"{yaml_path}: expected YAML dict, got {type(raw).__name__}")

    # ---- 必填字段 ----
    name = raw.get("name")
    description = raw.get("description", "")
    if not name:
        raise ValueError(f"{yaml_path}: 'name' is required")

    # ---- 适配器声明 ----
    dsl = raw.get("dsl")
    framework = raw.get("framework")
    backend = raw.get("backend")
    arch = raw.get("arch")

    # ---- 文件 ----
    editable_files = raw.get("editable_files", [])
    eval_script = raw.get("eval_script")  # 自定义 eval 脚本 (覆盖 adapter 生成)

    # ---- eval 块 ----
    eval_block = raw.get("eval", {})
    eval_timeout = eval_block.get("timeout", 600)
    import_timeout = eval_block.get("import_timeout", 15)

    # ---- smoke_test 块 ----
    smoke_block = raw.get("smoke_test", {})
    smoke_test_script = smoke_block.get("script")
    smoke_test_timeout = smoke_block.get("timeout", 10)

    # ---- metric 块 ----
    metric_block = raw.get("metric", {})
    primary_metric = metric_block.get("primary", "score")
    lower_is_better = metric_block.get("lower_is_better", True)
    improvement_threshold = metric_block.get("improvement_threshold", 0.0)

    # ---- constraints ----
    constraints = {}
    for metric_name, spec in raw.get("constraints", {}).items():
        if isinstance(spec, dict):
            constraints[metric_name] = (spec["op"], spec["value"])
        elif isinstance(spec, (list, tuple)) and len(spec) == 2:
            constraints[metric_name] = tuple(spec)

    # ---- guardrails 块 ----
    guardrails = raw.get("guardrails", {})
    max_patch_size = guardrails.get("max_patch_size", 15000)
    # Build forbidden_patterns by merging edit_guardrails.yaml defaults
    # (global + dsl + hardware + framework) with the task-level override.
    forbidden_patterns = build_forbidden_patterns(
        dsl=dsl,
        backend=backend,
        framework=framework,
        hardware=arch
        or raw.get("metadata", {}).get("hardware")
        or raw.get("metadata", {}).get("arch"),
        task_override=guardrails.get("forbidden_patterns"),
    )

    # ---- agent 块 ----
    agent_block = raw.get("agent", {})
    program_file = agent_block.get("program_file")
    ref_file = agent_block.get("ref_file")
    context_files = agent_block.get("context_files", [])
    max_rounds = agent_block.get("max_rounds", 30)

    # AgentConfig 覆盖
    agent_config_overrides = {}
    agent_config_block = agent_block.get("config", {})
    for key in AgentConfig.__dataclass_fields__:
        if key in agent_config_block:
            agent_config_overrides[key] = agent_config_block[key]
    agent_config = AgentConfig(**agent_config_overrides)

    # ---- git 块 ----
    git_block = raw.get("git", {})
    git_push = git_block.get("push", False)
    git_branch = git_block.get("branch")

    # ---- metadata ----
    metadata = raw.get("metadata", {})
    dsl_config = flatten_task_yaml_dsl_blocks(raw, yaml_path=yaml_path)

    return TaskConfig(
        name=name,
        description=description,
        dsl=dsl,
        framework=framework,
        backend=backend,
        arch=arch,
        dsl_config=dsl_config,
        eval_script=eval_script,
        editable_files=editable_files,
        eval_timeout=eval_timeout,
        primary_metric=primary_metric,
        lower_is_better=lower_is_better,
        improvement_threshold=improvement_threshold,
        constraints=constraints,
        smoke_test_script=smoke_test_script,
        smoke_test_timeout=smoke_test_timeout,
        import_timeout=import_timeout,
        max_patch_size=max_patch_size,
        forbidden_patterns=forbidden_patterns,
        program_file=program_file,
        ref_file=ref_file,
        context_files=context_files,
        git_push=git_push,
        git_branch=git_branch,
        max_rounds=max_rounds,
        agent=agent_config,
        metadata=metadata,
    )
