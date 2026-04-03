"""
YAML 任务配置加载器 — 从 task.yaml 构造 TaskConfig

task.yaml 示例:
    name: matmul_swiglu
    description: "Fuse Matmul + SwiGLU"
    dsl: triton_cuda
    framework: torch
    backend: cuda
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
    forbidden_patterns = guardrails.get("forbidden_patterns", {
        "content": [],
        "diff": ["^\\s*#"],
    })

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

    return TaskConfig(
        name=name,
        description=description,
        dsl=dsl,
        framework=framework,
        backend=backend,
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
