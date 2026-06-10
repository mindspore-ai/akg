"""
Task Scaffolder — Generates task directories for AKG autoresearch workflow.

Creates a self-contained task_dir with:
  - editable files (kernel.py etc.)
  - reference.py (task description)
  - task.yaml (config for AgentLoop)
  - program.md (agent instructions)
  - context_files (into system prompt)
  - extra_files (docs/ for read_file on-demand access)
  - .git/ (baseline commit)

The generated task_dir is AKG-workflow-only: no eval_script.
Adapter fields (dsl/framework/backend/arch) are written to task.yaml
for skill matching and hardware guardrails. Evaluation is handled by
the injected eval_fn -> KernelVerifier.
"""

import os
import subprocess
import time
import uuid

import yaml

from akg_agents.core.worker.interface import DEFAULT_EVAL_TIMEOUT_S
from akg_agents.op.utils.task_layout import REF_FILE_DEFAULT


def scaffold_task_dir(
    base_dir: str,
    op_name: str,
    task_desc: str,
    editable_files: dict[str, str],
    program_md: str = "",
    context_files: dict[str, str] | None = None,
    extra_files: dict[str, str] | None = None,
    max_rounds: int = 20,
    eval_timeout: int = DEFAULT_EVAL_TIMEOUT_S,
    dsl: str = "",
    framework: str = "",
    backend: str = "",
    arch: str = "",
    kernel_project_src: str | None = None,
    dsl_config: dict | None = None,
) -> str:
    """Create a fresh task_dir and git init. Returns task_dir path.

    Each call generates a unique directory (base_dir/{op_name}_{timestamp}_{uuid}/).
    ExperimentRunner restores _best_result from same-directory history (runner.py:401);
    reusing an old directory would inherit stale best_metrics.

    Args:
        base_dir: Parent directory for the task.
        op_name: Operator name (used in directory name and task.yaml).
        task_desc: Framework code (Model/get_inputs) -> reference.py.
        editable_files: {filename: content} written to task_dir. For
            multi-file DSLs, the adapter's project files are added to
            task.yaml editable_files after the project tree is materialized.
        program_md: Agent instructions -> program.md (enters system prompt).
        context_files: {filename: content} — key is relative path within task_dir
            (may contain subdirectories like "api/api.md"). Listed in task.yaml
            context_files (loaded into system prompt). Parent dirs created automatically.
        extra_files: {filename: content} — same path convention as context_files
            (may contain subdirectories like "docs/api/api.md"). Written to task_dir
            but NOT listed in task.yaml. Parent dirs created automatically.
            Agent accesses these via read_file tool.
        max_rounds: Maximum eval rounds.
        eval_timeout: Eval timeout in seconds.

    Returns:
        Absolute path to the created task_dir.
    """
    dir_name = f"{op_name}_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    task_dir = os.path.join(base_dir, dir_name)
    os.makedirs(task_dir)

    # Write reference file
    _write_file(task_dir, REF_FILE_DEFAULT, task_desc)

    # Write editable files
    for filename, content in editable_files.items():
        _write_file(task_dir, filename, content)

    editable_file_names = list(editable_files.keys())

    # Multi-file DSL project subtree (e.g. catlass_op/). The DSL adapter
    # owns the copy logic — pass the source directory through; adapter's
    # materialize_project_tree handles per-DSL patching (catlass cmake
    # rewrite etc.). Single-file DSLs pass kernel_project_src=None and
    # this is a no-op.
    if kernel_project_src:
        from akg_agents.op.verifier.adapters.factory import get_dsl_adapter
        adapter = get_dsl_adapter(dsl)
        adapter.materialize_project_tree(task_dir, kernel_project_src)
        for filename in getattr(adapter, "kernel_project_files", []) or []:
            if filename not in editable_file_names:
                editable_file_names.append(filename)

    # program.md is optional now — fundamentals flow through
    # task_dir/skills/<name>/SKILL.md (Layer 0 in system prompt)
    # instead of a single concatenated file. A non-empty program_md
    # still lands on disk and task.yaml so legacy callers work.
    if program_md:
        _write_file(task_dir, "program.md", program_md)

    # Write context_files (listed in task.yaml, enter system prompt)
    context_files = dict(context_files or {})
    for filename, content in context_files.items():
        _write_file(task_dir, filename, content)

    # Only keep program_file in task.yaml when we actually wrote one.
    _program_yaml_key: dict[str, str] = (
        {"program_file": "program.md"} if program_md else {}
    )

    # Write extra_files (NOT in task.yaml, agent reads via read_file)
    if extra_files:
        for filename, content in extra_files.items():
            _write_file(task_dir, filename, content)

    # Generate task.yaml
    task_yaml = {
        "name": op_name,
        "description": "autoresearch optimization",
        "dsl": dsl or None,
        "framework": framework or None,
        "backend": backend or None,
        "arch": arch or None,
        "editable_files": editable_file_names,
        "eval": {
            "timeout": eval_timeout,
            "import_timeout": 0,  # AKG mode: skip import check, syntax-only quick_check
        },
        "metric": {
            "primary": "latency_us",
            "lower_is_better": True,
        },
        "agent": {
            **_program_yaml_key,
            "ref_file": REF_FILE_DEFAULT,
            "context_files": list(context_files.keys()),
            "max_rounds": max_rounds,
        },
    }
    # Per-DSL block (e.g. ``catlass: {root, op_dir}``). Loader's
    # dsl_config normalizer rebuilds the flat keys at load time.
    if dsl_config:
        if dsl == "ascendc_catlass":
            catlass_block = {}
            if dsl_config.get("catlass_root"):
                catlass_block["root"] = dsl_config["catlass_root"]
            if dsl_config.get("catlass_op_dir"):
                catlass_block["op_dir"] = dsl_config["catlass_op_dir"]
            if catlass_block:
                task_yaml["catlass"] = catlass_block
    yaml_content = yaml.dump(task_yaml, default_flow_style=False, allow_unicode=True)
    _write_file(task_dir, "task.yaml", yaml_content)

    # Git init + baseline commit
    _git_init(task_dir)

    return os.path.abspath(task_dir)


def _write_file(task_dir: str, rel_path: str, content: str):
    """Write content to task_dir/rel_path, creating parent dirs as needed."""
    full_path = os.path.join(task_dir, rel_path)
    parent = os.path.dirname(full_path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(content)


def _git_init(task_dir: str):
    """Initialize git repo and create baseline commit."""
    def _run(cmd):
        subprocess.run(cmd, cwd=task_dir, capture_output=True, check=True)

    _run(["git", "init"])
    _run(["git", "config", "user.name", "autoresearch"])
    _run(["git", "config", "user.email", "auto@research"])
    _run(["git", "add", "."])
    _run(["git", "commit", "-m", "scaffold: baseline"])
