#!/usr/bin/env python3
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

"""
Quick static check for editable files before eval.

Delegates to `akg_agents.op.utils.code_checker.CodeChecker` — covers
syntax, py_compile, import availability, stray-Chinese-in-code, the DSL
compliance gate (no @triton.jit, forbidden torch.* in forward, etc.),
and Triton-autotune restore_value. Catches regressions before we pay
the cost of a real eval.

Honors `config.code_checker_enabled` — when off (task.yaml
`code_checker.enabled: false` or scaffold's `--no-code-checker`), only
file existence + the optional smoke test run.

Usage:
    python scripts/engine/quick_check.py <task_dir>

Output:
    stdout: 'OK' on pass, JSON error blob on fail
    exit 0 = pass, 1 = fail
"""
import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_config import load_task_config
from utils.settings import target_backend, target_dsl


def _run_codechecker(code: str, file_name: str = "") -> tuple:
    """One-shot AKG code check. Returns (passed, error_msg, errors)."""
    from akg_agents.op.utils.code_checker import CodeChecker
    return CodeChecker(backend=target_backend(), dsl=target_dsl()).check(
        code, task_info={"file": file_name}
    )


def _diff_changed_lines(diff_text: str) -> list[tuple[str, str]]:
    """Return (file_path, line_text) for added/removed content lines."""
    out: list[tuple[str, str]] = []
    current_file = ""
    for raw in diff_text.splitlines():
        if raw.startswith("+++ "):
            path = raw[4:].strip()
            if path.startswith("b/"):
                path = path[2:]
            current_file = path
            continue
        if raw.startswith(("diff --git ", "index ", "--- ", "@@")):
            continue
        if not raw or raw[0] not in "+-":
            continue
        if raw.startswith(("+++", "---")):
            continue
        out.append((current_file, raw[1:]))
    return out


def _is_comment_line(path: str, line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True

    name = os.path.basename(path).lower()
    ext = os.path.splitext(name)[1]

    if ext == ".py" or name == "cmakelists.txt" or ext == ".cmake":
        return stripped.startswith("#")

    if ext in {".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hh", ".asc"}:
        return (
            stripped.startswith("//")
            or stripped.startswith("/*")
            or stripped.startswith("*")
            or stripped.startswith("*/")
        )

    return False


def effective_edit_issue(task_dir: str, config) -> dict | None:
    """Reject no-op, whitespace-only, and comment-only edit rounds.

    CodeChecker validates file legality, but it cannot tell whether this
    round made a meaningful exploratory edit. That is a task/git question,
    so keep it here in quick_check where pipeline.py already has task_dir.
    """
    if not config.editable_files:
        return None

    try:
        diff = subprocess.run(
            ["git", "diff", "--no-ext-diff", "--unified=0", "--",
             *config.editable_files],
            cwd=task_dir,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
    except Exception as exc:
        return {
            "file": "(edit-diff)",
            "report": f"cannot inspect edit diff: {type(exc).__name__}: {exc}",
            "errors": [],
        }

    if diff.returncode != 0:
        msg = (diff.stderr or diff.stdout or "git diff failed").strip()
        return {
            "file": "(edit-diff)",
            "report": msg[:800],
            "errors": [],
        }

    diff_text = diff.stdout or ""
    if not diff_text.strip():
        return {
            "file": "(edit-diff)",
            "report": (
                "Zero-edit: no git-visible changes in editable files. "
                "The code may have rolled back to HEAD. Read broader "
                "SKILL.md references, then make a substantive code edit."
            ),
            "errors": [],
        }

    changed = _diff_changed_lines(diff_text)
    if changed and all(_is_comment_line(path, line) for path, line in changed):
        return {
            "file": "(edit-diff)",
            "report": (
                "Comment-only edit: the diff changes only comments or "
                "whitespace. Keep SKILL citations in the plan, then edit "
                "executable/source logic."
            ),
            "errors": [],
        }

    return None


def check_editable_files(task_dir: str, config) -> list:
    """Run the AKG code checker on every editable .py.

    Honors `config.code_checker_enabled` — when off, only the
    file-existence check fires; the AST check is skipped. This is the
    single gate consulted by the runtime quick-check CLI.

    Returns `[{file, report, errors}]` per failing editable. `report` is
    CodeChecker's formatted multi-line error_msg; `errors` is the
    underlying list of `{line, error_type, detail, suggestion,
    code_snippet, fix_strategy}` dicts.
    """
    issues = []
    use_checker = config.code_checker_enabled
    dsl = target_dsl()
    for fname in config.editable_files:
        if not fname.endswith(".py") and dsl != "ascendc":
            continue
        fpath = os.path.join(task_dir, fname)
        if not os.path.exists(fpath):
            issues.append({"file": fname, "report": "file not found", "errors": []})
            continue
        if not use_checker:
            continue
        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
            code = f.read()
        passed, error_msg, errors = _run_codechecker(code, fname)
        if not passed:
            issues.append({
                "file": fname,
                "report": error_msg or "code check failed",
                "errors": errors or [],
            })
    return issues


def _run_smoke_test(task_dir: str, config) -> list:
    if not config.smoke_test_script:
        return []
    smoke_path = os.path.join(task_dir, config.smoke_test_script)
    if not os.path.exists(smoke_path):
        return []
    try:
        r = subprocess.run(
            [sys.executable, smoke_path],
            capture_output=True, text=True,
            encoding="utf-8", errors="replace",
            timeout=config.smoke_test_timeout, cwd=task_dir,
        )
    except subprocess.TimeoutExpired:
        return [f"smoke test timed out after {config.smoke_test_timeout}s"]
    except Exception as e:
        return [f"smoke test launch error: {e}"]
    if r.returncode != 0:
        tail = (r.stderr or "")[-500:]
        return [f"smoke test failed (exit {r.returncode}): {tail}"]
    return []


def main():
    parser = argparse.ArgumentParser(description="Quick static check")
    parser.add_argument("task_dir", help="Path to task directory")
    args = parser.parse_args()

    task_dir = os.path.abspath(args.task_dir)
    config = load_task_config(task_dir)
    if config is None:
        print(json.dumps({"ok": False, "error": "task.yaml not found"}))
        sys.exit(1)

    if not config.code_checker_enabled:
        print("[quick_check] Triton regression check disabled in task.yaml — "
              "only file-existence and smoke test will run.")

    edit_issue = effective_edit_issue(task_dir, config)
    file_issues = [edit_issue] if edit_issue else check_editable_files(
        task_dir, config)
    smoke_errors = _run_smoke_test(task_dir, config)

    if not file_issues and not smoke_errors:
        print("OK")
        sys.exit(0)

    blob = {"ok": False}
    if file_issues:
        blob["file_issues"] = file_issues
    if smoke_errors:
        blob["smoke_errors"] = smoke_errors
    print(json.dumps(blob, ensure_ascii=False))
    sys.exit(1)


if __name__ == "__main__":
    main()
