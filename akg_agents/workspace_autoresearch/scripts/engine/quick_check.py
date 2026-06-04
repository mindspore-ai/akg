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


def _run_codechecker(code: str) -> tuple:
    """One-shot AKG code check. Returns (passed, error_msg, errors)."""
    from akg_agents.op.utils.code_checker import CodeChecker
    return CodeChecker(backend=target_backend(), dsl=target_dsl()).check(code)


def check_editable_files(task_dir: str, config) -> list:
    """Run the AKG code checker on every editable .py.

    Honors `config.code_checker_enabled` — when off, only the
    file-existence check fires; the AST check is skipped. This is the
    single gate consulted by both the runtime quick check and
    `phase_machine.validate_kernel`. Public lib API (no leading
    underscore): both the CLI `main()` below and
    `validators.validate_kernel` call this directly; do not duplicate.

    Returns `[{file, report, errors}]` per failing editable. `report` is
    CodeChecker's formatted multi-line error_msg; `errors` is the
    underlying list of `{line, error_type, detail, suggestion,
    code_snippet, fix_strategy}` dicts.
    """
    issues = []
    use_checker = config.code_checker_enabled
    for fname in config.editable_files:
        if not fname.endswith(".py"):
            continue
        fpath = os.path.join(task_dir, fname)
        if not os.path.exists(fpath):
            issues.append({"file": fname, "report": "file not found", "errors": []})
            continue
        if not use_checker:
            continue
        with open(fpath, "r", encoding="utf-8") as f:
            code = f.read()
        passed, error_msg, errors = _run_codechecker(code)
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
              "only file-existence and smoke test will run.", file=sys.stderr)

    file_issues = check_editable_files(task_dir, config)
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
