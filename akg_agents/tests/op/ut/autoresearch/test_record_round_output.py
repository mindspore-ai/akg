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

from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import yaml


SCRIPTS_DIR = Path(__file__).resolve().parents[4] / "workspace_autoresearch" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from phase_machine import save_state  # noqa: E402
from workflow.round import record_round  # noqa: E402


pytestmark = pytest.mark.skipif(
    shutil.which("git") is None,
    reason="git not on PATH; record_round rollback tests require git",
)


def _git(task_dir: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=task_dir, check=True,
                   capture_output=True, text=True)


def _make_task(tmp_path: Path) -> Path:
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "kernel.py").write_text("x = 1\n", encoding="utf-8")
    (task_dir / "task.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "case",
                "editable_files": ["kernel.py"],
                "metric": {
                    "primary": "latency_us",
                    "lower_is_better": True,
                    "improvement_threshold": 0.0,
                },
                "agent": {"max_rounds": 3},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    _git(task_dir, "init", "-q")
    _git(task_dir, "config", "user.email", "test@autoresearch")
    _git(task_dir, "config", "user.name", "Autoresearch Test")
    _git(task_dir, "add", "kernel.py", "task.yaml")
    _git(task_dir, "commit", "-q", "-m", "seed")
    save_state(str(task_dir), {
        "phase": "EDIT",
        "progress_initialized": True,
        "task": "case",
        "eval_rounds": 0,
        "max_rounds": 3,
        "consecutive_failures": 0,
        "best_metric": 100.0,
        "best_commit": "seed",
        "baseline_metric": 100.0,
        "baseline_source": "ref",
        "baseline_outcome": "ok",
        "baseline_error_source": None,
        "baseline_per_shape_us": [100.0],
        "baseline_fingerprint": {"num_cases": 1, "shape_signature": "test"},
        "seed_metric": 100.0,
        "plan_version": 1,
        "next_pid": 1,
    })
    return task_dir


def test_discard_rollback_message_uses_stdout_not_stderr(tmp_path, capsys):
    task_dir = _make_task(tmp_path)
    (task_dir / "kernel.py").write_text("x = 2\n", encoding="utf-8")

    result = record_round(
        str(task_dir),
        {"correctness": True, "metrics": {"latency_us": 110.0}},
        description="slower edit",
        plan_item="p1",
    )

    captured = capsys.readouterr()
    assert result["decision"] == "DISCARD"
    assert "[record_round] DISCARD: rolled back editable files" in captured.out
    assert captured.err == ""


def test_fail_rollback_message_uses_stdout_not_stderr(tmp_path, capsys):
    task_dir = _make_task(tmp_path)
    (task_dir / "kernel.py").write_text("x = 2\n", encoding="utf-8")

    result = record_round(
        str(task_dir),
        {"correctness": False, "metrics": {}, "error": "bad result"},
        description="bad edit",
        plan_item="p1",
    )

    captured = capsys.readouterr()
    assert result["decision"] == "FAIL"
    assert "[record_round] FAIL: correctness check failed" in captured.out
    assert "[record_round] FAIL: rolled back editable files" in captured.out
    assert captured.err == ""
