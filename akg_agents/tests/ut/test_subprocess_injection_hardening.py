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

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from akg_agents.cli.runtime.common_tools import BashTool
from akg_agents.core_v2.tools import basic_tools
from akg_agents.op.verifier import profiler_utils


class _Completed:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_check_markdown_passes_shell_metachar_path_as_single_arg(monkeypatch, tmp_path):
    malicious_path = tmp_path / "doc;touch injected.md"
    malicious_path.write_text("# title\n", encoding="utf-8")
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return _Completed(returncode=0)

    monkeypatch.setattr(basic_tools.subprocess, "run", fake_run)

    result = basic_tools.check_markdown(str(malicious_path))

    assert result["status"] == "success"
    assert calls[0][0] == ["markdownlint", "--version"]
    assert calls[1][0] == ["markdownlint", str(malicious_path)]
    assert all(call_kwargs.get("shell") is not True for _, call_kwargs in calls)


def test_bash_execute_splits_command_without_shell_interpretation(monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return _Completed(returncode=0, stdout="ok\n")

    monkeypatch.setattr(subprocess, "run", fake_run)

    output = BashTool._execute("printf '%s' 'a;b'", cwd=None, timeout=1)

    assert output == "[exit_code=0]\nok"
    assert calls == [
        (
            ["printf", "%s", "a;b"],
            {"cwd": None, "text": True, "capture_output": True, "timeout": 1},
        )
    ]


def test_run_msprof_passes_shell_metachar_path_as_single_application_arg(monkeypatch):
    script_path = "/tmp/profile;touch injected.py"
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return _Completed(
            returncode=0,
            stdout="[INFO] Process profiling data complete. Data is saved in /tmp/prof\n",
        )

    monkeypatch.setattr(profiler_utils.subprocess, "run", fake_run)

    success, error, prof_path = profiler_utils.run_msprof(script_path, timeout=7)

    assert success is True
    assert error == ""
    assert prof_path == "/tmp/prof"
    assert calls == [
        (
            ["msprof", "--application", f"python {script_path}"],
            {"capture_output": True, "text": True, "timeout": 7},
        )
    ]


def test_run_nsys_and_stats_use_argv_without_shell(monkeypatch, tmp_path):
    script_path = str(tmp_path / "profile&&touch injected.py")
    rep_path = str(tmp_path / "report;touch injected.nsys-rep")
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return _Completed(returncode=0)

    monkeypatch.setattr(profiler_utils.subprocess, "run", fake_run)

    profiler_utils.run_nsys(script_path, timeout=9)
    profiler_utils.analyze_nsys_data(rep_path, warmup_times=1, run_times=1)

    assert calls[0] == (
        ["nsys", "profile", "--output=nsys_report_profile&&touch injected", "python", script_path],
        {"capture_output": True, "text": True, "timeout": 9},
    )
    assert calls[1][0][-1] == rep_path
    assert calls[1][1] == {"check": True}
    assert all(call_kwargs.get("shell") is not True for _, call_kwargs in calls)
