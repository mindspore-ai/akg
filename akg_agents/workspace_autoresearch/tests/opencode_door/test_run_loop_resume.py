# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Ownership wiring for the OpenCode headless resume entry."""

import importlib.util
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "autoresearch_run_loop_resume_test", ROOT / ".opencode" / "run_loop.py"
)
RUN_LOOP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUN_LOOP)


@pytest.mark.level0
def test_resume_uses_canonical_lifecycle_and_explicit_force(tmp_path, monkeypatch):
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(
            cmd, 0, stdout=f"[resume] Phase: REPLAN\n{task_dir}\n", stderr=""
        )

    monkeypatch.setattr(RUN_LOOP.subprocess, "run", fake_run)

    resumed = RUN_LOOP._resume_task(str(task_dir), force=True)

    assert resumed == str(task_dir.resolve())
    assert captured["cmd"] == [
        RUN_LOOP.sys.executable,
        str(ROOT / "scripts" / "resume.py"),
        str(task_dir.resolve()),
        "--force",
    ]
    assert captured["kwargs"]["cwd"] == str(ROOT)


@pytest.mark.level0
def test_resume_refusal_is_not_bypassed(tmp_path, monkeypatch):
    task_dir = tmp_path / "task"
    task_dir.mkdir()

    monkeypatch.setattr(
        RUN_LOOP.subprocess,
        "run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(
            cmd, 1, stdout="", stderr="owned by a live session"
        ),
    )

    with pytest.raises(SystemExit, match="owned by a live session"):
        RUN_LOOP._resume_task(str(task_dir), force=False)
