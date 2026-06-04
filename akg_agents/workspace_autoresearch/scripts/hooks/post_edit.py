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
PostToolUse hook for Edit/Write — surfaces guidance after EDIT-phase edits.

Edits during EDIT phase are followed by pipeline.py to settle a round.

plan.md is never a legal target for Edit/Write — hooks/guard_edit blocks it
at every phase and directs Claude to create_plan.py.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hooks.utils import read_hook_input, emit_status, extract_target_path
from phase_machine import (
    read_phase,
    get_task_dir, touch_heartbeat,
    EDIT,
)


_WRITE_TOOLS = {"Edit", "Write", "MultiEdit", "NotebookEdit"}


def _safe_load_config(task_dir: str):
    """Best-effort TaskConfig load. Returns None on any failure — this
    hook fires on every Edit/Write, including before AR scaffolding has
    written task.yaml, so we must tolerate the load failing without
    bubbling an exception out of the hook process.
    """
    try:
        from task_config import load_task_config
        return load_task_config(task_dir)
    except Exception:
        return None


def main():
    hook_input = read_hook_input()
    if hook_input.get("tool_name", "") not in _WRITE_TOOLS:
        sys.exit(0)

    task_dir = get_task_dir()
    if not task_dir:
        sys.exit(0)
    touch_heartbeat(task_dir)

    file_path = extract_target_path(hook_input)
    if not file_path:
        sys.exit(0)

    phase = read_phase(task_dir)

    config = _safe_load_config(task_dir)
    is_editable = False
    if config:
        try:
            rel = os.path.relpath(file_path, task_dir).replace("\\", "/")
            is_editable = rel in set(config.editable_files)
        except ValueError:
            is_editable = False

    if is_editable and phase == EDIT:
        emit_status(
            f"[AR] Code edited. Continue editing OR run: "
            f"python scripts/engine/pipeline.py \"{task_dir}\""
        )

    sys.exit(0)


if __name__ == "__main__":
    main()
