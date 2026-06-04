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

"""Stop hook: allow Stop at FINISH or when baseline can't be established.

Default behaviour blocks Stop in every non-FINISH phase so the agent
can't bail out of the optimisation loop. One state breaks that rule:
the task is parked at BASELINE with NO committed progress — the ref-
baseline gate (workflow.baseline) refused to commit because no valid
PyTorch reference was measured (reference.py invalid, env/runtime
missing, worker disconnect). The agent can't fix any of those, and
guard_edit blocks "fixes" to ref / external files, so without this
carve-out the agent would loop forever. The emitted status text tells
the user what to fix and how to resume.
"""
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hooks.utils import read_hook_input, emit_status
from phase_machine import (
    BASELINE, FINISH, get_guidance, get_task_dir,
    load_progress, read_phase, update_progress,
)


def _block_stop_with_reason(reason: str) -> None:
    print(json.dumps({"decision": "block", "reason": reason}))
    sys.exit(0)


def _is_stuck(phase: str, progress) -> bool:
    """True when the task is parked at BASELINE with no committed
    progress — the baseline gate refused to commit because no valid
    PyTorch reference was measured (env / ref / worker broken). A
    kernel rewrite cannot fix any of those, so the agent must be allowed
    to Stop and the banner must point at the real cause. PLAN/EDIT etc.
    are not "stuck" even if eval fails (max_rounds routes them to
    FINISH)."""
    return phase == BASELINE and progress is None


def main():
    stop_reason = read_hook_input().get("stop_reason", "unknown")

    task_dir = get_task_dir()
    if not task_dir:
        sys.exit(0)

    progress = load_progress(task_dir)
    phase = read_phase(task_dir)
    stuck = _is_stuck(phase, progress)

    if phase != FINISH and not stuck:
        _block_stop_with_reason(
            f"[AR] Cannot Stop at phase={phase}. Continue the loop:\n\n"
            f"{get_guidance(task_dir)}"
        )

    if stuck:
        # Baseline pending (no committed progress at BASELINE) — the ref-
        # baseline gate refused to commit because no valid PyTorch ref
        # was measured. There's no progress record to update; emit the
        # recovery banner and exit. The specific cause (ref vs env vs
        # worker) isn't recorded — they all converge here.
        emit_status(
            "\n[AR] Task aborted at BASELINE: no valid PyTorch reference "
            "(env / ref / worker)."
        )
        emit_status(
            f"[AR] Fix the cause and `/autoresearch --resume {task_dir}` "
            f"to retry baseline, or re-scaffold from a fixed --ref."
        )
        return

    if progress is None:
        sys.exit(0)

    update_progress(
        task_dir,
        last_stop_reason=stop_reason,
        last_stop_time=datetime.now(timezone.utc).isoformat(),
    )

    rounds = progress.get("eval_rounds", 0)
    max_rounds = progress.get("max_rounds", 0)
    best = progress.get("best_metric")
    baseline = progress.get("baseline_metric")

    improv = ""
    if best is not None and baseline is not None and baseline != 0:
        pct = (baseline - best) / abs(baseline) * 100
        improv = f" ({pct:+.1f}%)"

    emit_status(f"\n[AR] Session stopped at FINISH: {stop_reason}")
    emit_status(f"[AR] {rounds}/{max_rounds} rounds | Best: {best}{improv}")
    emit_status(f"[AR] Resume: /autoresearch --resume {task_dir}")


if __name__ == "__main__":
    main()
