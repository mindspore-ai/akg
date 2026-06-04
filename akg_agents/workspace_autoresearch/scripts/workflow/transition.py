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

"""PhaseController — single owner of phase transitions. Callers invoke
``on_*`` events; the controller decides the target phase and writes
``state.json``'s phase field via state_store.write_phase (atomic).

The phase write is its own atomic file write; it doesn't participate
in a multi-file transaction because the new single-file state.json
design eliminated the need for begin_txn/commit_txn coordination.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase_machine import (  # noqa: E402
    BASELINE, EDIT, FINISH, PLAN,
    compute_next_phase, compute_resume_phase, load_progress,
    write_phase,
)


class PhaseController:
    def __init__(self, task_dir: str):
        self.task_dir = task_dir

    # ---- Activation -----------------------------------------------------
    def on_activation_resume(self) -> str:
        return self._write(compute_resume_phase(self.task_dir))

    def on_activation_ready(self) -> str:
        return self._write(BASELINE)

    def on_baseline_settled(self) -> str:
        """Committed baseline (progress present) → PLAN. No progress means
        the baseline gate refused to commit (no valid ref baseline), so
        park the task at BASELINE for a retry after env/ref/worker is
        fixed."""
        progress = load_progress(self.task_dir)
        if progress is None:
            return self._write(BASELINE)
        return self._write(PLAN)

    def on_plan_validated(self) -> str:
        return self._write(EDIT)

    def on_round_settled(self) -> str:
        return self._write(compute_next_phase(self.task_dir))

    def _write(self, phase: str) -> str:
        write_phase(self.task_dir, phase)
        return phase

    # Re-export so callers don't need a separate `from phase_machine import FINISH`.
    FINISH = FINISH
