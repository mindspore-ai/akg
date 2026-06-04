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

"""phase_machine package — facade over four single-concern submodules.

Dependency direction (top depends on lower):
    guidance, phase_policy
        → validators
            → state_store

Public surface centers on a single per-task state record
(<task_dir>/.ar_state/state.json). Every piece of control state lives
in state.json, atomic write of state.json IS the transaction commit,
and cross-file consistency with the two durable artifacts (plan.md,
history.jsonl) is checked via state.expected_* fields.
"""
# fmt: off
from .models import Progress
from .state_store import (
    # Phase constants
    INIT, BASELINE, PLAN, EDIT,
    DIAGNOSE, REPLAN, FINISH, ALL_PHASES,
    # File constants
    STATE_FILE, HISTORY_FILE, PLAN_FILE, PLAN_ITEMS_FILE, EDIT_MARKER_FILE,
    INTENT_FILE,
    DIAGNOSE_ARTIFACT_TEMPLATE, DIAGNOSE_MARKER_TEMPLATE, DIAGNOSE_ATTEMPTS_CAP,
    # Path builders
    state_path, state_record_path, plan_path, history_path, edit_marker_path,
    intent_path, diagnose_artifact_path, diagnose_marker,
    # Journal / write-ahead intent (closes the bodies-without-state crash window)
    write_intent, read_intent, clear_intent, replay_intent,
    # State record I/O — single source of truth
    load_state, save_state, update_state,
    # Typed views over state.json
    read_phase, write_phase,
    load_progress, save_progress, append_history, update_progress,
    # Ownership (per-task owner field in state.json)
    get_task_dir, set_task_dir, clear_active_task, touch_heartbeat,
    find_active_task_dir, current_session_task_dir,
    # Per-op task_dir pointer (scaffold -> batch.run.py handoff)
    task_dir_pointer_path, write_task_dir_pointer, read_task_dir_pointer,
    # Cross-file consistency gate
    check_state_consistency, format_state_inconsistency,
    require_state_consistency,
    # Outward-facing facade — preferred over direct state.json reads in
    # batch / resume / dashboard / scaffold.
    task_summary, is_task_active, task_owner_info,
)
from .validators import (
    validate_plan,
    DiagnoseState, diagnose_state,
    DIAGNOSE_NEED_DIAGNOSIS, DIAGNOSE_READY, DIAGNOSE_MANUAL_FALLBACK,
    get_plan_items, parse_plan_text, has_pending_items, get_active_item,
    is_settled_table_header,
    _PLAN_ITEM_RE,
)
from .phase_policy import (
    classify,
    parse_script_names, parse_invoked_ar_script,
    is_single_foreground_ar_invocation,
    check_bash, check_edit,
    compute_next_phase, compute_resume_phase,
)
from .guidance import (
    get_guidance,
)

# Re-export auto_rollback from utils.git_utils (lives there alongside
# commit_in_task / ensure_git_identity).
import os
import sys
_scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)
from utils.git_utils import auto_rollback  # noqa: E402
# fmt: on
