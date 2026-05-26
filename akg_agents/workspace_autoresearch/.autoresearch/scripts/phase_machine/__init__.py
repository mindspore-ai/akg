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

This `__init__.py` re-exports only the names imported from outside the
package. Internal helpers (phase tables, regex constants, prompt
templates) stay private to their submodules. Code that needs an
internal can import directly from the submodule
(`from phase_machine.phase_policy import _AR_ALLOWED_BY_PHASE`), making
the cross-package coupling explicit instead of laundering everything
through the facade.

Previously this file re-exported ~30 underscore-prefixed helpers "in
case something needs them"; that turned the facade into a flat
namespace and meant any submodule rename rippled here. The audit that
produced this trim found that most of those re-exports had no external
caller at all.

auto_rollback historically lived here and now sits in utils.git_utils;
the re-export is preserved because too many hook sites import it from
phase_machine.
"""
# fmt: off

# pylint: disable=wrong-import-position,wrong-import-order
import os
import sys

from .models import Progress
from .state_store import (
    # Phase constants
    INIT, BASELINE, PLAN, EDIT,
    DIAGNOSE, REPLAN, FINISH, ALL_PHASES,
    # File constants
    PHASE_FILE, PROGRESS_FILE, HISTORY_FILE, PLAN_FILE, PLAN_ITEMS_FILE,
    EDIT_MARKER_FILE, PENDING_SETTLE_FILE, HEARTBEAT_FILE, ACTIVE_TASK_FILE,
    DIAGNOSE_ARTIFACT_TEMPLATE, DIAGNOSE_MARKER_TEMPLATE, DIAGNOSE_ATTEMPTS_CAP,
    # Path builders
    state_path, plan_path, progress_path, history_path, edit_marker_path,
    pending_settle_path,
    diagnose_artifact_path, diagnose_marker,
    # Phase I/O
    read_phase, write_phase,
    # Progress + history I/O
    load_progress, save_progress, append_history, update_progress,
    # Active-task pointer
    get_task_dir, set_task_dir, touch_heartbeat,
    find_active_task_dir,
)
from .validators import (
    validate_kernel, validate_plan, validate_diagnose,
    DiagnoseState, diagnose_state,
    DIAGNOSE_NEED_DIAGNOSIS, DIAGNOSE_READY, DIAGNOSE_MANUAL_FALLBACK,
    get_plan_items, parse_plan_text, has_pending_items, get_active_item,
    is_settled_table_header,
    # workflow.planning is the one external user of the plan-line regex.
    # Underscore is kept to flag "not a stable public name; use a higher
    # level helper if you can".
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

# auto_rollback used to live in phase_machine; the implementation moved to
# git_utils alongside commit_in_task / ensure_git_identity. Re-export so
# stale `from phase_machine import auto_rollback` still resolves.
_scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)
from utils.git_utils import auto_rollback  # noqa: E402
# fmt: on
