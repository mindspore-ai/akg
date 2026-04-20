# Copyright 2025-2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Source-level invariants after the Phase 1 supervisor cleanup.

Pins the post-supervisor + post-abandon surface so shim aliases /
dead fields / terminal-state leftovers cannot sneak back in via a
copy-paste:

  * SkillBuilder exposes only canonical names (``register``,
    ``mark_unbound``, ``record_applied``) — no ``mark_selected`` /
    ``force_abandon_binding`` shim aliases, no ``mark_item_active`` /
    ``on_plan_item_settled`` / ``on_plan_superseded`` legacy stubs,
    no terminal ``mark_abandoned`` / ``is_bindable`` /
    ``get_abandoned_names`` from the pre-tier era.
  * scaffold_task_dir no longer accepts ``supervisor=`` nor writes a
    ``supervisor_enabled`` key into task.yaml.
  * TurnResult no longer carries the ``last_edit_diff`` field (dead
    since the supervisor was removed — nothing read it).
"""
import inspect
from types import SimpleNamespace

import pytest

from akg_agents.op.autoresearch.adapters.task_scaffolder import scaffold_task_dir
from akg_agents.op.autoresearch.agent.skill_builder import SkillBuilder
from akg_agents.op.autoresearch.agent.turn import TurnResult


# -- SkillBuilder surface -----------------------------------------------------


class TestSkillBuilderSurface:
    def test_canonical_names_present(self):
        sb = SkillBuilder()
        assert callable(sb.register)
        assert callable(sb.mark_unbound)
        assert callable(sb.record_applied)

    def test_shim_aliases_removed(self):
        sb = SkillBuilder()
        assert not hasattr(sb, "mark_selected"), (
            "mark_selected shim must be removed — callers should use register"
        )
        assert not hasattr(sb, "force_abandon_binding"), (
            "force_abandon_binding shim must be removed — callers should use "
            "mark_unbound"
        )

    def test_terminal_abandon_api_is_gone(self):
        """Post-abandon refactor: every registered skill is bindable.
        The three terminal-state surfaces from the pre-tier era must
        NOT reappear — they hard-excluded skills from later binding."""
        sb = SkillBuilder()
        for name in ("mark_abandoned", "is_bindable", "get_abandoned_names"):
            assert not hasattr(sb, name), (
                f"SkillBuilder.{name} belonged to the terminal-state "
                f"design; the tier model replaces it with mark_unbound / "
                f"get_unbound_names / tier()."
            )

    def test_legacy_noop_stubs_removed(self):
        """The three supervisor-era plan-state hooks should be gone."""
        sb = SkillBuilder()
        for name in ("mark_item_active", "on_plan_item_settled",
                     "on_plan_superseded"):
            assert not hasattr(sb, name), (
                f"SkillBuilder.{name} was a supervisor-era no-op stub; "
                f"feedback.py now calls record_applied / mark_unbound "
                f"directly."
            )

    def test_register_after_unbind_does_not_reset_history(self):
        """Re-registering a previously-unbound skill keeps the
        ``unbound_at_versions`` list — the history is a badge, not a
        reset trigger."""
        sb = SkillBuilder()
        skill = SimpleNamespace(name="fx", category="guide",
                                description="d", content="c", metadata={})
        sb.register(skill, reason="first", plan_version=1)
        sb.mark_unbound("fx", reason="bad", plan_version=1)
        sb.register(skill, reason="second", plan_version=2)
        rec = sb.get("fx")
        assert rec.unbound_at_versions == [1]
        # Still bindable: registered = bindable post-refactor.
        assert "fx" in sb


# -- scaffold_task_dir signature ---------------------------------------------


class TestScaffoldSignature:
    def test_supervisor_param_gone(self):
        sig = inspect.signature(scaffold_task_dir)
        assert "supervisor" not in sig.parameters, (
            "scaffold_task_dir(supervisor=...) was a dead param — the "
            "supervisor system is gone."
        )

    def test_yaml_omits_supervisor_enabled(self, tmp_path):
        td = scaffold_task_dir(
            base_dir=str(tmp_path),
            op_name="probe",
            task_desc="class Model: pass",
            editable_files={"kernel.py": "x = 1"},
            max_rounds=3,
        )
        import yaml
        import os
        with open(os.path.join(td, "task.yaml"), encoding="utf-8") as f:
            data = yaml.safe_load(f)
        # The key must not be present anywhere under agent.*
        agent_cfg = (data.get("agent") or {}).get("config") or {}
        assert "supervisor_enabled" not in agent_cfg
        # And there should be no top-level leftover either.
        assert "supervisor_enabled" not in data


# -- TurnResult shape --------------------------------------------------------


class TestTurnResultShape:
    def test_last_edit_diff_field_removed(self):
        fields = {f.name for f in TurnResult.__dataclass_fields__.values()}
        assert "last_edit_diff" not in fields, (
            "TurnResult.last_edit_diff was only read by the now-removed "
            "supervisor; the field and its producer _compute_edit_diff are "
            "dead weight."
        )

    def test_core_fields_still_present(self):
        """Regression guard for the fields that ARE still in use."""
        fields = {f.name for f in TurnResult.__dataclass_fields__.values()}
        for required in ("outcome", "tool_calls", "results_log",
                         "feedback", "eval_record", "has_finish"):
            assert required in fields
