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
"""Tests for SkillBuilder registry — post-abandon refactor.

No terminal states. Every registered skill stays bindable; the
``unbound_at_versions`` list is a demotion signal used by
``tier()`` to put previously-unbound skills last in binding priority.
Applied always beats Unbound: a KEEP on a skill that was once
unbound promotes it back to tier 0.
"""

from types import SimpleNamespace


from akg_agents.op.autoresearch.agent.skill_builder import (
    SkillBuilder, SkillRecord,
)


def _skill(name: str, category: str = "guide", content: str = "",
           description: str = "desc") -> SimpleNamespace:
    return SimpleNamespace(
        name=name, category=category, content=content,
        description=description, metadata={},
    )


class TestRegisterAndQuery:
    def test_register_creates_record(self):
        sb = SkillBuilder()
        sb.register(_skill("guide-a"), reason="initial", plan_version=1)
        assert "guide-a" in sb
        rec = sb.get("guide-a")
        assert rec is not None
        assert rec.unbound_at_versions == []
        assert rec.applied_versions == []
        assert rec.registered_reason == "initial"
        assert rec.registered_at_version == 1

    def test_register_is_idempotent_updates_reason(self):
        sb = SkillBuilder()
        sb.register(_skill("g"), reason="first", plan_version=1)
        sb.register(_skill("g"), reason="second", plan_version=2)
        assert len(sb._skills) == 1
        assert sb.get("g").registered_reason == "second"
        assert sb.get("g").registered_at_version == 2

    def test_is_empty(self):
        sb = SkillBuilder()
        assert sb.is_empty() is True
        sb.register(_skill("g"), reason="r")
        assert sb.is_empty() is False

    def test_registered_is_always_bindable_via_contains(self):
        """Post-abandon refactor: 'registered' == 'bindable'. Callers
        check ``name in sb`` instead of the removed ``is_bindable``."""
        sb = SkillBuilder()
        sb.register(_skill("g"), reason="r", plan_version=0)
        assert "g" in sb
        assert "unknown" not in sb


class TestMarkUnbound:
    def test_unbind_appends_version(self):
        sb = SkillBuilder()
        sb.register(_skill("g"), reason="r", plan_version=1)
        sb.mark_unbound("g", reason="pattern mismatch", plan_version=2)
        rec = sb.get("g")
        assert rec.unbound_at_versions == [2]
        assert "pattern mismatch" in rec.last_reason

    def test_unbind_is_not_terminal(self):
        """Previously-unbound skills stay in the registry and remain
        bindable — ``tier()`` demotes them to 2 instead."""
        sb = SkillBuilder()
        sb.register(_skill("g"), reason="r", plan_version=1)
        sb.mark_unbound("g", reason="bad fit", plan_version=2)
        assert "g" in sb  # still bindable
        assert sb.get("g").tier() == 2

    def test_same_version_unbind_is_suppressed(self):
        sb = SkillBuilder()
        sb.register(_skill("g"), reason="r", plan_version=1)
        sb.mark_unbound("g", reason="first", plan_version=2)
        sb.mark_unbound("g", reason="second", plan_version=2)
        assert sb.get("g").unbound_at_versions == [2]

    def test_multiple_unbinds_accumulate(self):
        sb = SkillBuilder()
        sb.register(_skill("g"), reason="r", plan_version=1)
        sb.mark_unbound("g", reason="x", plan_version=2)
        sb.mark_unbound("g", reason="y", plan_version=5)
        assert sb.get("g").unbound_at_versions == [2, 5]

    def test_unknown_skill_is_noop(self):
        sb = SkillBuilder()
        sb.mark_unbound("missing", reason="r", plan_version=1)
        assert sb.is_empty() is True

    def test_get_unbound_names(self):
        sb = SkillBuilder()
        sb.register(_skill("a"), reason="r", plan_version=1)
        sb.register(_skill("b"), reason="r", plan_version=1)
        sb.mark_unbound("b", reason="x", plan_version=2)
        assert sb.get_unbound_names() == {"b"}


class TestRecordApplied:
    def test_applied_badge_added(self):
        sb = SkillBuilder()
        sb.register(_skill("g"), reason="r", plan_version=1)
        sb.record_applied("g", plan_version=2, reason="KEEP")
        rec = sb.get("g")
        assert rec.applied_versions == [2]
        assert "KEEP" in rec.last_reason
        assert sb.get_applied_names() == {"g"}

    def test_multiple_versions_accumulate(self):
        sb = SkillBuilder()
        sb.register(_skill("g"), reason="r")
        sb.record_applied("g", plan_version=2)
        sb.record_applied("g", plan_version=5)
        sb.record_applied("g", plan_version=7)
        assert sb.get("g").applied_versions == [2, 5, 7]

    def test_duplicate_version_suppressed(self):
        sb = SkillBuilder()
        sb.register(_skill("g"), reason="r")
        sb.record_applied("g", plan_version=3)
        sb.record_applied("g", plan_version=3)
        assert sb.get("g").applied_versions == [3]

    def test_applied_after_unbind_promotes_back_to_tier_0(self):
        """The whole point of dropping terminal abandon: a skill
        unbound on one item can be KEEP'd on another, and the KEEP
        promotes it back to tier 0 (applied beats unbound)."""
        sb = SkillBuilder()
        sb.register(_skill("g"), reason="r", plan_version=1)
        sb.mark_unbound("g", reason="bad fit on p1", plan_version=2)
        assert sb.get("g").tier() == 2
        sb.record_applied("g", plan_version=5, reason="KEEP on p3")
        assert sb.get("g").tier() == 0
        # Both history lists are retained — they're badges, not gates.
        assert sb.get("g").applied_versions == [5]
        assert sb.get("g").unbound_at_versions == [2]

    def test_applied_noop_on_unknown(self):
        sb = SkillBuilder()
        sb.record_applied("missing", plan_version=1)
        assert sb.is_empty() is True


class TestTier:
    def test_tier_default_is_1_for_fresh_register(self):
        sb = SkillBuilder()
        sb.register(_skill("g"), reason="r", plan_version=1)
        assert sb.get("g").tier() == 1

    def test_applied_wins_over_unbound(self):
        sb = SkillBuilder()
        sb.register(_skill("g"), reason="r", plan_version=1)
        sb.mark_unbound("g", reason="x", plan_version=2)
        sb.record_applied("g", plan_version=3)
        assert sb.get("g").tier() == 0

    def test_unbound_alone_is_tier_2(self):
        sb = SkillBuilder()
        sb.register(_skill("g"), reason="r", plan_version=1)
        sb.mark_unbound("g", reason="x", plan_version=2)
        assert sb.get("g").tier() == 2


class TestSerialization:
    def test_roundtrip_empty(self):
        sb = SkillBuilder()
        d = sb.skill_state_to_dict()
        assert d["version"] == 5
        sb2 = SkillBuilder()
        sb2.skill_state_from_dict(d)
        assert sb2.is_empty()

    def test_roundtrip_populated(self):
        sb = SkillBuilder()
        sb.register(_skill("a"), reason="ra", plan_version=1)
        sb.register(_skill("b", "example"), reason="rb", plan_version=1)
        sb.record_applied("a", plan_version=2, reason="kept")
        sb.mark_unbound("b", reason="bad fit", plan_version=3)
        d = sb.skill_state_to_dict()

        sb2 = SkillBuilder()
        sb2.skill_state_from_dict(d)
        assert "a" in sb2 and "b" in sb2
        assert sb2.get("a").applied_versions == [2]
        assert sb2.get("b").unbound_at_versions == [3]
        assert sb2.get_unbound_names() == {"b"}
        assert sb2.get_applied_names() == {"a"}

    def test_v4_is_migrated_to_unbound(self):
        """v4 payloads used terminal ``abandoned=True`` + scalar
        ``abandoned_at_version``. Loading must convert to v5's
        ``unbound_at_versions=[version]``."""
        sb = SkillBuilder()
        sb.skill_state_from_dict({
            "version": 4,
            "skills": {
                "legacy": {
                    "name": "legacy",
                    "category": "guide",
                    "applied_versions": [],
                    "abandoned": True,
                    "abandoned_at_version": 7,
                    "last_reason": "old terminal abandon",
                },
            },
        })
        rec = sb.get("legacy")
        assert rec is not None
        assert rec.unbound_at_versions == [7]
        # v4's abandoned entry becomes tier 2 post-migration, not terminal.
        assert rec.tier() == 2
        assert "legacy" in sb  # still bindable

    def test_v4_non_abandoned_migrates_cleanly(self):
        sb = SkillBuilder()
        sb.skill_state_from_dict({
            "version": 4,
            "skills": {
                "fresh": {
                    "name": "fresh",
                    "applied_versions": [2],
                    "abandoned": False,
                    "abandoned_at_version": None,
                },
            },
        })
        rec = sb.get("fresh")
        assert rec is not None
        assert rec.unbound_at_versions == []
        assert rec.applied_versions == [2]
        assert rec.tier() == 0

    def test_unknown_version_rejected(self):
        sb = SkillBuilder()
        sb.skill_state_from_dict({
            "version": 3,
            "skills": {"legacy": {"name": "legacy", "status": "abandoned"}},
        })
        assert sb.is_empty()


class TestRenderPlanMd:
    def test_empty_registry_is_empty_string(self):
        sb = SkillBuilder()
        assert sb.render_for_plan_md([]) == ""

    def test_four_buckets_derived_from_plan_items(self):
        sb = SkillBuilder()
        sb.register(_skill("active-g"), reason="r", plan_version=1)
        sb.register(_skill("idle-g"), reason="r", plan_version=1)
        sb.register(_skill("done-g"), reason="r", plan_version=1)
        sb.register(_skill("demoted-g"), reason="r", plan_version=1)
        sb.record_applied("done-g", plan_version=2, reason="kept")
        sb.mark_unbound("demoted-g", reason="bad fit", plan_version=3)
        plan_items = [
            {"id": "p1", "status": "active", "backing_skill": "active-g"},
            {"id": "p2", "status": "pending", "backing_skill": None},
        ]
        md = sb.render_for_plan_md(plan_items)
        assert "### Active (adopting now)" in md
        assert "active-g" in md
        assert "### Applied (pattern adopted successfully)" in md
        assert "done-g" in md and "applied@v2" in md
        assert "### Selected (candidates, not yet bound)" in md
        assert "idle-g" in md
        assert "### Previously unbound (last-resort tier" in md
        assert "demoted-g" in md and "bad fit" in md
        # The old terminal "Abandoned (tried and failed)" header MUST
        # not appear — the whole concept was removed.
        assert "Abandoned (tried and failed)" not in md

    def test_applied_skill_currently_bound_shows_under_active(self):
        sb = SkillBuilder()
        sb.register(_skill("g"), reason="r", plan_version=1)
        sb.record_applied("g", plan_version=2, reason="kept")
        plan_items = [
            {"id": "p3", "status": "active", "backing_skill": "g"},
        ]
        md = sb.render_for_plan_md(plan_items)
        active_block = md.split("### Applied")[0] if "### Applied" in md else md
        assert "- [>>>] g" in active_block

    def test_unbound_then_applied_renders_under_applied_not_unbound(self):
        """A KEEP on a previously-unbound skill must surface it in the
        Applied bucket (matches tier() precedence)."""
        sb = SkillBuilder()
        sb.register(_skill("comeback"), reason="r", plan_version=1)
        sb.mark_unbound("comeback", reason="bad fit on p1", plan_version=2)
        sb.record_applied("comeback", plan_version=5, reason="KEEP on p3")
        md = sb.render_for_plan_md([])
        assert "### Applied (pattern adopted successfully)" in md
        applied_block = md.split("### Previously unbound")[0]
        assert "comeback" in applied_block
        # And NOT in the Previously-unbound bucket.
        if "### Previously unbound" in md:
            unbound_block = md.split("### Previously unbound")[1]
            assert "comeback" not in unbound_block


class TestShortStatus:
    def test_counts_are_accurate(self):
        sb = SkillBuilder()
        sb.register(_skill("a"), reason="r")
        sb.register(_skill("b"), reason="r")
        sb.register(_skill("c"), reason="r")
        sb.record_applied("a", plan_version=1)
        sb.mark_unbound("c", reason="x", plan_version=1)
        s = sb.format_short_status()
        assert "3 registered" in s
        assert "1 applied" in s
        assert "1 previously unbound" in s

    def test_applied_then_unbound_not_double_counted_as_unbound(self):
        """A skill that was unbound and later KEEP'd counts as
        applied, not as 'previously unbound', in the short status."""
        sb = SkillBuilder()
        sb.register(_skill("g"), reason="r")
        sb.mark_unbound("g", reason="x", plan_version=1)
        sb.record_applied("g", plan_version=2)
        s = sb.format_short_status()
        assert "1 applied" in s
        assert "0 previously unbound" in s

    def test_empty_status(self):
        sb = SkillBuilder()
        assert sb.format_short_status() == ""
