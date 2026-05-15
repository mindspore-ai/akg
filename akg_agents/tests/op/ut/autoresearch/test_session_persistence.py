# Copyright 2025-2026 Huawei Technologies Co., Ltd
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
Regression tests for SessionStore persistence + AgentLoop resume flow.

Two specific bugs being pinned:

1. **skill_state silently dropped on save**. AgentLoop._save_session
   passed ``state["skill_state"]`` into SessionStore.save(), but save()
   never actually wrote the field to session.json. On --resume the
   loader returned no skill_state, the SkillBuilder loaded as empty,
   and the hint advisor's exclude set (applied | abandoned) was empty
   too — meaning the selector could re-pick already-tried skills.

2. **Hint demotion notice overwritten by buffer rebuild**. The
   "hint advisor session could not be restored" message was appended
   to the empty buffer inside the resume detection block, but every
   subsequent buffer-restore branch (force_rebuild / load_latest /
   replace) replaced the buffer wholesale, so the agent never saw it.
"""

import inspect
from dataclasses import dataclass

import pytest

from akg_agents.op.autoresearch.agent.loop import AgentLoop
from akg_agents.op.autoresearch.agent.session import SessionStore


# ---------------------------------------------------------------------------
# Stub config — minimal SessionStore needs
# ---------------------------------------------------------------------------


@dataclass
class _StubAgentCfg:
    session_dir: str = "agent_session"
    heartbeat_file: str = "heartbeat.txt"


@dataclass
class _StubConfig:
    name: str = "stub_op"
    editable_files: tuple = ("kernel.py",)
    eval_script: str = ""
    smoke_test_script: str = ""
    program_file: str = ""
    ref_file: str = ""
    agent: _StubAgentCfg = None

    def __post_init__(self):
        if self.agent is None:
            self.agent = _StubAgentCfg()


# ---------------------------------------------------------------------------
# Finding 1 — skill_state save/load roundtrip
# ---------------------------------------------------------------------------


class _StubGitRepo:
    """Minimal GitRepo stand-in. SessionStore only calls these two methods."""

    def current_commit(self):
        return "stub-head"

    def dirty_files(self, files):
        return []


class TestSkillStatePersistence:
    """Pin that SessionStore actually persists and restores skill_state."""

    def _make_store(self, tmp_path):
        """Build a SessionStore over tmp_path with a stubbed GitRepo.

        Direct constructor injection — no monkeypatching of module-level
        helpers needed (P5/P8 deleted them; SessionStore now takes a
        GitRepo via __init__).
        """
        return SessionStore(
            str(tmp_path), _StubConfig(), git=_StubGitRepo(), verbose=False,
        )

    def test_skill_state_round_trip(self, tmp_path):
        """A non-empty skill_state must survive save → load."""
        store = self._make_store(tmp_path)

        # Mimic the shape SkillBuilder.skill_state_to_dict produces:
        # one applied + one abandoned skill, which is what the hint
        # advisor's exclude set is built from.
        skill_state = {
            "skills": {
                "triton-ascend-matmul": {
                    "name": "triton-ascend-matmul",
                    "status": "applied",
                    "retry": 1,
                    "history": [
                        {"event": "selected", "plan_version": 1},
                        {"event": "applied", "plan_version": 1, "via": "p1"},
                    ],
                },
                "triton-ascend-reduce": {
                    "name": "triton-ascend-reduce",
                    "status": "abandoned",
                    "retry": 0,
                    "history": [
                        {"event": "selected", "plan_version": 1},
                        {"event": "abandoned", "plan_version": 2},
                    ],
                },
            },
        }

        store.save({
            "model": "stub",
            "counters": {"eval_calls_made": 5, "consecutive_failures": 1},
            "baseline_commit": "abc",
            "plan_state": {"items": []},
            "skill_state": skill_state,
        })

        loaded = store.load()
        assert loaded is not None
        assert loaded["skill_state"] == skill_state, (
            "skill_state was silently dropped on save() — hint advisor's "
            "exclude set will be empty after --resume."
        )

    def test_empty_skill_state_load_default(self, tmp_path):
        """A session saved with no skill_state must still load (default {})."""
        store = self._make_store(tmp_path)
        store.save({
            "model": "stub",
            "counters": {},
            "baseline_commit": "abc",
            "plan_state": {"items": []},
            # No skill_state key — represents a run that never used skills.
        })
        loaded = store.load()
        assert loaded is not None
        # Defaulted to empty dict so SkillBuilder.skill_state_from_dict
        # treats it as the no-op restore path.
        assert loaded["skill_state"] == {}

    def test_legacy_session_without_skill_state_loads(self, tmp_path):
        """Pre-fix session.json files (no skill_state key) must still load."""
        store = self._make_store(tmp_path)
        # Hand-write a session.json that mimics the pre-fix layout:
        # has counters but no skill_state key.
        import json
        import os
        session_dir = os.path.join(str(tmp_path), "agent_session")
        os.makedirs(session_dir, exist_ok=True)
        with open(os.path.join(session_dir, "session.json"), "w") as f:
            json.dump({
                "version": 3,
                "task_name": "stub_op",
                "counters": {"eval_calls_made": 3},
                "baseline_commit": "abc",
                "head_commit": "stub-head",
            }, f)
        loaded = store.load()
        assert loaded is not None
        assert loaded["skill_state"] == {}


# ---------------------------------------------------------------------------
# Finding 2 — hint demotion notice survives buffer rebuild
# ---------------------------------------------------------------------------


class TestNoLegacyHintDemotion:
    """The supervisor rewrite removed the hint demotion block from
    ``_run_body``. ``run_supervisor_review`` is one-shot per DISCARD/FAIL, so
    there is no live session to lose across resume — no demotion path
    needed. This test pins that the deleted block does not creep back."""

    def test_run_body_has_no_hint_demotion_branch(self):
        src = inspect.getsource(AgentLoop._run_body)
        for forbidden in (
            "demote_hint_plan",
            "is_hint_plan",
            "hint_session",
            "had an active hint plan",
        ):
            assert forbidden not in src, (
                f"_run_body still contains {forbidden!r} — the supervisor "
                f"rewrite removed all hint demotion logic; do not "
                f"reintroduce it."
            )
