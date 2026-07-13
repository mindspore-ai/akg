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

from contextlib import contextmanager
from pathlib import Path
import subprocess
import sys

import pytest
import yaml


SCRIPTS_DIR = Path(__file__).resolve().parents[4] / "workspace_autoresearch" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import phase_machine.state_store as state_store  # noqa: E402
import scaffold as scaffold_module  # noqa: E402
import task_handle as task_module  # noqa: E402
from decide import AgentEvent, decide  # noqa: E402
from phase_machine import (  # noqa: E402
    BASELINE, DIAGNOSE, EDIT, FINISH, INIT, PLAN, REPLAN,
    load_progress, load_state, save_state, task_summary, write_intent,
)
from phase_machine.phase_policy import classify  # noqa: E402
from task_handle import (  # noqa: E402
    Role, TaskCorrupted, TaskPhaseError, open_task,
)
from workflow.baseline import (  # noqa: E402
    BaselinePrecheckOutcome, run_baseline_init,
)
from workflow.transition import phase_after_round, phase_on_resume  # noqa: E402


def test_scaffold_activates_before_running_initial_baseline(
        tmp_path, monkeypatch):
    task_dir = tmp_path / "fresh"
    (task_dir / ".ar_state").mkdir(parents=True)
    observed = []

    def fake_run(command):
        observed.append((command, state_store.read_phase(str(task_dir))))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(scaffold_module.subprocess, "run", fake_run)

    assert scaffold_module._run_initial_baseline(str(task_dir)) == 0
    assert observed[0][1] == BASELINE
    assert observed[0][0][-1] == str(task_dir)


def test_headless_skill_discovery_find_is_read_only_but_actions_are_not():
    command = (
        "find /tmp/skills -name SKILL.md | "
        "grep -iE 'basics|hardware|affinity'"
    )
    assert classify(command).klass == "READONLY"
    assert classify("find /tmp/skills -delete").klass == "OTHER"
    assert classify(
        "find /tmp/skills -exec rm -rf '{}' ';'"
    ).klass == "OTHER"
    assert classify(
        "find /tmp/skills -fprint /tmp/skill-list"
    ).klass == "OTHER"


def _state(task_dir: Path, phase: str, **extra) -> None:
    state = {
        "phase": phase,
        "owner": None,
        "progress_initialized": True,
        "pending_settle": None,
        "expected_history_round": 0,
        "expected_plan_version": 0,
        "task": "toy",
        "eval_rounds": 0,
        "max_rounds": 5,
        "consecutive_failures": 0,
        "best_metric": 10.0,
        "baseline_metric": 20.0,
        "baseline_outcome": "ok",
        "seed_metric": 10.0,
        "plan_version": 0,
        "next_pid": 1,
    }
    state.update(extra)
    save_state(str(task_dir), state)


def test_task_summary_exposes_recorded_best_speedup(tmp_path):
    _state(tmp_path, EDIT, best_speedup=1.25)

    summary = task_summary(str(tmp_path))

    assert summary is not None
    assert summary["best_speedup"] == 1.25


def _plan(task_dir: Path, version: int, items: list[tuple[str, bool]]) -> None:
    lines = [
        f"# Plan v{version}", "", "## Active Items",
    ]
    for index, (pid, done) in enumerate(items):
        if done:
            lines.append(f"- [x] **{pid}** [KEEP, metric=1.0]: done")
        else:
            active = " (ACTIVE)" if index == 0 else ""
            lines.append(f"- [ ] **{pid}**{active}: optimize")
            lines.append("  - rationale: enough detail for a test plan")
    lines.extend([
        "", "## Settled History",
        "| Item | Outcome | Metric | Reason |",
        "|------|---------|--------|--------|",
    ])
    path = task_dir / ".ar_state" / "plan.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@contextmanager
def _agent_task(task_dir: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AR_SESSION_ID", f"test-{task_dir.name}")
    monkeypatch.setattr(
        state_store, "_SESSION_TASKS_DIR", str(task_dir.parent / "session-index"))
    with open_task(str(task_dir), role=Role.AGENT, force=True) as task:
        yield task
        task.release(force=True)


def test_round_transition_matrix(tmp_path, monkeypatch):
    monkeypatch.setenv("AKG_AR_CONSECUTIVE_FAIL_THRESHOLD", "3")
    task_dir = tmp_path / "task"
    task_dir.mkdir()

    _state(task_dir, EDIT, eval_rounds=5, max_rounds=5)
    assert phase_after_round(str(task_dir)) == FINISH

    _state(task_dir, EDIT, consecutive_failures=3)
    assert phase_after_round(str(task_dir)) == DIAGNOSE

    _state(task_dir, EDIT, expected_plan_version=1, plan_version=1)
    _plan(task_dir, 1, [("p1", False)])
    assert phase_after_round(str(task_dir)) == EDIT

    _plan(task_dir, 1, [("p1", True)])
    assert phase_after_round(str(task_dir)) == REPLAN

    _state(task_dir, EDIT, progress_initialized=False)
    with pytest.raises(RuntimeError):
        phase_after_round(str(task_dir))


def test_resume_transition_matrix(tmp_path):
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    _state(task_dir, BASELINE, progress_initialized=False)
    assert phase_on_resume(str(task_dir)) == BASELINE

    _state(task_dir, EDIT, eval_rounds=5, max_rounds=5)
    assert phase_on_resume(str(task_dir)) == FINISH

    _state(task_dir, PLAN, seed_metric=None)
    assert phase_on_resume(str(task_dir)) == PLAN

    _state(task_dir, PLAN, expected_plan_version=1, plan_version=1)
    _plan(task_dir, 1, [("p1", False)])
    assert phase_on_resume(str(task_dir)) == EDIT

    _plan(task_dir, 1, [("p1", True)])
    assert phase_on_resume(str(task_dir)) == REPLAN


@pytest.mark.parametrize(
    ("phase", "pending", "expected"),
    [
        (INIT, None, BASELINE),
        (DIAGNOSE, {"round": 2}, EDIT),
    ],
)
def test_activation_normalizes_interrupted_control_state(
        tmp_path, monkeypatch, phase, pending, expected):
    task_dir = tmp_path / phase.lower()
    task_dir.mkdir()
    _state(
        task_dir, phase,
        progress_initialized=phase != INIT,
        pending_settle=pending,
    )

    with _agent_task(task_dir, monkeypatch) as task:
        assert task.activate(fresh=False) == expected

    assert load_state(str(task_dir))["phase"] == expected


@pytest.mark.parametrize("source_phase", [PLAN, REPLAN, DIAGNOSE])
def test_plan_commit_atomically_enters_edit(tmp_path, monkeypatch, source_phase):
    task_dir = tmp_path / source_phase.lower()
    task_dir.mkdir()
    _state(task_dir, source_phase, consecutive_failures=3)

    with _agent_task(task_dir, monkeypatch) as task:
        result = task.commit_plan([
            {"desc": "try a new tiling", "rationale": "test", "skill": ""},
        ])

    state = load_state(str(task_dir))
    assert result["version"] == 1
    assert state["phase"] == EDIT
    assert state["pending_settle"] is None
    assert state["expected_plan_version"] == 1
    assert state["consecutive_failures"] == (0 if source_phase == DIAGNOSE else 3)


def test_plan_commit_clears_edit_recovery_in_same_save(tmp_path, monkeypatch):
    task_dir = tmp_path / "recovery"
    task_dir.mkdir()
    _state(task_dir, EDIT, pending_settle={"round": 1})

    with _agent_task(task_dir, monkeypatch) as task:
        task.commit_plan([
            {"desc": "replace broken plan", "rationale": "test", "skill": ""},
        ])

    state = load_state(str(task_dir))
    assert state["phase"] == EDIT
    assert state["pending_settle"] is None


def test_plan_commit_rejects_wrong_phase_before_body_write(tmp_path, monkeypatch):
    task_dir = tmp_path / "baseline"
    task_dir.mkdir()
    _state(task_dir, BASELINE)

    with _agent_task(task_dir, monkeypatch) as task:
        with pytest.raises(TaskPhaseError):
            task.commit_plan([
                {"desc": "illegal plan", "rationale": "test", "skill": ""},
            ])

    assert not (task_dir / ".ar_state" / "plan.md").exists()


def test_settle_commits_phase_and_sentinel_together(tmp_path, monkeypatch):
    task_dir = tmp_path / "settle"
    task_dir.mkdir()
    result = {
        "decision": "DISCARD", "round_metric": 11.0,
        "plan_item": "p1", "round": 1,
    }
    _state(
        task_dir, EDIT, expected_plan_version=1, plan_version=1,
        eval_rounds=1, pending_settle=result,
    )
    _plan(task_dir, 1, [("p1", False), ("p2", False)])

    with _agent_task(task_dir, monkeypatch) as task:
        settled = task.settle_round()

    state = load_state(str(task_dir))
    assert settled["settled_item"] == "p1"
    assert state["phase"] == EDIT
    assert state["pending_settle"] is None


def test_settle_replays_after_plan_body_precedes_state_commit(
        tmp_path, monkeypatch):
    task_dir = tmp_path / "settle-replay"
    task_dir.mkdir()
    result = {
        "decision": "DISCARD", "round_metric": 11.0,
        "plan_item": "p1", "round": 1,
    }
    _state(
        task_dir, EDIT, expected_plan_version=1, plan_version=1,
        eval_rounds=1, pending_settle=result,
    )
    _plan(task_dir, 1, [("p1", False), ("p2", False)])

    real_save = task_module.save_state
    monkeypatch.setattr(
        task_module, "save_state",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk full")),
    )
    with pytest.raises(TaskCorrupted):
        with _agent_task(task_dir, monkeypatch) as task:
            task.settle_round()

    state = load_state(str(task_dir))
    assert state["phase"] == EDIT
    assert state["pending_settle"] == result

    monkeypatch.setattr(task_module, "save_state", real_save)
    with _agent_task(task_dir, monkeypatch) as task:
        settled = task.settle_round()

    state = load_state(str(task_dir))
    assert settled["settled_item"] == "p1"
    assert state["phase"] == EDIT
    assert state["pending_settle"] is None


def _write_task_config(task_dir: Path) -> None:
    (task_dir / "task.yaml").write_text(
        yaml.safe_dump({
            "name": "toy",
            "editable_files": ["kernel.py"],
            "metric": {
                "primary": "latency_us",
                "lower_is_better": True,
                "improvement_threshold": 0.0,
            },
            "agent": {"max_rounds": 3},
        }, sort_keys=False),
        encoding="utf-8",
    )


def test_post_baseline_reports_committed_plan_phase(tmp_path):
    task_dir = tmp_path / "post-baseline"
    task_dir.mkdir()
    _write_task_config(task_dir)
    _state(task_dir, PLAN, baseline_outcome="ok", seed_metric=10.0)

    decision = decide(AgentEvent(
        kind="post_tool",
        tool_kind="shell",
        command=f'python scripts/engine/baseline.py "{task_dir}"',
    ))

    assert any("Baseline complete. Phase -> PLAN" in line
               for line in decision.status)


def test_incomplete_ok_baseline_remains_uncommitted(tmp_path):
    task_dir = tmp_path / "baseline-incomplete"
    task_dir.mkdir()
    (task_dir / "kernel.py").write_text("x = 1\n", encoding="utf-8")
    _write_task_config(task_dir)
    subprocess.run(["git", "init", "-q"], cwd=task_dir, check=True)
    _state(task_dir, BASELINE, progress_initialized=False)

    rc = run_baseline_init(str(task_dir), {
        "outcome": "ok",
        "correctness": True,
        "metrics": {"ref_latency_us": 20.0},
    })

    assert rc == 2
    assert load_state(str(task_dir))["phase"] == BASELINE
    assert load_progress(str(task_dir)) is None
    assert not (task_dir / ".ar_state" / "history.jsonl").exists()


def test_committed_baseline_retry_is_idempotent_in_plan(
        tmp_path, monkeypatch):
    task_dir = tmp_path / "baseline-retry"
    task_dir.mkdir()
    _state(task_dir, PLAN, expected_history_round=0)
    history = task_dir / ".ar_state" / "history.jsonl"
    history.write_text('{"round": 0, "decision": "SEED"}\n', encoding="utf-8")

    with _agent_task(task_dir, monkeypatch) as task:
        assert task.baseline_preflight().outcome == \
            BaselinePrecheckOutcome.ALREADY_DONE


def test_replay_restores_transaction_phase_and_sentinel(tmp_path):
    task_dir = tmp_path / "replay"
    task_dir.mkdir()
    _state(task_dir, DIAGNOSE, expected_plan_version=1, plan_version=1,
           pending_settle={"round": 2})
    _plan(task_dir, 2, [("p2", False)])
    write_intent(str(task_dir), {
        "kind": "plan",
        "version": 2,
        "state_patch": {
            "plan_version": 2,
            "next_pid": 3,
            "expected_plan_version": 2,
            "phase": EDIT,
            "pending_settle": None,
        },
    })

    state_store.replay_intent(str(task_dir))
    state = load_state(str(task_dir))
    assert state["phase"] == EDIT
    assert state["pending_settle"] is None
    assert state["plan_version"] == 2
