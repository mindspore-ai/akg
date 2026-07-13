# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import sys

SCRIPTS_DIR = Path(__file__).resolve().parents[4] / "workspace_autoresearch" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from phase_machine import (  # noqa: E402
    append_history,
    load_state,
    replay_intent,
    save_state,
    set_task_dir,
    state_transaction,
    write_intent,
)


def _state(task_dir: Path, **extra) -> None:
    state = {
        "phase": "EDIT",
        "progress_initialized": False,
        "expected_history_round": 0,
        "expected_plan_version": 0,
    }
    state.update(extra)
    save_state(str(task_dir), state)


def test_replay_round_rebuilds_pending_settle(tmp_path):
    _state(tmp_path, progress_initialized=True, expected_history_round=2)
    append_history(str(tmp_path), {"round": 3, "decision": "KEEP"})
    write_intent(str(tmp_path), {
        "kind": "round",
        "round": 3,
        "kd_json": {"decision": "KEEP"},
        "progress_fields": {"eval_rounds": 3, "best_metric": 1.25},
    })

    result = replay_intent(str(tmp_path))
    state = load_state(str(tmp_path))

    assert result["action"] == "rebuilt"
    assert state["expected_history_round"] == 3
    assert state["pending_settle"] == {"decision": "KEEP"}
    assert state["eval_rounds"] == 3


def test_replay_baseline_rebuilds_committed_progress(tmp_path):
    _state(tmp_path)
    append_history(str(tmp_path), {"round": 0, "decision": "SEED"})
    write_intent(str(tmp_path), {
        "kind": "baseline",
        "progress_fields": {"task": "toy", "baseline_metric": 2.5},
    })

    result = replay_intent(str(tmp_path))
    state = load_state(str(tmp_path))

    assert result["action"] == "rebuilt"
    assert state["progress_initialized"] is True
    assert state["baseline_metric"] == 2.5


def test_replay_plan_rebuilds_version_and_pid(tmp_path):
    _state(tmp_path, progress_initialized=True, expected_plan_version=1)
    state_dir = tmp_path / ".ar_state"
    (state_dir / "plan.md").write_text("# Plan v2\n", encoding="utf-8")
    write_intent(str(tmp_path), {
        "kind": "plan",
        "version": 2,
        "progress_fields": {"plan_version": 2, "next_pid": 7},
    })

    result = replay_intent(str(tmp_path))
    state = load_state(str(tmp_path))

    assert result["action"] == "rebuilt"
    assert state["expected_plan_version"] == 2
    assert state["plan_version"] == 2
    assert state["next_pid"] == 7


def test_state_transaction_serializes_concurrent_updates(tmp_path):
    _state(tmp_path, counter=0)

    def increment() -> None:
        for _ in range(8):
            with state_transaction(str(tmp_path)):
                state = load_state(str(tmp_path))
                state["counter"] += 1
                save_state(str(tmp_path), state)

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(increment) for _ in range(4)]
        for future in futures:
            future.result()

    assert load_state(str(tmp_path))["counter"] == 32


def test_session_claim_keeps_one_owned_task(monkeypatch, tmp_path):
    import phase_machine.state_store as store

    monkeypatch.setenv("AR_SESSION_ID", "concurrent-session")
    monkeypatch.setattr(store, "_SESSION_TASKS_DIR", str(tmp_path / "sessions"))
    tasks = [tmp_path / "task-a", tmp_path / "task-b"]
    for task in tasks:
        task.mkdir()

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda task: set_task_dir(str(task)), tasks))

    assert results == [True, True]
    owners = [
        (load_state(str(task)) or {}).get("owner")
        for task in tasks
    ]
    assert sum(owner is not None for owner in owners) == 1
    assert Path(store.current_session_task_dir()).resolve() in {
        task.resolve() for task in tasks
    }
