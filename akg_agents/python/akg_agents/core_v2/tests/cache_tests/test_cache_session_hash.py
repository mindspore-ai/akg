import pytest
from types import SimpleNamespace

from akg_agents.op.langgraph_op.task import LangGraphTask


def _build_dummy_task(**overrides):
    base = {
        "workflow_name": "default_workflow",
        "task_id": "task-0",
        "task_desc": "build relu kernel",
        "op_name": "relu",
        "dsl": "triton_cuda",
        "backend": "cuda",
        "arch": "a100",
        "framework": "torch",
        "task_type": "precision_only",
        "source_backend": None,
        "source_arch": None,
        "user_requirements": "",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_generate_cache_session_hash_is_deterministic():
    task_like = _build_dummy_task()
    h1 = LangGraphTask._generate_cache_session_hash(task_like)
    h2 = LangGraphTask._generate_cache_session_hash(task_like)
    assert h1 == h2


def test_generate_cache_session_hash_changes_with_identity_fields():
    task_a = _build_dummy_task(dsl="triton_cuda")
    task_b = _build_dummy_task(dsl="triton_ascend")

    hash_a = LangGraphTask._generate_cache_session_hash(task_a)
    hash_b = LangGraphTask._generate_cache_session_hash(task_b)

    assert hash_a != hash_b


def _build_task_like_for_prepare_initial_state(config):
    return SimpleNamespace(
        task_id="task-0",
        op_name="relu",
        task_desc="build relu kernel",
        backend="cuda",
        arch="a100",
        dsl="triton_cuda",
        framework="torch",
        task_type="precision_only",
        inspirations=None,
        meta_prompts=None,
        handwrite_suggestions=[],
        user_requirements="",
        workflow_name="default_workflow",
        source_backend=None,
        source_arch=None,
        config=config,
        agents={},
        _generate_cache_session_hash=lambda: "generated-session-hash",
    )


def test_prepare_initial_state_replay_requires_cache_session_hash():
    task_like = _build_task_like_for_prepare_initial_state(
        {
            "task_label": "cache-test",
            "cache_mode": "replay",
            "cache_session_hash": "",
        }
    )

    with pytest.raises(ValueError, match="cache_session_hash is required when cache_mode is replay"):
        LangGraphTask._prepare_initial_state(task_like, None)


def test_prepare_initial_state_record_auto_generates_cache_session_hash():
    task_like = _build_task_like_for_prepare_initial_state(
        {
            "task_label": "cache-test",
            "cache_mode": "record",
            "cache_session_hash": "",
        }
    )

    state = LangGraphTask._prepare_initial_state(task_like, None)

    assert state["cache_mode"] == "record"
    assert state["cache_session_hash"] == "generated-session-hash"
