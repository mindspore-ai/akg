from __future__ import annotations

from akg_agents.op.dynamic_tune.measure.parse_export_db import parse_step_durations_us

from .test_parse_export_db import _build_api_event, _build_ascend_task


def test_parse_skips_warmup_group_when_connection_durations_missing():
    api = _build_api_event([
        ("launch", "AKG_l2cache_clear", "", 0, 5, 100),
        ("launch", "kernel_a", "", 6, 10, 1),
        ("launch", "AKG_l2cache_clear", "", 100, 105, 100),
        ("launch", "kernel_b", "", 106, 110, 2),
    ])
    task = _build_ascend_task([
        (1, -1, -1),
        (2, 3_000, 106),
    ])
    try:
        out = parse_step_durations_us(
            api_event_connection=api, ascend_task_connection=task
        )
    finally:
        api.close()
        task.close()

    assert out.warmup_and_active_us == (0.0, 3.0)
    assert out.launch_groups == ((1,), (2,))
    assert "warmup_skipped_missing=1" in out.note
