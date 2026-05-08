from __future__ import annotations

from akg_agents.op.dynamic_tune.measure.parse_export_db import parse_step_durations_us

from .test_parse_export_db import _build_api_event, _build_ascend_task


def test_parse_skips_negative_duration_rows_when_positive_row_exists():
    api = _build_api_event([
        ("launch", "AKG_l2cache_clear", "", 0, 5, 100),
        ("launch", "kernel_a", "", 6, 10, 1),
        ("launch", "AKG_l2cache_clear", "", 100, 105, 100),
        ("launch", "kernel_b", "", 106, 110, 2),
    ])
    task = _build_ascend_task([
        (1, -1, 6),
        (1, 2_000, 7),
        (2, 3_000, 106),
    ])
    try:
        out = parse_step_durations_us(
            api_event_connection=api, ascend_task_connection=task
        )
    finally:
        api.close()
        task.close()

    assert out.warmup_and_active_us == (2.0, 3.0)
