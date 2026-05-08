"""parse_step_durations_us 的纯函数验证：用内存 sqlite 模拟 msprof export-db。

关键契约：step 边界用 `AKG_l2cache_clear` launch 切分。
batch_profiler 在每个被测 launch 之前主动跑一次 L2 clear，trace 里的 L2 clear
出现位置就是逻辑 step 边界。
"""

from __future__ import annotations

import sqlite3

import pytest

from akg_agents.op.dynamic_tune.measure.parse_export_db import (
    ExportDbParseError,
    parse_step_durations_us,
)


def _build_api_event(rows):
    """rows: [(struct_type, id, item_id, start, end, connection_id), ...]"""

    conn = sqlite3.connect(":memory:")
    cur = conn.cursor()
    cur.execute(
        "CREATE TABLE ApiData ("
        " struct_type TEXT, id TEXT, item_id TEXT,"
        " start INTEGER, end INTEGER, connection_id INTEGER)"
    )
    for row in rows:
        cur.execute(
            "INSERT INTO ApiData (struct_type, id, item_id, start, end, connection_id)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            row,
        )
    conn.commit()
    return conn


def _build_ascend_task(rows):
    """rows: [(connection_id, duration_ns, start_time), ...]"""

    conn = sqlite3.connect(":memory:")
    cur = conn.cursor()
    cur.execute(
        "CREATE TABLE AscendTask ("
        " connection_id INTEGER, duration INTEGER, start_time INTEGER)"
    )
    for row in rows:
        cur.execute(
            "INSERT INTO AscendTask (connection_id, duration, start_time)"
            " VALUES (?, ?, ?)",
            row,
        )
    conn.commit()
    return conn


def test_parse_basic_l2_clear_marks_step_boundary():
    """模拟标准 batch_profiler 流程：3 个 step（warmup + 2 active）。"""

    api = _build_api_event([
        ("launch", "AKG_l2cache_clear", "", 0, 5, 100),
        ("launch", "kernel_a", "k_a", 6, 10, 1),
        ("launch", "AKG_l2cache_clear", "", 100, 105, 100),
        ("launch", "kernel_a", "k_a", 106, 115, 2),
        ("launch", "kernel_b", "k_b", 116, 120, 3),
        ("launch", "AKG_l2cache_clear", "", 200, 205, 100),
        ("launch", "kernel_a", "k_a", 206, 215, 4),
    ])
    task = _build_ascend_task([
        (1, 1_000, 6),
        (2, 3_000, 106),
        (3, 2_000, 116),
        (4, 4_000, 206),
        (100, 9_999, 0),  # L2 clear 自身的 duration 不应被聚合到 step 上
    ])
    try:
        out = parse_step_durations_us(
            api_event_connection=api, ascend_task_connection=task
        )
    finally:
        api.close()
        task.close()
    assert out.warmup_and_active_us == (1.0, 5.0, 4.0)
    assert out.launch_groups == ((1,), (2, 3), (4,))


def test_parse_ignores_sync_events_and_other_apis():
    """non-launch 事件（sync / aclrt* / 等）必须被无视。"""

    api = _build_api_event([
        ("launch", "AKG_l2cache_clear", "", 0, 5, 100),
        ("api", "aclrtSynchronizeStream", "", 6, 7, 0),
        ("launch", "kernel_a", "", 8, 10, 1),
        ("api", "aclrtSynchronizeDevice", "", 11, 12, 0),
        ("launch", "AKG_l2cache_clear", "", 100, 105, 100),
        ("api", "aclrtSynchronizeStream", "", 106, 107, 0),
        ("launch", "kernel_a", "", 108, 110, 2),
    ])
    task = _build_ascend_task([
        (1, 2_000, 8),
        (2, 5_000, 108),
    ])
    try:
        out = parse_step_durations_us(
            api_event_connection=api, ascend_task_connection=task
        )
    finally:
        api.close()
        task.close()
    assert out.warmup_and_active_us == (2.0, 5.0)


def test_parse_handles_multiple_launches_per_step_no_sync_dependency():
    """同一 step 内多次 launch + 多次 sync 时，按 L2 clear 切分得到一个 group。"""

    api = _build_api_event([
        ("launch", "AKG_l2cache_clear", "", 0, 5, 100),
        ("launch", "kernel_x", "", 10, 20, 1),
        ("api", "aclrtSynchronizeStream", "", 21, 22, 0),
        ("launch", "kernel_y", "", 23, 30, 2),
        ("api", "aclrtSynchronizeStream", "", 31, 32, 0),
        ("launch", "kernel_z", "", 33, 40, 3),
        ("api", "aclrtSynchronizeDevice", "", 41, 42, 0),
        ("launch", "AKG_l2cache_clear", "", 100, 105, 100),
        ("launch", "kernel_x", "", 106, 110, 4),
        ("api", "aclrtSynchronizeStream", "", 111, 112, 0),
    ])
    task = _build_ascend_task([
        (1, 1_000, 10),
        (2, 2_000, 23),
        (3, 3_000, 33),
        (4, 4_000, 106),
    ])
    try:
        out = parse_step_durations_us(
            api_event_connection=api, ascend_task_connection=task
        )
    finally:
        api.close()
        task.close()
    # 按 L2 clear 切：只切出 2 个 step，第 1 个 step 包含 3 个 launch，duration 累加
    assert out.launch_groups == ((1, 2, 3), (4,))
    assert out.warmup_and_active_us == (6.0, 4.0)


def test_parse_skips_negative_connection_in_ascend_task():
    api = _build_api_event([
        ("launch", "AKG_l2cache_clear", "", 0, 5, 100),
        ("launch", "kernel_a", "", 6, 10, 1),
    ])
    task = _build_ascend_task([
        (1, 1_500, 6),
        (-1, 9_999_999, 50),  # 后台任务，应被跳过
    ])
    try:
        out = parse_step_durations_us(
            api_event_connection=api, ascend_task_connection=task
        )
    finally:
        api.close()
        task.close()
    assert out.warmup_and_active_us == (1.5,)


def test_parse_raises_when_no_l2_clear_marker():
    """没有 L2 clear marker 时必须明确报错——指引 caller 打开
    NpuProfilerBackend.enable_l2_cache_clear。"""

    api = _build_api_event([
        ("launch", "kernel_a", "", 10, 20, 1),
        ("launch", "kernel_a", "", 30, 40, 2),
    ])
    task = _build_ascend_task([
        (1, 1_000, 10),
        (2, 2_000, 30),
    ])
    try:
        with pytest.raises(ExportDbParseError, match="l2_clear_marker_missing"):
            parse_step_durations_us(
                api_event_connection=api, ascend_task_connection=task
            )
    finally:
        api.close()
        task.close()


def test_parse_raises_on_missing_connection_in_task():
    api = _build_api_event([
        ("launch", "AKG_l2cache_clear", "", 0, 5, 100),
        ("launch", "kernel_a", "", 10, 20, 1),
        ("launch", "AKG_l2cache_clear", "", 100, 105, 100),
        ("launch", "kernel_b", "", 121, 130, 2),
        ("launch", "kernel_c", "", 131, 140, 3),
    ])
    task = _build_ascend_task([
        (1, 1_000, 10),
        (2, 2_000, 121),
        # connection_id=3 缺 duration 记录；非 warmup group 仍应报错
    ])
    try:
        with pytest.raises(ExportDbParseError, match="missing_connection_duration"):
            parse_step_durations_us(
                api_event_connection=api, ascend_task_connection=task
            )
    finally:
        api.close()
        task.close()


def test_parse_raises_on_negative_launch_connection():
    api = _build_api_event([
        ("launch", "AKG_l2cache_clear", "", 0, 5, 100),
        ("launch", "kernel_a", "", 10, 20, -7),
    ])
    task = _build_ascend_task([(1, 1_000, 10)])
    try:
        with pytest.raises(ExportDbParseError, match="launch_connection_negative"):
            parse_step_durations_us(
                api_event_connection=api, ascend_task_connection=task
            )
    finally:
        api.close()
        task.close()


def test_parse_raises_on_empty_api():
    api = _build_api_event([])
    task = _build_ascend_task([(1, 1, 0)])
    try:
        with pytest.raises(ExportDbParseError, match="api_event_empty"):
            parse_step_durations_us(
                api_event_connection=api, ascend_task_connection=task
            )
    finally:
        api.close()
        task.close()
