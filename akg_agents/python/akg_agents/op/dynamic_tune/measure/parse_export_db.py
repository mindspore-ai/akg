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

"""自定义解析 msprof export-db 的双表（api_event.db + ascend_task.db）。

设计思想：

torch_npu 的 `--export=db` 实际上会在 prof_root 下产出多个 .db：
    - api_event.db    : ApiData(struct_type, id, item_id, start, end, connection_id)
    - ascend_task.db  : AscendTask(connection_id, duration, start_time)
    - 还有一些 op_summary.db / step_trace.db 等辅助文件，本模块都不依赖。

**step 边界用 `AKG_l2cache_clear` launch 切分**，不依赖 sync 次数：
    被测 kernel 内部 / wrapper 里可能存在多次 sync 或多次 launch 的情况，
    sync 数量与逻辑 step 不一一对应。L2 clear 是我们自己在每个 step 之前
    主动跑的、命名稳定的 marker kernel，用它切才与 batch_profiler 的 step
    流程严格对齐。

因此 caller 必须保证 `BatchProfiler` 配置 `enable_l2_cache_clear=True`（默认
即如此）；关掉之后本解析层会因为找不到任何 L2 clear 边界而抛错。

每个 launch group 的 latency = 该 group 内所有 connection_id 在 AscendTask
里 duration（ns）累计 / 1000。

特殊处理：第一个 group 是 profiler warmup。实测某些 case 在 warmup 上会只产出
host 侧 launch 事件，而 device 侧 AscendTask 只留下 `duration=-1,start_time=-1`
占位行。此时 warmup latency 直接记为 0 并跳过，不影响后续 active steps。

为了让纯逻辑测试能脱离真实 NPU 环境，关键解析函数 `parse_step_durations_us`
接受两个已打开的 sqlite3.Connection——测试可以注入内存库验证。
"""

from __future__ import annotations

import math
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

DEFAULT_EXPORT_DB_FILE_TIMEOUT_SECONDS = 30.0
_POLL_INTERVAL_SECONDS = 0.2

API_EVENT_DB_NAME = "api_event.db"
ASCEND_TASK_DB_NAME = "ascend_task.db"


@dataclass(frozen=True)
class ExportDbStepDurations:
    """解析后的 step 时长。

    每个 step 一个 latency；launch_groups[i] 是该 step 内 launch 的 connection_id
    列表，便于诊断（比如哪个 step 没有任何 launch、被 cache clear 污染等）。
    """

    warmup_and_active_us: tuple[float, ...]
    launch_groups: tuple[tuple[int, ...], ...]
    api_event_db: Path
    ascend_task_db: Path
    note: str = ""


@dataclass(frozen=True)
class ExportDbDiagnostics:
    """解析失败时回填的诊断信息，方便排错。"""

    prof_root: str
    api_event_db: str = ""
    ascend_task_db: str = ""
    extra: str = ""


class ExportDbNotFoundError(RuntimeError):
    pass


class ExportDbParseError(RuntimeError):
    def __init__(self, message: str, diagnostics: ExportDbDiagnostics) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics


def find_unique_file(
    prof_root: str | Path,
    name: str,
    *,
    db_file_timeout_seconds: float = DEFAULT_EXPORT_DB_FILE_TIMEOUT_SECONDS,
) -> Path:
    """在 prof_root 树下精确找到一个文件名为 `name` 的文件。"""

    prof_root_path = Path(prof_root)
    deadline = time.monotonic() + max(0.0, float(db_file_timeout_seconds))
    while True:
        candidates = sorted(prof_root_path.rglob(name))
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            raise ExportDbParseError(
                f"export_db_multiple:{name}:count={len(candidates)}",
                ExportDbDiagnostics(
                    prof_root=str(prof_root_path),
                    extra=";".join(str(p) for p in candidates),
                ),
            )
        if time.monotonic() >= deadline:
            raise ExportDbNotFoundError(
                f"export_db_missing:{name}:dir={prof_root_path}:waited={db_file_timeout_seconds}s"
            )
        time.sleep(_POLL_INTERVAL_SECONDS)


def _is_launch_event(struct_type: object) -> bool:
    return str(struct_type or "").strip().lower() == "launch"


def _is_l2_cache_clear_launch(api_name: object, item_name: object) -> bool:
    target = "akg_l2cache_clear"
    return (
        target in str(api_name or "").casefold()
        or target in str(item_name or "").casefold()
    )


def _load_launch_groups_from_api_event(
    connection: sqlite3.Connection,
) -> list[tuple[int, ...]]:
    """从 api_event.db 切出 launch_groups（按 L2 cache clear 边界）。

    - 只看 struct_type=='launch' 的事件，其它（包括 sync）一律忽略。
    - 遇到 `AKG_l2cache_clear` launch：把当前 group 关闭、开一段新 group；
      L2 clear 自身不进入任何 group。
    - 其它 launch 按顺序追加到当前 group。
    - connection_id 为负的 launch 直接报错（trace 损坏，无法对应 duration）。

    注：如果整段 trace 完全没有 L2 clear，则会拿到一个聚合所有 launch 的大
    group，下游会因为 step 数对不上 active_steps+1 报错——这通常意味着
    NpuProfilerBackend 关掉了 enable_l2_cache_clear，需要打开。
    """

    cursor = connection.cursor()
    rows = cursor.execute(
        """
        SELECT struct_type, id, item_id, start, end, connection_id
        FROM ApiData
        ORDER BY start ASC, end ASC
        """
    ).fetchall()
    if not rows:
        raise ExportDbParseError(
            "export_db_api_event_empty",
            ExportDbDiagnostics(prof_root=""),
        )

    groups: list[tuple[int, ...]] = []
    current: list[int] = []
    saw_l2_clear = False
    for struct_type, api_name, item_name, _start, _end, connection_id in rows:
        if not _is_launch_event(struct_type):
            continue
        if _is_l2_cache_clear_launch(api_name, item_name):
            saw_l2_clear = True
            if current:
                groups.append(tuple(dict.fromkeys(current)))
                current = []
            continue
        try:
            cid = int(connection_id)
        except (TypeError, ValueError) as exc:
            raise ExportDbParseError(
                f"export_db_launch_connection_invalid:{connection_id}",
                ExportDbDiagnostics(prof_root=""),
            ) from exc
        if cid < 0:
            raise ExportDbParseError(
                f"export_db_launch_connection_negative:{cid}",
                ExportDbDiagnostics(prof_root=""),
            )
        current.append(cid)

    if current:
        groups.append(tuple(dict.fromkeys(current)))

    if not saw_l2_clear:
        raise ExportDbParseError(
            "export_db_l2_clear_marker_missing:开启 NpuProfilerBackend.enable_l2_cache_clear",
            ExportDbDiagnostics(prof_root=""),
        )
    if not groups:
        raise ExportDbParseError(
            "export_db_no_launch_groups", ExportDbDiagnostics(prof_root="")
        )
    return groups


def _load_connection_durations_us(
    connection: sqlite3.Connection,
) -> dict[int, float]:
    """从 ascend_task.db 加载 connection_id → duration_us。

    - duration 单位 ns，需 /1000 转 us；
    - connection_id<0 的后台任务（如 fused 残留）跳过；
    - 同一 connection_id 出现多条记录时累加，以匹配 launch_group 的累计语义；
    - msprof export-db 偶发会产出 duration<0 的占位 task，先跳过；如果某个
      connection 最终没有任何正时长记录，下游会在聚合 step 时以
      `missing_connection_duration` 报错（warmup group 例外）。
    """

    cursor = connection.cursor()
    rows = cursor.execute(
        """
        SELECT connection_id, duration
        FROM AscendTask
        ORDER BY start_time ASC
        """
    ).fetchall()
    if not rows:
        raise ExportDbParseError(
            "export_db_ascend_task_empty", ExportDbDiagnostics(prof_root="")
        )

    out: dict[int, float] = {}
    for connection_id, duration_ns in rows:
        try:
            cid = int(connection_id)
            dur_ns = float(duration_ns)
        except (TypeError, ValueError) as exc:
            raise ExportDbParseError(
                f"export_db_task_duration_invalid:{connection_id}:{duration_ns}",
                ExportDbDiagnostics(prof_root=""),
            ) from exc
        if cid < 0:
            continue
        if not math.isfinite(dur_ns):
            raise ExportDbParseError(
                f"export_db_task_duration_invalid:{cid}:{dur_ns}",
                ExportDbDiagnostics(prof_root=""),
            )
        if dur_ns < 0:
            continue
        out[cid] = out.get(cid, 0.0) + dur_ns / 1000.0
    return out


def parse_step_durations_us(
    *,
    api_event_connection: sqlite3.Connection,
    ascend_task_connection: sqlite3.Connection,
    api_event_db: Path | str = "",
    ascend_task_db: Path | str = "",
) -> ExportDbStepDurations:
    """纯函数版本：接受两个 sqlite 连接，返回 step latencies。

    单元测试可以构造内存 sqlite 直接调本函数，不需要真实文件。
    """

    launch_groups = _load_launch_groups_from_api_event(api_event_connection)
    connection_durations_us = _load_connection_durations_us(ascend_task_connection)
    warmup_and_active_us: list[float] = []
    warmup_missing_cids: tuple[int, ...] = ()
    for group_index, group in enumerate(launch_groups):
        if not group:
            raise ExportDbParseError(
                "export_db_empty_launch_group", ExportDbDiagnostics(prof_root="")
            )
        missing_cids = tuple(
            cid for cid in group if cid not in connection_durations_us
        )
        if missing_cids:
            if group_index == 0:
                warmup_missing_cids = missing_cids
                warmup_and_active_us.append(0.0)
                continue
            missing_cid = missing_cids[0]
            raise ExportDbParseError(
                f"export_db_missing_connection_duration:{missing_cid}:"
                f"available={len(connection_durations_us)}",
                ExportDbDiagnostics(prof_root=""),
            )
        latency_us = 0.0
        for cid in group:
            if cid not in connection_durations_us:
                raise ExportDbParseError(
                    f"export_db_missing_connection_duration:{cid}:"
                    f"available={len(connection_durations_us)}",
                    ExportDbDiagnostics(prof_root=""),
                )
            latency_us += connection_durations_us[cid]
        warmup_and_active_us.append(latency_us)
    note = (
        f"groups={len(launch_groups)};durations_indexed={len(connection_durations_us)}"
    )
    if warmup_missing_cids:
        note += (
            ";warmup_skipped_missing="
            + ",".join(str(cid) for cid in warmup_missing_cids)
        )
    return ExportDbStepDurations(
        warmup_and_active_us=tuple(warmup_and_active_us),
        launch_groups=tuple(launch_groups),
        api_event_db=Path(str(api_event_db)),
        ascend_task_db=Path(str(ascend_task_db)),
        note=note,
    )


def parse_export_db_dir(
    prof_root: str | Path,
    *,
    db_file_timeout_seconds: float = DEFAULT_EXPORT_DB_FILE_TIMEOUT_SECONDS,
) -> ExportDbStepDurations:
    """高层入口：找两个 .db、连接、调 parse_step_durations_us。"""

    prof_root_path = Path(prof_root)
    api_event_db = find_unique_file(
        prof_root_path,
        API_EVENT_DB_NAME,
        db_file_timeout_seconds=db_file_timeout_seconds,
    )
    ascend_task_db = find_unique_file(
        prof_root_path,
        ASCEND_TASK_DB_NAME,
        db_file_timeout_seconds=db_file_timeout_seconds,
    )
    api_event_connection = sqlite3.connect(
        f"file:{api_event_db}?mode=ro", uri=True
    )
    ascend_task_connection = sqlite3.connect(
        f"file:{ascend_task_db}?mode=ro", uri=True
    )
    try:
        return parse_step_durations_us(
            api_event_connection=api_event_connection,
            ascend_task_connection=ascend_task_connection,
            api_event_db=api_event_db,
            ascend_task_db=ascend_task_db,
        )
    finally:
        api_event_connection.close()
        ascend_task_connection.close()


def run_msprof_export_db(profiler_run_dir: str | Path) -> str:
    """调用 `msprof --export=on --type=db`。

    msprof 自带一套独立 Python 工具链，必须把 PYTHONPATH 清空，避免本仓库
    的模块解析路径污染它。
    """

    import os
    import subprocess

    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    proc = subprocess.run(
        [
            "msprof",
            "--export=on",
            "--type=db",
            f"--output={profiler_run_dir}",
        ],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    if proc.returncode == 0:
        return "msprof_export_db_ok"
    stderr = (proc.stderr or proc.stdout or "").strip().replace("\n", "|")
    if not stderr:
        stderr = f"code={proc.returncode}"
    raise RuntimeError(f"msprof_export_db_failed:{stderr}")


__all__ = [
    "API_EVENT_DB_NAME",
    "ASCEND_TASK_DB_NAME",
    "DEFAULT_EXPORT_DB_FILE_TIMEOUT_SECONDS",
    "ExportDbDiagnostics",
    "ExportDbNotFoundError",
    "ExportDbParseError",
    "ExportDbStepDurations",
    "find_unique_file",
    "parse_export_db_dir",
    "parse_step_durations_us",
    "run_msprof_export_db",
]
