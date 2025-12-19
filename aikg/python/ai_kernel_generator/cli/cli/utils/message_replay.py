# Copyright 2025 Huawei Technologies Co., Ltd
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

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Optional

from textual import log

from ai_kernel_generator.cli.cli.utils.message_recording import (
    ReplaySpeed,
    choose_speed_multiplier,
    load_recorded_frames,
)
from ai_kernel_generator.cli.messages import (
    ErrorMessage,
    FinalResultMessage,
    JobSubmittedMessage,
    LLMEndMessage,
    LLMStartMessage,
    LLMStreamMessage,
    NodeEndMessage,
    NodeStartMessage,
    ProgressMessage,
    unpack_message,
)


async def replay_recorded_messages(
    *,
    presenter: Any,
    record_path: Path,
    speed: ReplaySpeed,
) -> dict[str, Any]:
    """回放 messages.jsonl：用同一条 unpack_message + presenter.on_xxx 路径重放 UI。"""

    frames = load_recorded_frames(record_path)
    if not frames:
        raise FileNotFoundError(f"未找到回放数据: {record_path}")

    last_mono: Optional[float] = None
    final_result: dict[str, Any] | None = None

    for fr in frames:
        msg = unpack_message(fr.payload)
        if msg is None:
            continue

        # 速度控制（基于录制时相邻 frame 的 monotonic 间隔）
        if last_mono is not None:
            dt = max(0.0, float(fr.mono) - float(last_mono))
            mult = choose_speed_multiplier(msg, speed)
            if mult != float("inf"):
                try:
                    sleep_s = dt / max(1e-6, float(mult))
                except (TypeError, ValueError, ZeroDivisionError) as e:
                    log.debug("[Replay] sleep compute failed; fallback 0", exc_info=e)
                    sleep_s = 0.0
                sleep_s = min(
                    float(speed.max_sleep), max(float(speed.min_sleep), float(sleep_s))
                )
                if sleep_s > 0:
                    await asyncio.sleep(sleep_s)
        last_mono = float(fr.mono)

        # FinalResult 在 client 里不会 route；回放时记录结果但不要提前结束：
        # 一个 session 里可能包含多段 WS 执行（例如 task_init + job），会出现多个 FinalResult。
        if isinstance(msg, FinalResultMessage):
            try:
                final_result = dict(getattr(msg, "result", {}) or {})
            except (TypeError, ValueError) as e:
                log.debug(
                    "[Replay] FinalResultMessage.result cast failed; fallback {}",
                    exc_info=e,
                )
                final_result = {}
            continue

        # JobSubmitted 在 client 里会单独处理；回放时也要补齐，否则 job_id/追踪信息会缺失
        if isinstance(msg, JobSubmittedMessage):
            jid = str(getattr(msg, "job_id", "") or "")
            if jid and hasattr(presenter, "on_job_submitted"):
                try:
                    presenter.on_job_submitted(jid)
                except Exception as e:
                    log.warning(
                        "[Replay] presenter.on_job_submitted failed",
                        job_id=jid,
                        exc_info=e,
                    )
            continue

        if isinstance(msg, ErrorMessage):
            raise RuntimeError(f"服务器错误: {getattr(msg, 'error', '')}")

        _route_to_presenter(presenter, msg)

    return final_result or {}


def _route_to_presenter(presenter: Any, message: Any) -> None:
    """复用 CliClient._route_to_presenter 的分发逻辑（避免依赖私有方法）。"""
    if presenter is None:
        return
    try:
        if isinstance(message, NodeStartMessage):
            presenter.on_node_start(message)
            return
        if isinstance(message, NodeEndMessage):
            presenter.on_node_end(message)
            return
        if isinstance(message, LLMStartMessage):
            presenter.on_llm_start(message)
            return
        if isinstance(message, LLMEndMessage):
            presenter.on_llm_end(message)
            return
        if isinstance(message, LLMStreamMessage):
            presenter.on_llm_stream(message)
            return
        if isinstance(message, ProgressMessage):
            if hasattr(presenter, "on_progress"):
                presenter.on_progress(message)
            return
    except Exception as e:
        log.warning(
            "[Replay] route_to_presenter failed",
            message_type=type(message).__name__,
            exc_info=e,
        )
        return
