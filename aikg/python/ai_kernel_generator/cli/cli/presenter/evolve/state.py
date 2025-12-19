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

import time
from dataclasses import dataclass, field
from typing import Any

from ai_kernel_generator.cli.cli.constants import SyntaxLanguage
from textual import log


@dataclass(slots=True)
class EvolveTaskStore:
    watch_task_id: str = ""
    known_task_ids: list[str] = field(default_factory=list)
    task_events: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    task_llm_state: dict[str, dict[str, Any]] = field(default_factory=dict)
    # task_id -> TaskStreamSession（用于“切 tab 不断点”的流式渲染状态封装）
    task_stream_sessions: dict[str, Any] = field(default_factory=dict)
    task_node_state: dict[str, dict[str, Any]] = field(default_factory=dict)
    # task_id -> 主输出区（Chat）的渲染内容序列（用于“切 tab 即刻还原”，避免重放不一致）
    task_main_contents: dict[str, list[Any]] = field(default_factory=dict)
    # 单个 task 最多保留多少条 main_content（防止内存无限增长）
    task_main_contents_limit: int = 4000

    evolve_round_snapshots: dict[int, dict[str, Any]] = field(default_factory=dict)
    evolve_task_summary: dict[str, dict[str, Any]] = field(default_factory=dict)

    latest_progress_text: str = ""
    latest_progress_data: dict[str, Any] = field(default_factory=dict)

    replaying: bool = False

    global_trace_seq: int = 0
    # task_id -> Trace items（仅 node_start）：(text, task_id, event_idx)
    task_start_trace_items: dict[str, list[tuple[str, str, int]]] = field(
        default_factory=dict
    )

    # UI 事件泵：让 [ / ] 切换即时生效（不依赖 server 新消息）
    ui_pump_task: Any | None = None
    ui_pump_running: bool = False


class TaskStateRepository:
    """evolve 相关状态与事件仓库（纯数据 + 少量一致性维护）。"""

    def __init__(self, store: EvolveTaskStore) -> None:
        self._s = store

    def ensure_task_node_state(self, task_id: str) -> dict[str, Any]:
        tid = (task_id or "").strip()
        if not tid:
            tid = "main"
        st = self._s.task_node_state.get(tid)
        if not isinstance(st, dict):
            st = {
                "current_node": "",
                "status": "idle",
                "seen_nodes": [],
                # node -> run count（用于 coder×2 / coder#2）
                "run_counts": {},
                # 当前节点正在执行的 run_no（用于 Current: coder#2）
                "current_run_no": 0,
                # 递增序号（用于 Trace）
                "seq": 0,
                # 最近执行轨迹（字符串列表）
                "trace": [],
            }
            self._s.task_node_state[tid] = st
        st.setdefault("current_node", "")
        st.setdefault("status", "idle")
        st.setdefault("seen_nodes", [])
        st.setdefault("run_counts", {})
        st.setdefault("current_run_no", 0)
        st.setdefault("seq", 0)
        st.setdefault("trace", [])
        return st

    def apply_task_node_state(self, presenter: Any, task_id: str) -> None:
        """把某个 task 的节点态同步到当前 UI（用于 watch 切换）。"""
        p = presenter
        tid = (task_id or "").strip()
        if not tid:
            tid = "main"
        st = self.ensure_task_node_state(tid)
        p._current_node = str(st.get("current_node") or "")
        p._current_node_status = str(st.get("status") or "idle")
        try:
            p._current_node_run_no = int(st.get("current_run_no") or 0)
        except (TypeError, ValueError) as e:
            log.debug(
                "[EvolveState] current_run_no cast failed; fallback 0", exc_info=e
            )
            p._current_node_run_no = 0

        rc = st.get("run_counts")
        p._node_run_counts = dict(rc) if isinstance(rc, dict) else {}

        tr = st.get("trace")
        p._node_trace = [str(x) for x in tr[-12:]] if isinstance(tr, list) else []
        seen = st.get("seen_nodes")
        p._seen_nodes = (
            [str(x) for x in seen if str(x)] if isinstance(seen, list) else []
        )

    def ensure_task_known(self, task_id: str) -> None:
        tid = (task_id or "").strip()
        if not tid:
            return
        if tid not in self._s.known_task_ids:
            self._s.known_task_ids.append(tid)
        self._s.task_events.setdefault(tid, [])
        self._s.task_llm_state.setdefault(
            tid,
            {
                "running": False,
                "agent": "",
                "model": "",
                "language": SyntaxLanguage.TEXT,
            },
        )
        self._s.task_main_contents.setdefault(tid, [])

    def append_task_main_content(self, task_id: str, content: Any) -> None:
        tid = (task_id or "").strip()
        if not tid:
            return
        self.ensure_task_known(tid)
        buf = self._s.task_main_contents.get(tid)
        if not isinstance(buf, list):
            buf = []
            self._s.task_main_contents[tid] = buf
        buf.append(content)
        limit = int(self._s.task_main_contents_limit or 0)
        if limit > 0 and len(buf) > limit:
            # 只保留末尾 N 条（历史可由 server-side log/trace 另行补齐）
            del buf[: max(0, len(buf) - limit)]

        # 注意：不要在这里打高频日志，会显著拖慢流式渲染与 UI 刷新。

    def task_main_contents(self, task_id: str) -> list[Any]:
        tid = (task_id or "").strip()
        if not tid:
            return []
        buf = self._s.task_main_contents.get(tid)
        return list(buf or []) if isinstance(buf, list) else []

    def record_task_event(self, task_id: str, payload: dict[str, Any]) -> int | None:
        """记录轻量事件（不记录 prompt）；返回 event index（用于 trace jump）。"""
        tid = (task_id or "").strip()
        if not tid:
            return None
        self.ensure_task_known(tid)
        p = dict(payload or {})
        p.setdefault("ts", time.time())
        self._s.task_events[tid].append(p)
        return max(0, len(self._s.task_events[tid]) - 1)

    def task_events(self, task_id: str) -> list[dict[str, Any]]:
        return list(self._s.task_events.get((task_id or "").strip(), []) or [])

    def task_llm_state(self, task_id: str) -> dict[str, Any]:
        tid = (task_id or "").strip()
        if not tid:
            tid = "main"
        self.ensure_task_known(tid)
        return self._s.task_llm_state.get(tid) or {}

    def sorted_task_ids(self) -> list[str]:
        def _key(tid: str):
            parts = (tid or "").split("_")
            nums = []
            for p in parts:
                try:
                    nums.append(int(p))
                except (TypeError, ValueError):
                    nums.append(10**9)
            return nums + [len(parts), tid]

        return sorted(self._s.known_task_ids, key=_key)
