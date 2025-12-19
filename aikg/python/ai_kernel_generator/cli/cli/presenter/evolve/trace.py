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

from ai_kernel_generator.cli.cli.constants import DisplayStyle
from textual import log
from ...utils.i18n import t


class TraceController:
    """Trace 面板与任务 Tabs 的聚合刷新。"""

    def __init__(self, presenter, store, *, sorted_task_ids, get_watch_task_id) -> None:
        self._p = presenter
        self._s = store
        self._sorted_task_ids = sorted_task_ids
        self._get_watch_task_id = get_watch_task_id

    def trace_filter_task_id(self) -> str:
        """当前 Trace 面板过滤的 task_id（默认跟随 watch）。"""
        tid = (self._get_watch_task_id() or "").strip()
        if tid:
            return tid
        # 单任务/旧 server：没有 task_id 时，用默认桶；如果只存在一个 task，也用它
        if "task_init" in (self._s.task_start_trace_items or {}):
            return "task_init"
        if "main" in (self._s.task_start_trace_items or {}):
            return "main"
        keys = list((self._s.task_start_trace_items or {}).keys())
        if len(keys) == 1:
            return str(keys[0] or "")
        return ""

    def refresh_trace_panel(self) -> None:
        """按当前 watch_task_id 刷新 Trace 面板（过滤显示）。"""
        p = self._p
        try:
            if not hasattr(p.layout_manager, "set_trace_items"):
                return
            tid = self.trace_filter_task_id()
            items = self._s.task_start_trace_items.get(tid, []) if tid else []
            p.layout_manager.set_trace_items(list(items))
            if hasattr(p.layout_manager, "set_trace_title"):
                title = (
                    t("presenter.trace.title", tid=tid)
                    if tid
                    else t("presenter.trace.title_no_task")
                )
                p.layout_manager.set_trace_title(title + t("tui.hint.trace_click"))
        except Exception as e:
            log.warning("[Trace] refresh_trace_panel failed", exc_info=e)

    def refresh_task_tabs(self) -> None:
        """刷新任务 Tabs（用于更友好的 task 切换）。"""
        p = self._p
        try:
            if not hasattr(p.layout_manager, "set_task_tabs"):
                return
            ids = self._sorted_task_ids()
            items: list[tuple[str, str]] = []
            for tid in ids:
                t_id = str(tid or "")
                if not t_id:
                    continue
                if t_id == "main":
                    label = "main"
                else:
                    label = t_id
                items.append((t_id, label))
            active = (self._get_watch_task_id() or "").strip() or (
                ids[-1] if ids else ""
            )
            p.layout_manager.set_task_tabs(items, str(active or ""))
        except Exception as e:
            log.warning("[Trace] refresh_task_tabs failed", exc_info=e)

    def record_trace_node_start(
        self, *, task_id: str, node: str, run_no: int, event_idx: int
    ) -> None:
        """记录 node_start 到 Trace 列表，并在当前过滤 task 下增量更新面板。"""
        p = self._p
        try:
            self._s.global_trace_seq += 1
            seq = self._s.global_trace_seq
            line = (
                f"[{DisplayStyle.DIM}]{seq:04d}[/{DisplayStyle.DIM}] ▶ {node}#{run_no}"
            )
            self._s.task_start_trace_items.setdefault(task_id, []).append(
                (line, task_id, int(event_idx))
            )
            if task_id == self.trace_filter_task_id() and hasattr(
                p.layout_manager, "append_trace_item"
            ):
                p.layout_manager.append_trace_item(
                    line, task_id=task_id, event_idx=event_idx
                )
        except Exception as e:
            log.warning("[Trace] record_trace_node_start failed", exc_info=e)
