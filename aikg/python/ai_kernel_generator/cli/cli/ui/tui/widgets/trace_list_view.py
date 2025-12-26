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

"""Trace 列表视图组件 - 完全自治版本"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional
from textual.widgets import ListView
from textual import log

from .trace_list_item import TraceListItem

if TYPE_CHECKING:
    from .log_pane import LogPane


class TraceListView(ListView):
    """自定义 Trace 列表视图，完全自治处理事件"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._chat_log: Optional[LogPane] = None
        self._on_item_selected: Optional[Callable[[], None]] = None
        self._last_selected: Optional[tuple[str, int]] = None

    def set_chat_log(self, chat_log: LogPane) -> None:
        """注入 chat_log 依赖"""
        self._chat_log = chat_log

    def set_on_item_selected_callback(self, callback: Callable[[], None]) -> None:
        """注入选中时的回调（用于 App 级别的副作用，如 set_follow_tail）"""
        self._on_item_selected = callback

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """处理列表项选择 - 完全在组件内部完成，不向外发送任何消息"""
        event.stop()  # 阻止事件冒泡到 App

        item = event.item
        if not isinstance(item, TraceListItem):
            return

        if self._chat_log is None:
            return

        try:
            log.info(
                "[TraceListView] trace selected",
                task_id=str(getattr(item, "task_id", "") or ""),
                event_idx=int(getattr(item, "event_idx", 0) or 0),
                anchor_y=int(getattr(item, "anchor_y", 0) or 0),
            )
        except Exception as e:
            log.warning("[TraceListView] log selection failed", exc_info=e)

        try:
            self._last_selected = (str(item.task_id or ""), int(item.event_idx or 0))
        except Exception:
            self._last_selected = None

        # 执行回调（如果有）- 例如 set_follow_tail(False)
        if self._on_item_selected is not None:
            try:
                self._on_item_selected()
            except Exception as e:
                log.warning(
                    "[TraceListView] on_item_selected callback failed", exc_info=e
                )

        # 更新当前选择的 anchor_y（用于状态显示）
        try:
            if self.app is not None:
                getattr(self.app, "set_trace_anchor_y", lambda *_: None)(anchor_y)
        except Exception as e:
            log.debug("[TraceListView] set_trace_anchor_y failed", exc_info=e)

        # 聚焦到 chat_log
        try:
            self._chat_log.focus()
        except Exception as e:
            log.debug("[TraceListView] chat_log.focus failed", exc_info=e)

        # 滚动到指定位置
        anchor_y = int(getattr(item, "anchor_y", 0) or 0)
        try:
            self._chat_log.scroll_to(y=float(anchor_y), animate=False, immediate=True)
        except Exception as e:
            log.debug(
                "[TraceListView] scroll_to(immediate) failed; fallback", exc_info=e
            )
            try:
                self._chat_log.scroll_to(y=float(anchor_y), animate=False)
            except Exception as e2:
                log.warning("[TraceListView] scroll_to failed", exc_info=e2)

    def update_anchor(self, task_id: str, event_idx: int, anchor_y: int) -> None:
        """回填/更新指定 Trace 条目的 anchor_y。"""
        task_id_s = str(task_id or "")
        try:
            event_idx_i = int(event_idx or 0)
        except (TypeError, ValueError):
            return
        updated = False
        try:
            for item in self.query(TraceListItem):
                if (
                    str(getattr(item, "task_id", "") or "") == task_id_s
                    and int(getattr(item, "event_idx", 0) or 0) == event_idx_i
                ):
                    item.update_anchor_y(anchor_y)
                    updated = True
                    break
        except Exception as e:
            log.debug("[TraceListView] update_anchor query failed", exc_info=e)
            return
        if not updated:
            return
        if self._last_selected == (task_id_s, event_idx_i):
            try:
                if self.app is not None:
                    getattr(self.app, "set_trace_anchor_y", lambda *_: None)(anchor_y)
            except Exception as e:
                log.debug(
                    "[TraceListView] update current anchor_y failed", exc_info=e
                )
