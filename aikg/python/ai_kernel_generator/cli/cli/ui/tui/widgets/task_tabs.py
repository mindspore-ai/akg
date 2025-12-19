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

"""任务标签页组件 - 完全自治版本"""

from __future__ import annotations

from typing import TYPE_CHECKING
from queue import Queue
from textual.widgets import Tabs
from textual import log

if TYPE_CHECKING:
    from ai_kernel_generator.cli.cli.ui.intents import WatchSet


class TaskTabs(Tabs):
    """自定义任务标签页，完全自治处理标签切换"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._tab_id_to_task_id: dict[str, str] = {}
        self._active_task_id: str | None = None
        self._suppress_event: bool = False
        self._ui_event_queue: Queue | None = None

    def set_ui_event_queue(self, queue: Queue) -> None:
        """注入 UI 事件队列依赖"""
        self._ui_event_queue = queue

    def set_tab_mapping(self, tab_id_to_task_id: dict[str, str]) -> None:
        """设置 tab_id 到 task_id 的映射"""
        self._tab_id_to_task_id = tab_id_to_task_id

    def set_suppress_event(self, suppress: bool) -> None:
        """设置是否抑制事件"""
        self._suppress_event = suppress

    def get_active_task_id(self) -> str | None:
        """获取当前活动的 task_id"""
        return self._active_task_id

    def set_active_task_id(self, task_id: str | None) -> None:
        """设置当前活动的 task_id"""
        self._active_task_id = task_id

    def on_tabs_tab_activated(self, event: Tabs.TabActivated) -> None:
        """处理标签激活 - 完全在组件内部完成，不向外发送任何消息"""
        event.stop()  # 阻止事件冒泡到 App

        try:
            suppress = bool(self._suppress_event)
            # 兼容 App/QueueProcessor 的抑制开关：程序设置 active 时不要触发 WatchSet
            if bool(getattr(getattr(self, "app", None), "_suppress_tab_event", False)):
                suppress = True

            if suppress:
                log.debug("[TaskTabs] tab_activated suppressed")
                return

            tab = event.tab
            tid = ""
            try:
                tab_id = str(getattr(tab, "id", "") or "").strip()
                tid = str(self._tab_id_to_task_id.get(tab_id, "") or "").strip()
            except Exception as e:
                log.warning("[TaskTabs] resolve task_id failed", exc_info=e)
                tid = ""

            if not tid:
                log.warning(
                    "[TaskTabs] tab_activated: task_id not found",
                    tab_id=str(getattr(tab, "id", "") or ""),
                )
                return

            if tid == (self._active_task_id or ""):
                log.debug(
                    "[TaskTabs] tab_activated: already active",
                    task_id=tid,
                )
                return

            self._active_task_id = tid

            log.info(
                "[TaskTabs] tab_activated: watch_set",
                task_id=tid,
            )

            # 直接向队列发送事件，无需通过 App
            if self._ui_event_queue is not None:
                from ai_kernel_generator.cli.cli.ui.intents import WatchSet

                self._ui_event_queue.put_nowait(WatchSet(task_id=tid))

        except Exception as e:
            log.error("[TaskTabs] on_tabs_tab_activated failed", exc_info=e)
            return
