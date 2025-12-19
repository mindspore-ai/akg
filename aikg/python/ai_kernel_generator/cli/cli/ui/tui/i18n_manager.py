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

from typing import TYPE_CHECKING

from textual.binding import Binding
from textual import log

from ai_kernel_generator.cli.cli.utils.i18n import t

if TYPE_CHECKING:
    from .app import SplitViewApp


class I18nManager:
    """i18n 文案刷新（标题/placeholder/快捷键/响应式面板翻译器）。"""

    def __init__(self, app: "SplitViewApp") -> None:
        self.app = app

    def _update_binding_desc(self, key: str, action: str, desc: str) -> None:
        try:
            bmap = getattr(self.app, "_bindings", None)
            if bmap is None:
                return
            lst = list(getattr(bmap, "key_to_bindings", {}).get(key, []) or [])
            if not lst:
                return
            new_lst = []
            replaced = False
            for b in lst:
                if getattr(b, "action", None) == action:
                    new_lst.append(
                        Binding(
                            b.key,
                            b.action,
                            desc,
                            show=bool(getattr(b, "show", False)),
                            key_display=getattr(b, "key_display", None),
                            priority=bool(getattr(b, "priority", False)),
                            tooltip=str(getattr(b, "tooltip", "") or ""),
                            id=getattr(b, "id", None),
                        )
                    )
                    replaced = True
                else:
                    new_lst.append(b)
            if replaced:
                bmap.key_to_bindings[key] = new_lst
        except Exception as e:
            log.debug(
                "[I18n] update binding desc failed",
                key=str(key),
                action=str(action),
                exc_info=e,
            )
            return

    def apply(self) -> None:
        """刷新 UI 文案（标题/placeholder/快捷键描述）。"""
        # 标题/提示
        if self.app.task_tabs is not None:
            self.app.task_tabs.border_title = t("tui.title.tasks_bar")
        if self.app.chat_log is not None:
            self.app.chat_log.border_title = t("tui.title.chat")
        if self.app.info_panel is not None:
            self.app.info_panel.border_title = t("tui.title.task_info")
        if self.app.workflow_panel is not None:
            self.app.workflow_panel.border_title = t("tui.title.workflow")
        if self.app.trace_panel is not None:
            self.app.trace_panel.border_title = t("tui.title.trace")

        # placeholder（按当前状态尽力推断）
        try:
            if self.app.user_input is not None:
                if getattr(self.app.user_input, "disabled", False):
                    self.app.user_input.placeholder = t(
                        "tui.placeholder.input_disabled"
                    )
                else:
                    wf_done = bool(
                        self.app.workflow_task is not None
                        and self.app._workflow_runner_task is not None
                        and getattr(
                            self.app._workflow_runner_task, "done", lambda: False
                        )()
                    )
                    if wf_done:
                        self.app.user_input.placeholder = t(
                            "tui.placeholder.input_done"
                        )
                    elif self.app.workflow_running:
                        self.app.user_input.placeholder = t(
                            "tui.placeholder.input_enabled_hint"
                        )
                    else:
                        self.app.user_input.placeholder = t(
                            "tui.placeholder.input_initial"
                        )
        except Exception as e:
            log.debug("[I18n] update placeholder failed", exc_info=e)

        # Tail subtitle 也需要跟随语言变化
        try:
            self.app.set_follow_tail(self.app.follow_tail)
        except Exception as e:
            log.debug("[I18n] set_follow_tail failed", exc_info=e)

        # 更新快捷键文案
        self._update_binding_desc("ctrl+c", "quit", t("tui.binding.quit"))
        self._update_binding_desc("ctrl+l", "clear_chat", t("tui.binding.clear_chat"))
        self._update_binding_desc("ctrl+k", "clear_task", t("tui.binding.clear_task"))
        self._update_binding_desc("ctrl+e", "scroll_end", t("tui.binding.scroll_end"))
        self._update_binding_desc("escape", "focus_chat", t("tui.binding.focus_chat"))
        self._update_binding_desc("f2", "focus_input", t("tui.binding.focus_input"))
        self._update_binding_desc("f3", "focus_trace", t("tui.binding.focus_trace"))
        self._update_binding_desc("f4", "toggle_tail", t("tui.binding.toggle_tail"))
        self._update_binding_desc(
            "f5", "copy_task_desc", t("tui.binding.copy_task_desc")
        )
        self._update_binding_desc(
            "f6", "copy_kernel_code", t("tui.binding.copy_kernel_code")
        )
        self._update_binding_desc("f7", "copy_job_id", t("tui.binding.copy_job_id"))
        self._update_binding_desc(
            "f10", "toggle_language", t("tui.binding.toggle_language")
        )
        self._update_binding_desc("f11", "toggle_theme", t("tui.binding.toggle_theme"))
        self._update_binding_desc("[", "watch_prev", t("tui.binding.watch_prev"))
        self._update_binding_desc("]", "watch_next", t("tui.binding.watch_next"))
        self._update_binding_desc("f8", "watch_prev", t("tui.binding.watch_prev"))
        self._update_binding_desc("f9", "watch_next", t("tui.binding.watch_next"))

        try:
            self.app.refresh_bindings()
        except Exception as e:
            log.debug("[I18n] refresh_bindings failed", exc_info=e)

        # 翻译切换后：强制重绘响应式面板，让面板内 label 文案同步刷新
        try:
            self.app.reactive_info_panel.set_translator(t)
            self.app.reactive_workflow_panel.set_translator(t)
        except Exception as e:
            log.debug("[I18n] set_translator failed", exc_info=e)
