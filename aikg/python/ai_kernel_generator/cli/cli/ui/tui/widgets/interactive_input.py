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

import os
from typing import Callable, Optional

from textual import log
from textual.events import Key
from textual.widgets import TextArea


class InteractiveInput(TextArea):
    """底部交互式输入框（多行）。"""


    def __init__(
        self, on_submit: Optional[Callable[[str], None]] = None, *args, **kwargs
    ):
        kwargs.setdefault("soft_wrap", True)
        super().__init__(*args, **kwargs)
        self.on_submit_callback = on_submit
        # 关键：关闭光标闪烁，避免某些 SSH 客户端在 inline 模式下因为"持续重绘"而无法滚动回看。
        # （滚动回看一旦发生任何输出/重绘，客户端会自动跳回底部）
        try:
            self.cursor_blink = False
        except Exception as e:
            log.debug("[InteractiveInput] disable cursor_blink failed", exc_info=e)

    def _debug_key_event(self, event: Key) -> None:
        if os.getenv("AKG_TUI_KEY_DEBUG") != "1":
            return
        try:
            self.app.log(
                "InteractiveInput key=%r character=%r modifiers=%r ctrl=%r shift=%r alt=%r meta=%r",
                getattr(event, "key", None),
                getattr(event, "character", None),
                getattr(event, "modifiers", None),
                getattr(event, "ctrl", None),
                getattr(event, "shift", None),
                getattr(event, "alt", None),
                getattr(event, "meta", None),
            )
        except Exception as e:
            log.debug("[InteractiveInput] app.log failed", exc_info=e)

    def action_submit(self) -> None:
        value = (self.text or "").strip()
        if value and self.on_submit_callback:
            self.on_submit_callback(value)
        try:
            self.clear()
        except Exception as e:
            log.debug("[InteractiveInput] clear failed; fallback set text", exc_info=e)
            self.text = ""

        if not self.disabled:
            try:
                if self.app is not None:
                    self.app.call_later(self.focus)
                else:
                    self.focus()
            except Exception as e:
                log.debug("[InteractiveInput] refocus failed", exc_info=e)

    def action_newline(self) -> None:
        try:
            self.insert("\n")
        except Exception as e:
            log.debug("[InteractiveInput] insert newline failed", exc_info=e)

    def on_key(self, event: Key) -> None:
        # 交互输入：Enter 提交；Ctrl+J 换行。（Ctrl+Enter 在大多数终端下无法与 Enter 区分）
        self._debug_key_event(event)
        key = (event.key or "").lower()
        if key in {"enter", "ctrl+m"}:
            event.prevent_default()
            self.action_submit()
            return
        if key in {"ctrl+j"}:
            event.prevent_default()
            self.action_newline()
            return
