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

from textual.widgets import RichLog
from textual import log


class LogPane(RichLog):
    """可复用的日志区域（Chat/Task 都用 RichLog）。"""

    def __init__(self, title: str, *args, **kwargs):
        # RichLog 默认 min_width=78，容易在分屏时把整体撑出横向滚动；
        # 同时默认 wrap=False，会导致长行触发横向滚动条。
        kwargs.setdefault("min_width", 0)
        kwargs.setdefault("wrap", True)
        kwargs.setdefault("max_lines", 1000)
        # 实现"tmux tail"体验：
        # - 在底部：自动追踪最新输出
        # - 向上滚动离开底部：固定视图，不被新输出强制拉回底部
        kwargs.setdefault("auto_scroll", False)
        super().__init__(*args, **kwargs)
        self.border_title = title
        # 尊重传入的 max_lines（RichLog 会基于该值裁剪历史）。
        try:
            self.max_lines = int(kwargs.get("max_lines", 1000) or 0)
        except (TypeError, ValueError) as e:
            log.debug("[LogPane] max_lines cast failed; fallback 1000", exc_info=e)
            self.max_lines = 1000

    # 禁用滚动动画（大日志量时更顺滑）
    def action_page_up(self) -> None:
        try:
            if getattr(self, "id", "") == "chat-log" and self.app is not None:
                getattr(self.app, "set_follow_tail", lambda *_: None)(False)
        except Exception as e:
            log.debug("[LogPane] set_follow_tail(False) failed", exc_info=e)
        self.scroll_page_up(animate=False)

    def action_page_down(self) -> None:
        self.scroll_page_down(animate=False)

    def action_scroll_up(self) -> None:
        try:
            if getattr(self, "id", "") == "chat-log" and self.app is not None:
                getattr(self.app, "set_follow_tail", lambda *_: None)(False)
        except Exception as e:
            log.debug("[LogPane] set_follow_tail(False) failed", exc_info=e)
        self.scroll_up(animate=False)

    def action_scroll_down(self) -> None:
        self.scroll_down(animate=False)

    def action_scroll_home(self) -> None:
        try:
            if getattr(self, "id", "") == "chat-log" and self.app is not None:
                getattr(self.app, "set_follow_tail", lambda *_: None)(False)
        except Exception as e:
            log.debug("[LogPane] set_follow_tail(False) failed", exc_info=e)
        self.scroll_home(animate=False)

    def action_scroll_end(self) -> None:
        try:
            if getattr(self, "id", "") == "chat-log" and self.app is not None:
                getattr(self.app, "set_follow_tail", lambda *_: None)(True)
        except Exception as e:
            log.debug("[LogPane] set_follow_tail(True) failed", exc_info=e)
        self.scroll_end(animate=False)

    def action_scroll_left(self) -> None:
        self.scroll_left(animate=False)

    def action_scroll_right(self) -> None:
        self.scroll_right(animate=False)

    def action_page_left(self) -> None:
        self.scroll_page_left(animate=False)

    def action_page_right(self) -> None:
        self.scroll_page_right(animate=False)
