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

import hashlib
import os
from collections import OrderedDict
from typing import Any, Optional, TYPE_CHECKING

from rich.syntax import Syntax
from rich.text import Text
from textual import log

from ai_kernel_generator.cli.cli.constants import DisplayStyle, UISymbol
from ai_kernel_generator.cli.cli.utils.syntax_utils import build_syntax_theme
from .theme_manager import ThemeManager

if TYPE_CHECKING:
    from .app import SplitViewApp


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in ["1", "true", "yes", "on", "y"]


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


_SYNTAX_CACHE_ENABLED: bool = _env_bool("AIKG_TUI_SYNTAX_CACHE", True)
_SYNTAX_CACHE_MAX_ITEMS: int = _env_int("AIKG_TUI_SYNTAX_CACHE_MAX_ITEMS", 128)
_SYNTAX_CACHE_MAX_CHARS: int = _env_int("AIKG_TUI_SYNTAX_CACHE_MAX_CHARS", 1_000_000)


class ChatManager:
    """Chat 日志写入与滚动定位。"""

    def __init__(self, app: "SplitViewApp", theme: ThemeManager) -> None:
        self.app = app
        self.theme = theme
        self._syntax_cache_enabled = bool(
            _SYNTAX_CACHE_ENABLED
            and _SYNTAX_CACHE_MAX_ITEMS > 0
            and _SYNTAX_CACHE_MAX_CHARS > 0
        )
        self._syntax_cache_max_items = int(_SYNTAX_CACHE_MAX_ITEMS)
        self._syntax_cache_max_chars = int(_SYNTAX_CACHE_MAX_CHARS)
        self._syntax_cache: OrderedDict[
            tuple[str, str, str, int, str, int], tuple[list[Text], int]
        ] = OrderedDict()
        self._syntax_cache_chars: int = 0

    def _syntax_cache_key(
        self,
        code: str,
        lexer_name: str,
        width: int,
        theme_name: str,
        background_color: str | None,
    ) -> tuple[str, str, str, int, str, int]:
        code_b = str(code or "").encode("utf-8", "ignore")
        code_hash = hashlib.blake2b(code_b, digest_size=8).hexdigest()
        return (
            str(lexer_name or "text"),
            str(theme_name or ""),
            str(background_color or ""),
            int(width),
            code_hash,
            len(code_b),
        )

    def _syntax_cache_get(
        self, key: tuple[str, str, str, int, str, int]
    ) -> list[Text] | None:
        if not self._syntax_cache_enabled:
            return None
        entry = self._syntax_cache.get(key)
        if entry is None:
            return None
        self._syntax_cache.move_to_end(key)
        return entry[0]

    def _syntax_cache_set(
        self, key: tuple[str, str, str, int, str, int], lines: list[Text], size: int
    ) -> None:
        if not self._syntax_cache_enabled:
            return
        if size <= 0:
            return
        if size > self._syntax_cache_max_chars:
            return
        self._syntax_cache[key] = (lines, size)
        self._syntax_cache_chars += size
        while (
            len(self._syntax_cache) > self._syntax_cache_max_items
            or self._syntax_cache_chars > self._syntax_cache_max_chars
        ):
            _, (_, evict_size) = self._syntax_cache.popitem(last=False)
            self._syntax_cache_chars -= int(evict_size or 0)

    def tagged_line(self, tag: str, message: str, *, color_var: str) -> Text:
        color = self.theme.theme_color(color_var, "default")
        text = Text()
        text.append(str(tag), style=f"bold {color}")
        if message:
            text.append(" ")
            text.append(str(message))
        return text

    def clear(self) -> None:
        if self.app.chat_log is not None:
            try:
                self.app.chat_log.clear()
            except Exception as e:
                log.debug("[Chat] chat_log.clear failed", exc_info=e)
        try:
            self.app.trace_anchors.clear()
        except Exception as e:
            log.debug("[Chat] trace_anchors.clear failed", exc_info=e)

    def get_content_width(self) -> Optional[int]:
        if self.app.chat_log is None:
            return None
        w = getattr(getattr(self.app.chat_log, "content_size", None), "width", None)
        if w is None:
            w = getattr(getattr(self.app.chat_log, "size", None), "width", None)
        try:
            width_i = int(w)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
        return width_i if width_i > 0 else None

    def render_syntax_block_lines(
        self,
        code: str,
        lexer_name: str,
        *,
        add_end_separator: bool = True,
        line_number_start: int | None = None,
    ) -> list[Text]:
        def _prefix_for(n: int | None) -> Text:
            if n is None:
                return Text(f"    {UISymbol.VERTICAL_BAR} ")
            return Text.from_markup(
                f"[{DisplayStyle.DIM}]{int(n):3d}[/{DisplayStyle.DIM}] {UISymbol.VERTICAL_BAR} "
            )

        def _end_sep() -> Text:
            # 结束分隔线不占用行号，只保留竖线
            return Text(f"    {UISymbol.VERTICAL_BAR}")

        if not code:
            return [_end_sep()] if add_end_separator else []

        width = self.get_content_width()
        if width is None:
            width = int(getattr(self.app.console, "width", 80) or 80)

        theme_name = self.theme.syntax_theme_name()
        background_color = (
            None
            if self.theme.is_dark()
            else self.theme.theme_color("background", "#FFFFFF")
        )
        probe_prefix = _prefix_for(line_number_start)
        effective_width = max(1, int(width) - probe_prefix.cell_len - 1)
        options = self.app.console.options.update(width=effective_width)
        cache_key = self._syntax_cache_key(
            code, lexer_name, int(effective_width), theme_name, background_color
        )
        base_lines = self._syntax_cache_get(cache_key)
        if base_lines is None:
            syntax = Syntax(
                code,
                lexer_name or "text",
                theme=build_syntax_theme(theme_name, suppress_error_tokens=True),
                word_wrap=True,
                background_color=background_color,
                padding=(0, 1),
            )
            lines = self.app.console.render_lines(syntax, options)
            base_lines = [
                Text.assemble(*((seg.text, seg.style) for seg in line_segments))
                for line_segments in lines
            ]
            self._syntax_cache_set(cache_key, base_lines, len(code))

        out: list[Text] = []
        for i, line_text in enumerate(base_lines):
            ln = (int(line_number_start) + i) if line_number_start is not None else None
            out.append(_prefix_for(ln) + line_text)
        if add_end_separator:
            out.append(_end_sep())
        return out

    def write_text(self, text: Text) -> None:
        if self.app.chat_log is None:
            return
        try:
            self.app.chat_log.write(text)
        except Exception as e:
            log.warning("[Chat] chat_log.write(text) failed", exc_info=e)
            return

    def write_markup(self, markup: str) -> None:
        if self.app.chat_log is None:
            return
        try:
            self.app.chat_log.write(Text.from_markup(str(markup)))
        except Exception as e:
            log.debug("[Chat] Text.from_markup failed; fallback Text", exc_info=e)
            try:
                self.app.chat_log.write(Text(str(markup)))
            except Exception as e2:
                log.warning("[Chat] chat_log.write(markup) failed", exc_info=e2)
                return

    def write_renderable(self, renderable: Any) -> None:
        if isinstance(renderable, str):
            self.write_markup(renderable)
            return
        if isinstance(renderable, Text):
            self.write_text(renderable)
            return
        if self.app.chat_log is None:
            return
        try:
            self.app.chat_log.write(renderable)
        except Exception as e:
            log.warning("[Chat] chat_log.write(renderable) failed", exc_info=e)
            return

    def write_syntax_block(
        self,
        code: str,
        lexer_name: str,
        *,
        add_end_separator: bool = True,
        line_number_start: int | None = None,
    ) -> None:
        if self.app.chat_log is None:
            return
        try:
            for line in self.render_syntax_block_lines(
                str(code or ""),
                str(lexer_name or "text"),
                add_end_separator=add_end_separator,
                line_number_start=line_number_start,
            ):
                self.app.chat_log.write(line)
        except Exception as e:
            log.warning(
                "[Chat] write_syntax_block failed; fallback plain text", exc_info=e
            )
            try:
                self.app.chat_log.write(Text(str(code or "")))
            except Exception as e2:
                log.warning("[Chat] write plain text failed", exc_info=e2)
                return

    def record_trace_anchor(self, task_id: str, event_idx: int) -> None:
        if self.app.chat_log is None:
            return
        try:
            anchor_y = int(getattr(self.app.chat_log, "virtual_size").height or 0)
            self.app.trace_anchors[(str(task_id or ""), int(event_idx or 0))] = anchor_y
        except Exception as e:
            log.debug("[Chat] record_trace_anchor failed", exc_info=e)
            return
