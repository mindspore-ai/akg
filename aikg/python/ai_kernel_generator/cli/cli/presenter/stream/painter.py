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

import re
from typing import Callable, Tuple

from rich.console import Console
from rich.markup import escape
from rich.syntax import Syntax
from rich.text import Text
from textual import log

from ai_kernel_generator.cli.cli.constants import DisplayStyle, UISymbol
from ai_kernel_generator.cli.cli.ui.types import SyntaxBlockMainContent
from ai_kernel_generator.cli.cli.utils.syntax_utils import build_syntax_theme
from ai_kernel_generator.cli.cli.utils.text_utils import (
    apply_basic_markdown,
    wrap_rich_text_with_display_width,
)


class ConsolePainter:
    """负责将内容绘制到 Rich Console 或分屏布局。"""

    def __init__(
        self,
        console: Console,
        layout_manager=None,
        *,
        emit_hook: Callable[[object], None] | None = None,
    ):
        self.console = console
        self.layout_manager = layout_manager
        self._emit_hook = emit_hook

    def _emit(self, content) -> None:
        if self._emit_hook is not None:
            try:
                self._emit_hook(content)
            except Exception as e:
                log.debug("[Painter] emit_hook failed; ignore", exc_info=e)
            return
        if self.layout_manager:
            app = getattr(self.layout_manager, "app", None)
            q = getattr(app, "ui_event_queue", None) if app is not None else None
            if q is not None:
                try:
                    from ai_kernel_generator.cli.cli.ui.intents import WriteMainContent

                    q.put_nowait(WriteMainContent(content=content))
                    return
                except Exception as e:
                    log.debug(
                        "[Painter] enqueue WriteMainContent failed; drop",
                        exc_info=e,
                    )

    def _calc_left_panel_width(self) -> int:
        console_width = self.console.width or 80
        if not self.layout_manager:
            # 与 presenter 的分割线策略保持一致：保守减 2，避免刚好超宽触发软换行
            return max(1, console_width - 2)

        # Textual：优先使用真实内容区宽度（避免估算误差导致 RichLog 二次软换行）
        try:
            if hasattr(self.layout_manager, "get_panel_content_width"):
                w = self.layout_manager.get_panel_content_width("chat")
                if isinstance(w, int) and w > 0:
                    # 再保守扣 1，避免刚好贴边触发软换行
                    return max(1, w - 1)
        except Exception as e:
            log.debug(
                "[Painter] get_panel_content_width failed; fallback estimate",
                exc_info=e,
            )

        left = int(console_width * 0.7)
        is_textual = self.layout_manager.__class__.__name__ == "TextualLayoutManager"
        if is_textual:
            left -= 6
        else:
            left -= 2
        return max(1, left)

    def _separator_text(self, style: str) -> Text:
        width = self._calc_left_panel_width()
        return Text("─" * width, style=style)

    def _syntax_theme_name(self) -> str:
        if self.layout_manager:
            app = getattr(self.layout_manager, "app", None)
            theme_mgr = getattr(app, "theme_manager", None)
            if theme_mgr is not None and hasattr(theme_mgr, "syntax_theme_name"):
                try:
                    return str(theme_mgr.syntax_theme_name() or "monokai")
                except Exception as e:
                    log.debug(
                        "[Painter] theme_manager.syntax_theme_name failed",
                        exc_info=e,
                    )
        return "monokai"

    def print_divider(self, title: str = "") -> None:
        if title:
            self._emit(
                Text.from_markup(f"\n[{DisplayStyle.DIM}]{title}[/{DisplayStyle.DIM}]")
            )
        self._emit(self._separator_text(DisplayStyle.DIM))

    def print_finish_mark(self, agent_name: str) -> None:
        self._emit(self._separator_text(DisplayStyle.GREEN))
        if not self.layout_manager:
            self.console.print(
                f"[{DisplayStyle.BOLD_GREEN}] {agent_name} - 完成[/{DisplayStyle.BOLD_GREEN}]\n"
            )

    def _line_number_prefix(self, line_number: int) -> Text:
        return Text.from_markup(
            f"[{DisplayStyle.DIM}]{int(line_number):3d}[/{DisplayStyle.DIM}] {UISymbol.VERTICAL_BAR} "
        )

    def print_normal_line(self, line: str, line_number: int, *, dim: bool = False) -> None:
        console_width = self._calc_left_panel_width()

        prefix_text = self._line_number_prefix(line_number)

        rich_line = apply_basic_markdown(line)
        is_list, rich_line, indent_str = self._process_list_style(rich_line)
        if dim:
            rich_line.stylize(DisplayStyle.DIM)

        subsequent_indent_str = " " * (
            prefix_text.plain.find(UISymbol.VERTICAL_BAR)
            if UISymbol.VERTICAL_BAR in prefix_text.plain
            else prefix_text.cell_len
        )
        if UISymbol.VERTICAL_BAR in prefix_text.plain:
            subsequent_indent_str += f"{UISymbol.VERTICAL_BAR} "
        if is_list:
            subsequent_indent_str += indent_str

        subsequent_indent_width = len(subsequent_indent_str)
        effective_width = console_width - subsequent_indent_width

        wrapped_lines = wrap_rich_text_with_display_width(
            rich_line, effective_width, subsequent_indent_str, self.console
        )

        if self.layout_manager:
            self._emit(prefix_text + wrapped_lines[0])
            for extra in wrapped_lines[1:]:
                self._emit(extra)
            self._emit(Text.from_markup(f"    {UISymbol.VERTICAL_BAR}"))
        else:
            self.console.print(prefix_text + wrapped_lines[0])
            for extra in wrapped_lines[1:]:
                self.console.print(extra)
            self.console.print(f"    {UISymbol.VERTICAL_BAR}")

    def print_syntax_block(
        self,
        code: str,
        lexer_name: str,
        add_end_separator: bool = True,
        *,
        line_number_start: int | None = None,
        dim: bool = False,
    ) -> int:
        """打印语法高亮代码块。

        - `line_number_start` 传入时：使用“自维护行号”对代码块逐行编号，并返回消耗的行号数量。
        - 不传入时：保持旧行为（仅显示左侧竖线前缀，不编号）。
        """
        if self.layout_manager:
            try:
                self._emit(
                    SyntaxBlockMainContent(
                        code=str(code or ""),
                        lexer_name=str(lexer_name or "text"),
                        add_end_separator=bool(add_end_separator),
                        line_number_start=(
                            int(line_number_start)
                            if line_number_start is not None
                            else None
                        ),
                        dim=bool(dim),
                    )
                )
            except Exception as e:
                log.warning(
                    "[Painter] emit SyntaxBlockMainContent failed; fallback plain text",
                    exc_info=e,
                )
                self._emit(str(code or ""))
                if add_end_separator:
                    self._emit(Text.from_markup(f"    {UISymbol.VERTICAL_BAR}"))
            # renderer 依赖返回值推进全局行号：这里按当前面板宽度计算“物理行数”
            if not code:
                return 0
            console_width = self._calc_left_panel_width()
            prefix_width = (
                self._line_number_prefix(int(line_number_start)).cell_len
                if line_number_start is not None
                else 6
            )
            effective_width = max(1, console_width - prefix_width - 1)
            syntax = Syntax(
                str(code or ""),
                str(lexer_name or "text"),
                theme=build_syntax_theme(self._syntax_theme_name(), suppress_error_tokens=True),
                word_wrap=True,
                padding=(0, 1),
            )
            options = self.console.options.update(width=effective_width)
            return len(self.console.render_lines(syntax, options))

        if not code:
            if add_end_separator:
                self.console.print(f"    {UISymbol.VERTICAL_BAR}")
            return 0

        console_width = self._calc_left_panel_width()

        if line_number_start is None:
            effective_width = console_width - 6
            syntax = Syntax(
                code,
                lexer_name or "text",
                theme=build_syntax_theme(self._syntax_theme_name(), suppress_error_tokens=True),
                word_wrap=True,
                padding=(0, 1),
            )
            options = self.console.options.update(width=effective_width)
            lines = self.console.render_lines(syntax, options)
            prefix = Text(f"    {UISymbol.VERTICAL_BAR} ")
            for line_segments in lines:
                line_text = Text.assemble(
                    *((seg.text, seg.style) for seg in line_segments)
                )
                if dim:
                    line_text.stylize(DisplayStyle.DIM)
                if self.layout_manager:
                    self._emit(prefix + line_text)
                else:
                    self.console.print(prefix + line_text)
            if add_end_separator:
                if self.layout_manager:
                    self._emit(Text.from_markup(f"    {UISymbol.VERTICAL_BAR}"))
                else:
                    self.console.print(f"    {UISymbol.VERTICAL_BAR}")
            return len(lines)

        prefix_probe = self._line_number_prefix(line_number_start)
        prefix_width = prefix_probe.cell_len
        effective_width = max(1, console_width - prefix_width - 1)

        syntax = Syntax(
            code,
            lexer_name or "text",
            theme=build_syntax_theme(self._syntax_theme_name(), suppress_error_tokens=True),
            word_wrap=True,
            padding=(0, 1),
        )
        options = self.console.options.update(width=effective_width)
        lines = self.console.render_lines(syntax, options)

        consumed = 0
        for i, line_segments in enumerate(lines):
            current_no = int(line_number_start) + i
            prefix = self._line_number_prefix(current_no)
            line_text = Text.assemble(*((seg.text, seg.style) for seg in line_segments))
            if dim:
                line_text.stylize(DisplayStyle.DIM)
            if self.layout_manager:
                self._emit(prefix + line_text)
            else:
                self.console.print(prefix + line_text)
            consumed += 1

        if add_end_separator:
            if self.layout_manager:
                self._emit(Text.from_markup(f"    {UISymbol.VERTICAL_BAR}"))
            else:
                self.console.print(f"    {UISymbol.VERTICAL_BAR}")

        return consumed

    def print_json_line(
        self, text: str, line_number: int, is_code: bool = False, *, dim: bool = False
    ) -> int:
        if is_code:
            output = f"[{DisplayStyle.DIM}]{line_number:3d}[/{DisplayStyle.DIM}] {UISymbol.VERTICAL_BAR} [{DisplayStyle.CYAN}]{escape(text)}[/{DisplayStyle.CYAN}]"
            if self.layout_manager:
                rich_text = Text.from_markup(output)
                if dim:
                    rich_text.stylize(DisplayStyle.DIM)
                self._emit(rich_text)
            else:
                rich_text = Text.from_markup(output)
                if dim:
                    rich_text.stylize(DisplayStyle.DIM)
                self.console.print(rich_text)
            return 1

        console_width = self._calc_left_panel_width()

        prefix_text = self._line_number_prefix(line_number)
        prefix_width = prefix_text.cell_len
        # 再保守扣 1，避免长字符串贴边导致软换行
        effective_width = console_width - prefix_width - 3

        content_style = DisplayStyle.DIM if dim else "default"
        rich_text = Text.from_markup(
            f"[{content_style}]{escape(text)}[/{content_style}]"
        )
        wrapped_lines = wrap_rich_text_with_display_width(
            rich_text, effective_width, "", self.console
        )

        if not wrapped_lines:
            if self.layout_manager:
                self._emit(prefix_text)
            else:
                self.console.print(prefix_text)
            return 1

        # 注意：不要用 from_markup 拼接前缀（很容易漏 ']' 触发 MarkupError）
        # 这里直接用 Text(style=...)，避免任何 markup 解析问题。
        prefix_text.stylize(DisplayStyle.DIM, 0, len(prefix_text))
        if self.layout_manager:
            for line in wrapped_lines:
                self._emit(prefix_text + line)
        else:
            for line in wrapped_lines:
                self.console.print(prefix_text + line)
        return 1

    def print_json_structure_line(
        self, line: str, line_number: int, *, dim: bool = False
    ) -> int:
        content_style = DisplayStyle.DIM if dim else "default"
        output = f"[{DisplayStyle.DIM}]{line_number:3d}[/{DisplayStyle.DIM}] {UISymbol.VERTICAL_BAR} [{content_style}]{escape(line)}[/{content_style}]"
        if self.layout_manager:
            rich_text = Text.from_markup(output)
            self._emit(rich_text)
        else:
            rich_text = Text.from_markup(output)
            self.console.print(rich_text)
        return 1

    def _process_list_style(self, text: Text) -> Tuple[bool, Text, str]:
        plain = text.plain
        if match := re.match(r"^(\s*)([*\-+])\s+(.*)", plain):
            indent, _bullet, content = match.groups()
            styled_content = apply_basic_markdown(content)
            new_text = (
                Text(indent)
                + Text(UISymbol.BULLET, style=DisplayStyle.DIM)
                + Text(" ")
                + styled_content
            )
            indent_str = indent + " " * (len(f"{UISymbol.BULLET} ") + 1)
            return True, new_text, indent_str

        if match := re.match(r"^(\s*)(\d+\.)\s+(.*)", plain):
            indent, bullet, content = match.groups()
            styled_content = apply_basic_markdown(content)
            new_text = (
                Text(indent)
                + Text(bullet, style=DisplayStyle.DIM)
                + Text(" ")
                + styled_content
            )
            indent_str = indent + " " * (len(bullet) + 1)
            return True, new_text, indent_str

        return False, text, ""
