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

import logging

from rich.console import Console
from rich.markup import escape
from rich.syntax import Syntax
from rich.text import Text

from akg_agents.cli.constants import DisplayStyle, UISymbol

logger = logging.getLogger(__name__)
from akg_agents.cli.utils.syntax_utils import build_syntax_theme
from akg_agents.cli.utils.text_utils import (
    apply_basic_markdown,
    wrap_rich_text_with_display_width,
)


class ConsolePainter:
    """负责将内容绘制到 Rich Console 或分屏布局。"""

    def __init__(
        self,
        console: Console,
    ):
        self.console = console

    def _emit(self, content) -> None:
        # CLI 模式下直接输出到 console
        try:
            # 如果 content 是 Text 对象，直接打印（Rich 会自动处理样式）
            if isinstance(content, Text):
                self.console.print(content)
            # 如果 content 是字符串且可能包含 ANSI 转义序列，尝试解析
            elif isinstance(content, str):
                try:
                    ansi_text = Text.from_ansi(content)
                    self.console.print(ansi_text)
                except Exception:
                    # 如果解析失败，尝试作为 markdown 处理
                    try:
                        self.console.print(content, markup=True)
                    except Exception:
                        # 最后回退到纯文本
                        self.console.print(content, markup=False)
            else:
                self.console.print(content)
        except Exception as e:
            logger.debug("[Painter] console.print failed", exc_info=e)

    def _calc_left_panel_width(self) -> int:
        console_width = self.console.width or 80
        # CLI 模式下直接使用 console 宽度，保守减 2，避免刚好超宽触发软换行
        return max(1, console_width - 2)

    def _separator_text(self, style: str) -> Text:
        width = self._calc_left_panel_width()
        return Text("─" * width, style=style)

    def _syntax_theme_name(self) -> str:
        return "monokai"

    def print_divider(self, title: str = "") -> None:
        if title:
            self._emit(
                Text.from_markup(f"\n[{DisplayStyle.DIM}]{title}[/{DisplayStyle.DIM}]")
            )
        self._emit(self._separator_text(DisplayStyle.DIM))

    def print_finish_mark(self, agent_name: str) -> None:
        self._emit(self._separator_text(DisplayStyle.GREEN))
        self.console.print(
            f"[{DisplayStyle.BOLD_GREEN}] {agent_name} - 完成[/{DisplayStyle.BOLD_GREEN}]\n"
        )

    def _line_number_prefix(self, line_number: int) -> Text:
        return Text.from_markup(
            f"[{DisplayStyle.DIM}]{int(line_number):3d}[/{DisplayStyle.DIM}] {UISymbol.VERTICAL_BAR} "
        )

    def print_normal_line(
        self, line: str, line_number: int, *, dim: bool = False
    ) -> None:
        console_width = self._calc_left_panel_width()

        prefix_text = self._line_number_prefix(line_number)

        # 先尝试解析 ANSI 转义序列，如果失败则应用 markdown
        try:
            rich_line = Text.from_ansi(line)
        except Exception:
            rich_line = apply_basic_markdown(line)
        is_list, rich_line, indent_str = self._process_list_style(rich_line)
        if dim:
            # 对整个文本应用 dim 样式
            rich_line.stylize(DisplayStyle.DIM, 0, len(rich_line))

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
                theme=build_syntax_theme(
                    self._syntax_theme_name(), suppress_error_tokens=True
                ),
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
                self.console.print(prefix + line_text)
            if add_end_separator:
                self.console.print(f"    {UISymbol.VERTICAL_BAR}")
            return len(lines)

        prefix_probe = self._line_number_prefix(line_number_start)
        prefix_width = prefix_probe.cell_len
        effective_width = max(1, console_width - prefix_width - 1)

        syntax = Syntax(
            code,
            lexer_name or "text",
            theme=build_syntax_theme(
                self._syntax_theme_name(), suppress_error_tokens=True
            ),
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
            self.console.print(prefix + line_text)
            consumed += 1

        if add_end_separator:
            self.console.print(f"    {UISymbol.VERTICAL_BAR}")

        return consumed

    def print_json_line(
        self, text: str, line_number: int, is_code: bool = False, *, dim: bool = False
    ) -> int:
        if is_code:
            output = f"[{DisplayStyle.DIM}]{line_number:3d}[/{DisplayStyle.DIM}] {UISymbol.VERTICAL_BAR} [{DisplayStyle.CYAN}]{escape(text)}[/{DisplayStyle.CYAN}]"
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
            self.console.print(prefix_text)
            return 1

        # 注意：不要用 from_markup 拼接前缀（很容易漏 ']' 触发 MarkupError）
        # 这里直接用 Text(style=...)，避免任何 markup 解析问题。
        prefix_text.stylize(DisplayStyle.DIM, 0, len(prefix_text))
        for line in wrapped_lines:
            self.console.print(prefix_text + line)
        return 1

    def print_json_structure_line(
        self, line: str, line_number: int, *, dim: bool = False
    ) -> int:
        content_style = DisplayStyle.DIM if dim else "default"
        output = f"[{DisplayStyle.DIM}]{line_number:3d}[/{DisplayStyle.DIM}] {UISymbol.VERTICAL_BAR} [{content_style}]{escape(line)}[/{content_style}]"
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
