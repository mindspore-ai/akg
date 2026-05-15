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

import re
from rich.console import Console
from rich.text import Text
from rich.markup import escape
from akg_agents.cli.constants import DisplayStyle


def wrap_rich_text_with_display_width(
    text_obj: Text, max_width: int, subsequent_indent_str: str, console: Console
) -> list[Text]:
    """
    按显示宽度换行 rich.Text 对象，并处理后续行缩进。

    Args:
        text_obj: 要换行的 rich.Text 对象
        max_width: 最大显示宽度
        subsequent_indent_str: 后续行的缩进字符串
        console: rich.Console 实例，用于 wrapping

    Returns:
        换行后的 rich.Text 对象列表
    """
    if not text_obj.plain.strip():
        return [text_obj]

    wrapped_lines = text_obj.wrap(console, max_width)

    indented_wrapped_lines = []
    if not wrapped_lines:
        return []

    # First line
    indented_wrapped_lines.append(wrapped_lines[0])

    # Subsequent lines get indent
    subsequent_indent_text = Text(subsequent_indent_str)
    for i in range(1, len(wrapped_lines)):
        indented_wrapped_lines.append(subsequent_indent_text + wrapped_lines[i])

    return indented_wrapped_lines


def apply_basic_markdown(line: str) -> Text:
    """
    将基本 Markdown 语法转换为 rich.Text 对象。
    目前支持：
    - 粗体: **text**
    - 斜体: *text* 或 _text_
    - 行内代码: `text`
    """
    processed_line = escape(line)
    # Bold: **text** -> [bold]text[/bold]
    processed_line = re.sub(
        rf"\*\*([^\*]+?)\*\*",
        rf"[{DisplayStyle.BOLD}]\1[/{DisplayStyle.BOLD}]",
        processed_line,
    )
    # Italic: *text* (not preceded/followed by *) -> [italic]\1[/italic]
    processed_line = re.sub(
        rf"(?<!\*)\*([^\*]+?)\*(?!\*)",
        rf"[{DisplayStyle.ITALIC}]\1[/{DisplayStyle.ITALIC}]",
        processed_line,
    )
    # Italic: _text_ (not preceded/followed by _) -> [italic]\1[/italic]
    processed_line = re.sub(
        rf"(?<!\_)\_([^\_]+?)\_(?!\_)",
        rf"[{DisplayStyle.ITALIC}]\1[/{DisplayStyle.ITALIC}]",
        processed_line,
    )
    # Inline code: `text` -> [reverse]text[/reverse]
    processed_line = re.sub(
        rf"`([^`]+?)`",
        rf"[{DisplayStyle.REVERSE}]\1[/{DisplayStyle.REVERSE}]",
        processed_line,
    )
    return Text.from_markup(processed_line)
