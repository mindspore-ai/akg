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

from typing import TYPE_CHECKING, Any, Callable, Optional

import logging

from rich.console import Console
from rich.text import Text

from akg_agents.cli.constants import DisplayStyle
from akg_agents.cli.utils.i18n import t

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from akg_agents.cli.messages import (
        LLMStreamMessage,
        DisplayMessage,
        PanelDataMessage,
        AgentHeaderMessage,
        ToolStartMessage,
        ToolResultMessage,
    )

_TOOL_ICONS = {
    "read_file": "→",
    "write_file": "←",
    "bash": "$",
    "search": "✱",
    "web_search": "◈",
    "finish": "■",
    "ask_user": "?",
    "plan": "#",
}
_DEFAULT_TOOL_ICON = "⚙"

_THINKING_FLUSH_THRESHOLD = 100


class AKGConsole:
    """AKG CLI 控制台输出 - opencode 树形风格"""

    def __init__(self, console: Console, use_stream: bool = False) -> None:
        self._console = console
        self.use_stream = use_stream

        # 状态
        self.current_agent = ""
        self.current_model = ""
        self.llm_running = False
        self.op_name = ""

        # 取消处理器
        self._cancel_handler: Optional[Callable[[str], None]] = None
        self._runner_ref = None

        # opencode 风格状态
        self._in_think_block = False
        self._thinking_buffer = ""
        self._in_content_block = False
        self._content_buffer = ""
        self._current_tool: Optional[str] = None

    def set_cancel_handler(self, handler: Callable[[str], None]) -> None:
        self._cancel_handler = handler

    def set_runner_ref(self, runner) -> None:
        self._runner_ref = runner

    def print(self, *args: Any, **kwargs: Any) -> None:
        try:
            self._console.print(*args, **kwargs)
        except Exception as e:
            logger.debug("[AKGConsole] print failed", exc_info=e)

    def print_user_input(self, user_input: str) -> None:
        output = f"\n[{DisplayStyle.BOLD}]{t('presenter.user_request')}[/{DisplayStyle.BOLD}] {user_input}\n"
        try:
            self._console.print(Text.from_markup(output))
        except Exception:
            self._console.print(output, markup=True)

    def reset_state(self) -> None:
        self.current_agent = ""
        self.current_model = ""
        self.llm_running = False
        self.op_name = ""
        self._in_think_block = False
        self._thinking_buffer = ""
        self._in_content_block = False
        self._content_buffer = ""
        self._current_tool = None

    # ==================== opencode 树形渲染 ====================

    def _open_think_block(self) -> None:
        if self._in_think_block:
            return
        self._in_think_block = True
        self._thinking_buffer = ""
        txt = Text("  ┃  Thinking:", style="dim italic")
        self._console.print(txt)

    def _write_thinking_chunk(self, chunk: str) -> None:
        if not chunk:
            return
        if not self._in_think_block:
            self._open_think_block()

        self._thinking_buffer += chunk

        while "\n" in self._thinking_buffer:
            line, self._thinking_buffer = self._thinking_buffer.split("\n", 1)
            line = line.strip()
            if line:
                self._print_thinking_line(line)

        if len(self._thinking_buffer) >= _THINKING_FLUSH_THRESHOLD:
            self._flush_thinking()

    def _print_thinking_line(self, text: str) -> None:
        """打印一行 thinking，自动按终端宽度折行并保持 ┃ 前缀"""
        prefix = "  ┃  "
        max_width = max((self._console.width or 80) - len(prefix), 20)
        while len(text) > max_width:
            self._console.print(Text(f"{prefix}{text[:max_width]}", style="dim"))
            text = text[max_width:]
        if text:
            self._console.print(Text(f"{prefix}{text}", style="dim"))

    def _flush_thinking(self) -> None:
        text = self._thinking_buffer.strip()
        if not text:
            return
        self._thinking_buffer = ""
        self._print_thinking_line(text)

    def _close_think_block(self) -> None:
        if not self._in_think_block:
            return
        self._flush_thinking()
        self._in_think_block = False
        self._console.print()

    def _flush_content(self) -> None:
        text = self._content_buffer.rstrip()
        if not text:
            self._content_buffer = ""
            self._in_content_block = False
            return
        self._content_buffer = ""
        self._in_content_block = False
        for line in text.split("\n"):
            line = line.rstrip()
            if line:
                self._console.print(Text(f"  {line}"))
            else:
                self._console.print()

    def _flush_all_streams(self) -> None:
        if self._in_think_block:
            self._close_think_block()
        if self._in_content_block:
            self._flush_content()

    # ==================== 消息处理 ====================

    def on_agent_header(self, message: "AgentHeaderMessage") -> None:
        agent_name = getattr(message, "agent_name", "") or ""
        model_name = getattr(message, "model_name", "") or ""
        self._flush_all_streams()

        if model_name:
            self._console.print(Text(f"\n  {model_name}", style="dim"))
        self._console.print(Text(f"  └─ {agent_name}", style="bold cyan"))
        self._console.print()

    def on_tool_start(self, message: "ToolStartMessage") -> None:
        self._flush_all_streams()
        tool_name = getattr(message, "tool_name", "") or ""
        input_params = getattr(message, "input_params", "") or ""
        self._current_tool = tool_name

        icon = _TOOL_ICONS.get(tool_name, _DEFAULT_TOOL_ICON)
        line = Text()
        line.append(f"  {icon} ", style="yellow")
        line.append(tool_name, style="bold yellow")
        if input_params:
            line.append(f" {input_params}", style="dim")
        self._console.print(line)

    def on_tool_result(self, message: "ToolResultMessage") -> None:
        tool_name = getattr(message, "tool_name", "") or ""
        success = getattr(message, "success", True)
        duration_s = getattr(message, "duration_s", 0.0)
        self._current_tool = None

        if success:
            line = Text()
            line.append("  └─ ", style="dim")
            line.append("done", style="green")
            line.append(f" {duration_s:.1f}s", style="dim")
            self._console.print(line)
        else:
            output = getattr(message, "output", "") or ""
            line = Text()
            line.append("  └─ ", style="dim")
            line.append("error", style="bold red")
            if duration_s > 0:
                line.append(f" {duration_s:.1f}s", style="dim")
            self._console.print(line)
            if output:
                err_line = Text(f"     {output[:200]}", style="red dim")
                self._console.print(err_line)
        self._console.print()

    def on_llm_stream(self, message: "LLMStreamMessage") -> None:
        chunk = str(getattr(message, "chunk", "") or "")
        if not chunk:
            return
        is_reasoning = bool(getattr(message, "is_reasoning", False))

        if is_reasoning:
            self._write_thinking_chunk(chunk)
        elif self.use_stream:
            if not self._in_content_block:
                self._in_content_block = True
                self._content_buffer = ""
            self._content_buffer += chunk

    def on_panel_data(self, message: "PanelDataMessage") -> None:
        pass

    def on_display_message(self, message: "DisplayMessage") -> None:
        text = str(getattr(message, "text", "") or "")
        self._flush_all_streams()
        if not text:
            return
        for line in text.split("\n"):
            line = line.rstrip()
            if line:
                self._console.print(Text(f"  {line}"))
            else:
                self._console.print()
