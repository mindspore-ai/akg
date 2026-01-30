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
from rich.panel import Panel
from rich.box import ROUNDED

from akg_agents.cli.constants import DisplayStyle, SyntaxLanguage
from akg_agents.cli.stream import StreamRenderer
from akg_agents.cli.utils.i18n import t

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from akg_agents.cli.messages import (
        LLMStreamMessage,
        DisplayMessage,
        PanelDataMessage,
    )


class AKGConsole:
    """AKG CLI 控制台输出 - 统一所有输出接口，禁止直接使用原始 console"""

    def __init__(self, console: Console, use_stream: bool = False) -> None:
        self._console = console  # 私有化，禁止直接访问
        self.use_stream = use_stream
        self.stream_renderer = StreamRenderer(console)

        # 状态
        self.current_agent = ""
        self.current_model = ""
        self.llm_running = False
        self.op_name = ""
        self._stream_started = False

        # 取消处理器
        self._cancel_handler: Optional[Callable[[str], None]] = None

        # Runner 引用（保留用于扩展）
        self._runner_ref = None

    def set_cancel_handler(self, handler: Callable[[str], None]) -> None:
        """设置取消处理器"""
        self._cancel_handler = handler

    def set_runner_ref(self, runner) -> None:
        """设置 runner 引用（用于更新面板数据）"""
        self._runner_ref = runner

    def print(self, *args: Any, **kwargs: Any) -> None:
        """统一的输出接口 - 所有输出必须通过此方法"""
        try:
            self._console.print(*args, **kwargs)
        except Exception as e:
            logger.debug("[AKGConsole] print failed", exc_info=e)

    def _write(self, content: Any) -> None:
        """统一写入接口"""
        try:
            # Panel 和其他 rich renderable 对象直接打印
            if hasattr(content, "__rich__") or hasattr(content, "__rich_console__"):
                self._console.print(content)
            elif isinstance(content, Text):
                self._console.print(content)
            elif isinstance(content, str):
                try:
                    ansi_text = Text.from_ansi(content)
                    self._console.print(ansi_text)
                except Exception:
                    try:
                        self._console.print(content, markup=True)
                    except Exception:
                        self._console.print(content, markup=False)
            else:
                self._console.print(content)
        except Exception as e:
            logger.debug("[AKGConsole] _write failed", exc_info=e)

    def print_user_input(self, user_input: str) -> None:
        """打印用户输入"""
        output = f"\n[{DisplayStyle.BOLD}]{t('presenter.user_request')}[/{DisplayStyle.BOLD}] {user_input}\n"
        self._write(Text.from_markup(output))

    def reset_state(self) -> None:
        """重置状态"""
        self.current_agent = ""
        self.current_model = ""
        self.llm_running = False
        self.op_name = ""
        try:
            self.stream_renderer.full_buffer = ""
            self.stream_renderer.rendered_length = 0
            self.stream_renderer.agent_name = ""
            self.stream_renderer.op_name = ""
            self.stream_renderer.last_update_time = 0
            try:
                self.stream_renderer.state.reset()
            except Exception:
                pass
        except Exception as e:
            logger.debug("[AKGConsole] reset stream_renderer failed", exc_info=e)
        self._stream_started = False

    def on_llm_stream(self, message: "LLMStreamMessage") -> None:
        """处理 LLM 流式消息"""
        if self.use_stream:
            chunk = str(getattr(message, "chunk", "") or "")
            try:
                is_reasoning = bool(getattr(message, "is_reasoning", False))
                if not self._stream_started:
                    self.stream_renderer.start(
                        agent_name="llm",
                        language=SyntaxLanguage.TEXT,
                        op_name=self.op_name,
                    )
                    self._stream_started = True
                self.stream_renderer.add_chunk(chunk, is_reasoning=is_reasoning)
            except Exception as e:
                logger.warning("[AKGConsole] stream_renderer.add_chunk failed", exc_info=e)

    def on_panel_data(self, message: "PanelDataMessage") -> None:
        """处理面板数据更新消息（已移除面板，仅保留接口兼容性）"""
        # 面板已移除，以后通过斜杠命令显示数据
        pass

    def on_display_message(self, message: "DisplayMessage") -> None:
        """处理显示消息"""
        text = str(getattr(message, "text", "") or "")
        if self.use_stream and self._stream_started:
            try:
                self.stream_renderer.finish()
            except Exception as e:
                logger.warning("[AKGConsole] stream_renderer.finish failed", exc_info=e)
            self._stream_started = False
        if not text:
            return
        try:
            ansi_text = Text.from_ansi(text)
            # 使用 Panel 添加精美边框
            panel = Panel(
                ansi_text,
                box=ROUNDED,
                border_style="dim",
                padding=(0, 1),
            )
            self._write(panel)
        except Exception:
            try:
                # 如果解析失败，尝试直接使用文本创建 Panel
                panel = Panel(
                    text,
                    box=ROUNDED,
                    border_style="dim",
                    padding=(0, 1),
                )
                self._write(panel)
            except Exception:
                # 最后的回退方案
                self._write(" " + text)
