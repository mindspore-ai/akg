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

import asyncio
import contextlib
import os
import sys
from datetime import datetime
from typing import Optional, List

import typer
from prompt_toolkit import Application
from prompt_toolkit.application.current import get_app
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.filters import Condition
from prompt_toolkit.layout import (
    Layout,
    DynamicContainer,
    Window,
    HSplit,
    ConditionalContainer,
    FloatContainer,
    Float,
    Dimension,
)
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.widgets import Frame
from prompt_toolkit.styles import Style
from prompt_toolkit.utils import get_cwidth
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.patch_stdout import patch_stdout
import logging

from rich.text import Text
from rich.panel import Panel
from rich.box import ROUNDED

log = logging.getLogger(__name__)

from ai_kernel_generator.cli.service import (
    CLIAppServices,
    validate_target_config,
)
from ai_kernel_generator.cli.constants import DisplayStyle
from ai_kernel_generator.cli.runtime import LocalExecutor
from ai_kernel_generator.cli.utils.i18n import t
from ai_kernel_generator.cli.ui.plugin_registry import get_default_plugin, register_plugin
from ai_kernel_generator.cli.ui.plugins.kernel_impl_list import KernelImplListPlugin
from .types import ResolvedTargetConfig


class InteractiveOpRunner:
    def __init__(
        self,
        *,
        console,
        services: CLIAppServices,
        cli: LocalExecutor,
        auto_yes: bool,
        target: ResolvedTargetConfig,
        intent: Optional[str],
        rag: bool = False,
        output_path: Optional[str] = None,
        device_ids: Optional[List[int]] = None,
    ):
        self.console = console
        self.services = services
        self.cli = cli
        self.auto_yes = auto_yes
        self.target = target
        self.intent = intent
        self.rag = rag
        self.output_path = output_path
        self.device_ids = device_ids
        self._panel_visible = True
        self._panel_footer = ""

        # 注册 Kernel Impl List 插件
        kernel_impl_plugin = KernelImplListPlugin()
        register_plugin(kernel_impl_plugin)
        self._panel_plugin = kernel_impl_plugin

        # 设置 console 的 runner 引用（用于更新面板数据）
        if hasattr(self.console, 'set_runner_ref'):
            self.console.set_runner_ref(self)

    def _get_panel_mode(self) -> str:
        return (os.getenv("AKG_CLI_PANEL") or "llm_start").strip().lower()

    def _get_panel_lines(self, width: int) -> tuple[str, list[str]]:
        source = getattr(self.cli, "console", self.console)
        panel_mode = self._get_panel_mode()
        if panel_mode in ["time", "clock"]:
            title = "时间"
            lines = [datetime.now().strftime("当前时间: %Y-%m-%d %H:%M:%S")]
        elif panel_mode in ["node_start", "node"]:
            title = "节点启动"
            lines = list(getattr(source, "node_start_lines", []))
            if not lines:
                lines = ["等待节点启动..."]
        else:
            title = "模型调用"
            lines = list(getattr(source, "llm_start_lines", []))
            if not lines:
                lines = ["等待模型调用..."]
        lines = lines[-10:]
        if width > 0:
            lines = [line[:width] for line in lines]
            title = title[:width]
        return title, lines

    def _render_panel_text(self, width: int, footer: str | None = None) -> str:
        if not self._panel_visible:
            return ""
        title, lines = self._get_panel_lines(width)
        sep = "─" * max(1, width)
        header = title or ""
        parts = [sep, header] + lines + [sep]
        footer_text = self._panel_footer if footer is None else footer
        if footer_text:
            parts.append(footer_text[:width] if width > 0 else footer_text)
        return "\n".join(parts)

    def _render_panel_fragments(self, width: int, footer: str | None = None):
        if not self._panel_visible:
            return [("", "")]
        
        # 使用插件渲染
        plugin = get_default_plugin()
        if plugin:
            fragments = plugin.render_fragments(max(1, width))
        else:
            # 回退到原有逻辑
            title, lines = self._get_panel_lines(width)
            sep = "─" * max(1, width)
            fragments = [
                ("class:panel.separator", sep + "\n"),
                ("class:panel.title", (title or "") + "\n"),
            ]
            for line in lines:
                fragments.append(("class:panel.body", line + "\n"))
            fragments.append(("class:panel.separator", sep))
        
        # 添加 footer（如果有）
        footer_text = self._panel_footer if footer is None else footer
        if footer_text:
            fragments.append(("", "\n"))
            fragments.append(("class:panel.footer", footer_text[:width] if width > 0 else footer_text))
        
        return fragments

    def _update_panel_data(self, action: str, data: dict) -> None:
        """更新面板插件数据"""
        if self._panel_plugin:
            self._panel_plugin.on_data_update({
                "action": action,
                "data": data
            })

    def _build_resolved(self) -> dict:
        return {
            "framework": self.target.framework,
            "backend": self.target.backend,
            "arch": self.target.arch,
            "dsl": self.target.dsl,
        }

    def _validate_config_or_exit(self, resolved: dict) -> None:
        errs = validate_target_config(
            resolved.get("framework", ""),
            resolved.get("backend", ""),
            resolved.get("arch", ""),
            resolved.get("dsl", ""),
        )
        if errs:
            self.console.print(
                f"[{DisplayStyle.BOLD_RED}]{t('ops.error.config_invalid')}[/{DisplayStyle.BOLD_RED}]\n- "
                + "\n- ".join(errs)
            )
            raise typer.Exit(code=2)

    async def _ensure_worker_or_exit(self, resolved: dict) -> None:
        """确保 worker 已注册到本地 WorkerManager（用于运行时检查）"""
        from ai_kernel_generator.core.worker.manager import (
            get_worker_manager,
            register_remote_worker,
            register_local_worker,
        )

        backend = resolved.get("backend", "")
        arch = resolved.get("arch", "")

        # 检查是否已有匹配的 worker
        has_worker = await get_worker_manager().has_worker(backend=backend, arch=arch)
        if has_worker:
            return

        # 如果有 device_ids（即使用了 --devices），注册本地 worker
        if self.device_ids:
            try:
                await register_local_worker(
                    device_ids=self.device_ids,
                    backend=backend,
                    arch=arch,
                )
                log.info(
                    f"[OpRunner] Registered local worker to local WorkerManager: device_ids={self.device_ids}, backend={backend}, arch={arch}"
                )
            except Exception as e:
                log.warning(
                    f"[OpRunner] Failed to register local worker to local WorkerManager: device_ids={self.device_ids}",
                    exc_info=e,
                )
                # 不抛出异常，允许继续执行（可能 server 端有 worker）

        # 如果有 worker_url，注册到本地 WorkerManager
        if self.services.worker_service.workers:
            for worker_url in self.services.worker_service.workers:
                try:
                    await register_remote_worker(
                        backend=backend,
                        arch=arch,
                        worker_url=worker_url,
                        capacity=None,  # 自动查询
                        tags=None,
                    )
                    log.info(
                        f"[OpRunner] Registered worker to local WorkerManager: {worker_url}"
                    )
                except Exception as e:
                    log.warning(
                        f"[OpRunner] Failed to register worker to local WorkerManager: {worker_url}",
                        exc_info=e,
                    )
                    # 不抛出异常，允许继续执行（可能 server 端有 worker）

    async def _execute_main_agent(
        self,
        *,
        resolved: dict,
        user_input: str,
    ) -> dict:
        """执行主代理操作"""
        res = await self.cli.execute_main_agent(
            user_input=user_input,
            framework=resolved.get("framework", ""),
            backend=resolved.get("backend", ""),
            arch=resolved.get("arch", ""),
            dsl=resolved.get("dsl", ""),
            rag=self.rag,
            output_path=self.output_path,
        )
        return res

    class _TuiState:
        def __init__(self) -> None:
            self.is_generating = False
            self.loading_text = "运行中..."

    async def _run_tui_app(
        self,
        *,
        resolved: dict,
        history_file: str,
        initial_input: str,
    ) -> None:
        loop = asyncio.get_running_loop()
        state = self._TuiState()
        input_queue: asyncio.Queue[str] = asyncio.Queue()
        should_exit = asyncio.Event()
        current_agent_task: dict[str, asyncio.Task | None] = {"task": None}

        def _put_input(text: str) -> None:
            if not text.strip():
                return
            state.is_generating = True
            state.loading_text = "运行中..."
            app.invalidate()
            input_queue.put_nowait(text)

        def _accept_input(buff: Buffer) -> None:
            text = buff.text
            buff.text = ""
            _put_input(text)

        buffer = Buffer(
            history=FileHistory(history_file),
            auto_suggest=AutoSuggestFromHistory(),
            multiline=True,
            accept_handler=_accept_input,
        )

        kb = KeyBindings()

        @kb.add(Keys.Enter, eager=True)
        def _(event):
            if state.is_generating:
                return
            if not event.current_buffer.text.strip():
                return
            event.current_buffer.validate_and_handle()

        @kb.add("c-j")
        def _(event):
            if state.is_generating:
                return
            event.current_buffer.insert_text("\n")

        @kb.add("c-c")
        def _(event):
            if state.is_generating and current_agent_task["task"] is not None:
                loop.call_soon_threadsafe(
                    lambda: asyncio.create_task(
                        self.cli.cancel_main_agent(reason="cancelled by ctrl+c")
                    )
                )
                current_agent_task["task"].cancel()
                return
            should_exit.set()
            event.app.exit()

        @kb.add("f2")
        def _(event):
            self._panel_visible = not self._panel_visible
            event.app.invalidate()

        def _panel_text():
            if not self._panel_visible:
                return [("", "")]
            try:
                width = get_app().output.get_size().columns
            except Exception:
                width = 80
            return self._render_panel_fragments(max(1, width))

        # 检查是否是 KernelImplListPlugin，如果是则使用 Frame 方式
        plugin = get_default_plugin()
        is_kernel_impl_plugin = (
            plugin is not None
            and hasattr(plugin, "render_current_task_fragments")
            and hasattr(plugin, "render_history_fragments")
        )

        if is_kernel_impl_plugin:
            # 为 KernelImplListPlugin 使用 Frame 方式
            def _current_task_text():
                if not self._panel_visible:
                    return [("", "")]
                try:
                    width = get_app().output.get_size().columns
                except Exception:
                    width = 80
                # Frame 有左右边框，内侧再留一些空间
                inner_width = max(1, width - 4)
                return plugin.render_current_task_fragments(inner_width)

            def _history_text():
                if not self._panel_visible:
                    return [("", "")]
                try:
                    width = get_app().output.get_size().columns
                except Exception:
                    width = 80
                # Frame 有左右边框，内侧再留一些空间
                inner_width = max(1, width - 4)
                return plugin.render_history_fragments(inner_width)

            current_task_window = Window(
                FormattedTextControl(_current_task_text),
                wrap_lines=False,
                dont_extend_height=True,
                style="class:panel",
            )
            current_task_frame = Frame(
                body=current_task_window,
                title="当前任务",
                style="class:panel-frame",
            )

            history_window = Window(
                FormattedTextControl(_history_text),
                wrap_lines=False,
                dont_extend_height=True,
                style="class:panel",
            )
            history_frame = Frame(
                body=history_window,
                title="实现历史 Top 5",
                style="class:panel-frame",
            )

            panel_container = ConditionalContainer(
                content=HSplit([current_task_frame, history_frame]),
                filter=Condition(lambda: self._panel_visible),
            )
        else:
            # 使用原有的方式
            panel_container = ConditionalContainer(
                content=Window(
                    FormattedTextControl(_panel_text),
                    wrap_lines=False,
                    dont_extend_height=True,
                    style="class:panel",
                ),
                filter=Condition(lambda: self._panel_visible),
            )

        input_body = Window(
            BufferControl(buffer=buffer),
            height=Dimension(min=1, max=10),
            wrap_lines=True,
        )
        placeholder = FormattedTextControl(
            lambda: "F2 切换面板  Ctrl+C 退出" if not buffer.text else "",
        )
        input_container = Frame(
            body=FloatContainer(
                content=input_body,
                floats=[
                    Float(
                        content=Window(
                            placeholder,
                            height=1,
                            dont_extend_height=True,
                            style="class:placeholder",
                        ),
                        left=1,
                        top=0,
                    )
                ],
            ),
            title="输入",
            style="class:input-frame",
        )

        def _loading_line():
            hint = "F2 切换面板  Ctrl+C 终止"
            try:
                width = get_app().output.get_size().columns
            except Exception:
                width = 80
            # Frame 有左右边框，内侧再留 1 格空隙
            inner_width = max(1, width - 4)
            left = state.loading_text or ""
            left_width = get_cwidth(left)
            hint_width = get_cwidth(hint)
            if inner_width <= hint_width:
                return [("class:placeholder", hint[:inner_width])]
            max_left = inner_width - hint_width - 1
            if left_width > max_left:
                if max_left <= 3:
                    left = left[:max_left]
                else:
                    left = left[: max_left - 3] + "..."
            gap = inner_width - get_cwidth(left) - hint_width
            return [("", left + (" " * max(1, gap))), ("class:placeholder", hint)]

        loading_body = Window(
            FormattedTextControl(_loading_line),
            height=1,
            dont_extend_height=True,
        )
        loading_container = Frame(
            body=loading_body,
            title="运行中",
            height=3,
            style="class:input-frame",
        )

        container = HSplit(
            [
                panel_container,
                DynamicContainer(
                    lambda: input_container
                    if not state.is_generating
                    else loading_container
                ),
            ]
        )

        app = Application(
            layout=Layout(container),
            key_bindings=kb,
            full_screen=False,
            style=Style.from_dict(
                {
                    "input-frame": "fg:#4aa3ff",
                    "placeholder": "italic fg:#7a7a7a",
                    "panel": "fg:#c8d0d8",
                    "panel-frame": "fg:#4aa3ff",
                    "panel.separator": "fg:#3f5366",
                    "panel.title": "bold fg:#7dd3fc",
                    "panel.body": "fg:#c8d0d8",
                    "panel.footer": "italic fg:#6b7280",
                }
            ),
            erase_when_done=True,
        )

        # prompt_toolkit：重复调用 app.exit() 会报错：
        #   "Return value already set. Application.exit() failed."
        # 用一个本地 guard 保证只调用一次。
        _did_app_exit = False

        def _safe_app_exit() -> None:
            nonlocal _did_app_exit
            if _did_app_exit:
                return
            _did_app_exit = True
            try:
                app.exit()
            except Exception:
                # 退出路径中的重复调用/竞态不应导致 CLI 崩溃
                return

        async def _spinner(stop_event: asyncio.Event) -> None:
            frames = ["-", "\\", "|", "/"]
            idx = 0
            while not stop_event.is_set():
                state.loading_text = f"运行中... {frames[idx % len(frames)]}"
                idx += 1
                app.invalidate()
                await asyncio.sleep(0.2)

        async def _process_input(user_input: str) -> bool:
            user_input = (user_input or "").strip()
            if not user_input:
                state.is_generating = False
                app.invalidate()
                return True

            self.cli.console.print_user_input(user_input)

            stop_event = asyncio.Event()
            spinner_task = asyncio.create_task(_spinner(stop_event))
            try:
                current_agent_task["task"] = asyncio.create_task(
                    self._execute_main_agent(
                        resolved=resolved, user_input=user_input
                    )
                )
                try:
                    state_result = await current_agent_task["task"]
                except asyncio.CancelledError:
                    state_result = {
                        "current_step": "cancelled_by_user",
                        "should_continue": True,
                        "display_message": "⚠️ 操作已被用户取消",
                    }
            finally:
                stop_event.set()
                spinner_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await spinner_task
                current_agent_task["task"] = None
                state.is_generating = False
                state.loading_text = ""
                app.invalidate()

            self._render_agent_messages(state_result)

            if not isinstance(state_result, dict):
                return False

            current_step = str(state_result.get("current_step", "")).strip().lower()
            if current_step in ["cancelled_by_user", "cancelled"]:
                current_step = ""
                state_result["should_continue"] = True
            if not state_result.get("should_continue", True) or current_step in [
                "error"
            ]:
                return False
            return True

        async def _workflow_loop() -> None:
            if initial_input:
                state.is_generating = True
                state.loading_text = "运行中..."
                app.invalidate()
                cont = await _process_input(initial_input)
                if not cont:
                    should_exit.set()
                    _safe_app_exit()
                    return
            while not should_exit.is_set():
                user_input = await input_queue.get()
                cont = await _process_input(user_input)
                if not cont:
                    should_exit.set()
                    _safe_app_exit()
                    return

        app_task = asyncio.create_task(app.run_async())
        workflow_task = asyncio.create_task(_workflow_loop())
        done, _ = await asyncio.wait(
            {app_task, workflow_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if workflow_task in done and not app_task.done():
            _safe_app_exit()
        await app_task
        if not workflow_task.done():
            workflow_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await workflow_task

    async def _run_workflow(self) -> None:
        """CLI 模式下的工作流"""
        resolved = self._build_resolved()

        self._validate_config_or_exit(resolved)
        # 确保 worker 已注册到本地 WorkerManager（用于运行时检查）
        try:
            await self._ensure_worker_or_exit(resolved)
        except Exception as e:
            log.warning(
                "[OpRunner] _ensure_worker_or_exit failed; continue", exc_info=e
            )

        history_file = os.path.expanduser("~/.akg_cli_history")
        initial_input = (self.intent or "").strip()

        await self._run_tui_app(
            resolved=resolved,
            history_file=history_file,
            initial_input=initial_input,
        )

    def _render_agent_messages(self, state: dict) -> None:
        """渲染代理消息"""
        if not isinstance(state, dict):
            return

        display_message = str(state.get("display_message") or "").rstrip()
        if display_message:
            # 使用 Text.from_ansi() 正确处理包含 ANSI 转义序列的字符串
            try:
                ansi_text = Text.from_ansi(display_message)
                # 使用 Panel 添加精美边框
                panel = Panel(
                    ansi_text,
                    box=ROUNDED,
                    border_style="dim",
                    padding=(0, 1),
                )
                self.console.print(panel)
            except Exception:
                # 如果解析失败，回退到直接打印
                try:
                    panel = Panel(
                        display_message,
                        box=ROUNDED,
                        border_style="dim",
                        padding=(0, 1),
                    )
                    self.console.print(panel, markup=False)
                except Exception:
                    # 最后的回退方案
                    self.console.print(" " + display_message, markup=False)

        hint_message = str(state.get("hint_message") or "").rstrip()
        if hint_message:
            # 使用 Text.from_ansi() 正确处理包含 ANSI 转义序列的字符串
            try:
                ansi_text = Text.from_ansi(hint_message)
                # 对整个文本应用 dim 样式
                ansi_text.stylize(DisplayStyle.DIM, 0, len(ansi_text))
                self.console.print(ansi_text)
            except Exception:
                # 如果解析失败，回退到 Rich markup
                self.console.print(
                    f"[{DisplayStyle.DIM}]{hint_message}[/{DisplayStyle.DIM}]"
                )

        current_step = str(state.get("current_step", "")).strip().lower()
        if current_step == "error":
            error_message = str(
                state.get("error")
                or state.get("error_message")
                or state.get("message")
                or ""
            ).rstrip()
            if error_message:
                self.console.print(
                    f"[{DisplayStyle.BOLD_RED}]错误:[/{DisplayStyle.BOLD_RED}] {error_message}"
                )

    def run(self) -> None:
        """运行"""

        async def _workflow() -> None:
            await self._run_workflow()

        with patch_stdout(raw=True):
            try:
                if hasattr(self.console, "_console"):
                    self.console._console.file = sys.stdout
                if hasattr(self.cli, "console") and hasattr(self.cli.console, "_console"):
                    self.cli.console._console.file = sys.stdout
            except Exception:
                pass
            try:
                asyncio.run(_workflow())
            except KeyboardInterrupt:
                self.console.print(
                    f"\n[{DisplayStyle.YELLOW}]已退出[/{DisplayStyle.YELLOW}]"
                )
            except Exception as e:
                self.console.print(
                    f"[{DisplayStyle.RED}]错误:[/{DisplayStyle.RED}] {e}"
                )
                raise typer.Exit(code=1)
