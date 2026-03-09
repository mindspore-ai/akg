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
from typing import Optional, List

import typer
from prompt_toolkit import Application
from prompt_toolkit.application.current import get_app
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.layout import (
    Layout,
    DynamicContainer,
    Window,
    HSplit,
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

log = logging.getLogger(__name__)

from akg_agents.cli.service import (
    CLIAppServices,
    validate_target_config,
)
from akg_agents.cli.constants import DisplayStyle
from akg_agents.cli.runtime import LocalExecutor
from akg_agents.cli.utils.i18n import t
from akg_agents.cli.ui.progress_display import (
    AdaptiveSearchProgressDisplay,
    EvolveProgressState,
)
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
        task_file_content: Optional[str] = None,
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
        self.task_file_content = task_file_content
        self.rag = rag
        self.output_path = output_path
        self.device_ids = device_ids
        self.scene = "ops"

        # 🔥 命令调度器
        from akg_agents.cli.commands.slash_commands import create_dispatcher
        self._command_dispatcher = create_dispatcher()
        self._exit_printed = False
        self._should_exit = False


    def _print_session_id_on_exit(self):
        """退出时打印 session_id，方便用户后续 --resume"""
        if self._exit_printed:
            return
        self._exit_printed = True
        try:
            sid = self.cli.session_id
            self.console.print(
                f"\n[dim]Session ID: {sid}[/dim]"
                f"\n[dim]恢复命令: akg_cli resume {sid}[/dim]"
            )
        except Exception:
            pass

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
        from akg_agents.core.worker.manager import (
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
            task_file_content=self.task_file_content,
        )
        return res

    async def _handle_input(self, text: str, state, app, input_queue) -> None:
        """处理用户输入（斜杠命令或普通输入）"""
        # 🔥 检查新输入是否是阻塞命令
        is_new_blocking = True  # 默认普通输入是阻塞的
        if text.strip().startswith('/'):
            parts = text.strip()[1:].split()
            if parts:
                from akg_agents.cli.commands.slash_commands import get_registry
                cmd = get_registry().get(parts[0].lower())
                if cmd:
                    is_new_blocking = cmd.is_blocking
        
        # 🔥 如果当前有阻塞任务在运行，且新输入也是阻塞的，拒绝执行
        if state.is_generating and is_new_blocking:
            self.console.print("[yellow]⚠️  当前有阻塞任务正在执行，请等待完成或按 Ctrl+C 取消[/yellow]")
            return
        
        # 🔥 如果是阻塞命令，先设置状态
        if is_new_blocking:
            state.is_generating = True
            state.loading_text = "执行中..."
            app.invalidate()
        
        # 执行命令
        result = await self._command_dispatcher.dispatch(self, text)
        
        # 🔥 根据执行结果恢复或保持状态
        if not result.handled:
            # 不是斜杠命令，按原有逻辑处理（已经设置了 is_generating）
            input_queue.put_nowait(text)
        elif result.is_blocking:
            # 🔥 阻塞命令执行完成，恢复状态
            state.is_generating = False
            state.loading_text = ""
            app.invalidate()
        else:
            # 🔥 非阻塞命令：不影响 is_generating 状态
            pass
        
        # 检查是否需要退出
        if self._should_exit:
            app.exit()

    class _TuiState:
        def __init__(self) -> None:
            self.is_generating = False
            self.loading_text = "运行中..."
            self.progress_info: dict = {}
            self.progress_display: Optional["AdaptiveSearchProgressDisplay"] = None
            self.workflow_type: str = ""  # "search" / "evolve" / ""
            self.evolve_state: Optional[Any] = None  # EvolveProgressState
            self.reset_timer: bool = False  # spinner 收到此标记后重置计时
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
        exit_confirm = {"pressed_once": False}  # 🔥 退出确认状态

        def _accept_input(buff: Buffer) -> None:
            """处理输入提交"""
            text = buff.text
            if not text.strip():
                return
            
            # 清空 buffer（历史已由 prompt_toolkit 自动保存）
            buff.text = ""
            
            exit_confirm["pressed_once"] = False  # 🔥 重置退出确认
            
            # 🔥 立即显示用户输入（无论是斜杠命令还是普通输入）
            self.cli.console.print_user_input(text)
            
            # 🔥 快速判断：如果是斜杠命令，异步处理；否则直接入队
            if text.strip().startswith('/'):
                # 斜杠命令：异步处理（需要判断是否阻塞）
                loop.call_soon_threadsafe(
                    lambda: asyncio.create_task(self._handle_input(text, state, app, input_queue))
                )
            else:
                # 普通输入（是阻塞的）：检查是否有阻塞任务在运行
                if state.is_generating:
                    self.console.print("[yellow]⚠️  当前有阻塞任务正在执行，请等待完成或按 Ctrl+C 取消[/yellow]")
                    return
                
                # 🔥 设置 loading 状态（全局 spinner 会接管显示）
                state.is_generating = True
                app.invalidate()
                
                # 放入队列
                input_queue.put_nowait(text)

        # 🔥 新增：自动补全器
        from akg_agents.cli.ui.completers import EnhancedCompleter
        from akg_agents.cli.commands.slash_commands import get_registry
        
        buffer = Buffer(
            history=FileHistory(history_file),
            auto_suggest=AutoSuggestFromHistory(),
            completer=EnhancedCompleter(get_registry()),  # 🔥 Tab 键补全（作为备用）
            multiline=True,
            accept_handler=_accept_input,
        )

        kb = KeyBindings()

        @kb.add(Keys.Enter, eager=True)
        def _(event):
            text = event.current_buffer.text
            
            # 空输入忽略
            if not text.strip():
                return
            
            exit_confirm["pressed_once"] = False  # 🔥 重置退出确认
            
            # 🔥 手动保存到历史（因为我们覆盖了默认的 Enter 行为）
            if event.current_buffer.history:
                event.current_buffer.history.append_string(text)
            
            event.current_buffer.validate_and_handle()

        @kb.add("c-j")
        def _(event):
            if state.is_generating:
                return
            event.current_buffer.insert_text("\n")

        @kb.add("c-c")
        def _(event):
            # 优先级 1: 如果正在执行任务，取消任务
            if state.is_generating and current_agent_task["task"] is not None:
                loop.call_soon_threadsafe(
                    lambda: asyncio.create_task(
                        self.cli.cancel_main_agent(reason="cancelled by ctrl+c")
                    )
                )
                current_agent_task["task"].cancel()
                exit_confirm["pressed_once"] = False  # 重置退出确认
                return
            
            # 优先级 2: 如果输入框有内容，清空输入
            if event.current_buffer.text.strip():
                event.current_buffer.text = ""
                exit_confirm["pressed_once"] = False  # 重置退出确认
                return
            
            # 优先级 3: 输入框为空，需要两次 Ctrl+C 才退出
            if not exit_confirm["pressed_once"]:
                # 第一次按 Ctrl+C，提示需要再按一次
                exit_confirm["pressed_once"] = True
                self.console.print("[yellow]⚠️  再按一次 Ctrl+C 退出[/yellow]")
                return
            
            # 第二次按 Ctrl+C，退出 CLI
            self._print_session_id_on_exit()
            should_exit.set()
            event.app.exit()



        input_body = Window(
            BufferControl(buffer=buffer),
            height=Dimension(min=1, max=10),
            wrap_lines=True,
        )
        placeholder = FormattedTextControl(
            lambda: "/help 查看命令  ↑↓ 历史  Tab 补全  Ctrl+C 取消" if not buffer.text else "",
        )
        
        # 🔥 新增：补全提示显示区域
        def _completion_hints():
            """显示补全建议"""
            if state.is_generating:
                return [("", "")]
            
            text = buffer.text.strip()
            if not text.startswith('/'):
                return [("", "")]
            
            # 获取匹配的命令
            from akg_agents.cli.commands.slash_commands import get_registry
            registry = get_registry()
            word = text[1:].lower()  # 去掉 /，转小写
            
            matches = []
            seen = set()
            for cmd in registry.list_all():
                cmd_id = id(cmd)
                if cmd_id in seen:
                    continue
                
                # 匹配命令名或别名
                if cmd.name.startswith(word):
                    matches.append(cmd)
                    seen.add(cmd_id)
                elif any(alias.startswith(word) for alias in cmd.aliases):
                    matches.append(cmd)
                    seen.add(cmd_id)
            
            # 如果没有匹配或已完全匹配，不显示
            if not matches or (len(matches) == 1 and matches[0].name == word):
                return [("", "")]
            
            # 构建提示文本
            fragments = []
            for cmd in matches[:5]:  # 最多显示 5 个
                # 显示主命令名和别名
                aliases_text = f" ({', '.join(cmd.aliases)})" if cmd.aliases else ""
                line = f" {cmd.name}{aliases_text}".ljust(22) + f"{cmd.description}\n"
                fragments.append(("class:completion-hint", line))
            
            return fragments
        
        completion_hints_window = Window(
            FormattedTextControl(_completion_hints),
            height=Dimension(max=6),
            dont_extend_height=True,
            style="class:completion-hint",
        )
        
        # 🔥 Loading 状态显示区域
        def _loading_status():
            """显示 loading 状态"""
            if not state.is_generating or not state.loading_text:
                return [("", "")]
            return [("class:loading", f"{state.loading_text}\n")]
        
        loading_window = Window(
            FormattedTextControl(_loading_status),
            height=Dimension(max=2),
            dont_extend_height=True,
            style="class:loading",
        )
        
        input_container = Frame(
            body=HSplit([
                FloatContainer(
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
                completion_hints_window,  # 🔥 补全提示区域
            ]),
            title="输入",
            style="class:input-frame",
        )

        # 🔥 添加 loading 状态显示区域到输入框上方
        container = HSplit([
            loading_window,  # 🔥 Loading 状态（在输入框上方）
            input_container,
        ])

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
                    "completion-hint": "fg:#8b949e",  # 🔥 补全提示样式（灰色）
                    "loading": "fg:#fbbf24",  # 🔥 Loading 状态样式（黄色）
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

        async def _global_spinner() -> None:
            """全局 spinner 任务，根据 state.is_generating 动态显示"""
            import time
            spinner_frames = ["⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷"]
            frame_idx = 0
            start_time = None
            
            while not should_exit.is_set():
                if state.is_generating:
                    if start_time is None or state.reset_timer:
                        start_time = time.time()
                        state.reset_timer = False
                    
                    elapsed = int(time.time() - start_time)
                    wf_type = state.workflow_type
                    
                    if wf_type == "evolve" and state.evolve_state:
                        compact = state.evolve_state.render_compact()
                        state.loading_text = f"{spinner_frames[frame_idx]} Evolve {compact}"
                    elif wf_type == "search" and state.progress_info:
                        p = state.progress_info
                        op_name = p.get("op_name", "")
                        max_total = p.get("max_total_tasks", 0)
                        
                        if state.progress_display is None:
                            state.progress_display = AdaptiveSearchProgressDisplay(
                                op_name=op_name,
                                max_tasks=max_total,
                            )
                        
                        state.progress_display.update(p)
                        
                        if max_total > 0:
                            progress_bar = state.progress_display.render_compact()
                            state.loading_text = f"{spinner_frames[frame_idx]} {progress_bar}"
                        else:
                            state.loading_text = f"{spinner_frames[frame_idx]} 正在思考... ({elapsed}s)"
                    elif state.progress_info:
                        p = state.progress_info
                        max_total = p.get("max_total_tasks", 0)
                        if state.progress_display is None and max_total > 0:
                            state.progress_display = AdaptiveSearchProgressDisplay(
                                op_name=p.get("op_name", ""),
                                max_tasks=max_total,
                            )
                        if state.progress_display:
                            state.progress_display.update(p)
                            progress_bar = state.progress_display.render_compact()
                            state.loading_text = f"{spinner_frames[frame_idx]} {progress_bar}"
                        else:
                            state.loading_text = f"{spinner_frames[frame_idx]} 正在思考... ({elapsed}s)"
                    else:
                        state.progress_display = None
                        state.loading_text = f"{spinner_frames[frame_idx]} 正在思考... ({elapsed}s)"
                    
                    app.invalidate()
                    frame_idx = (frame_idx + 1) % len(spinner_frames)
                else:
                    start_time = None
                    state.progress_display = None
                    state.progress_info = {}
                    state.evolve_state = None
                    state.workflow_type = ""
                    state.loading_text = ""
                    app.invalidate()
                
                await asyncio.sleep(0.1)

        async def _process_input(user_input: str, print_input: bool = False) -> bool:
            user_input = (user_input or "").strip()
            if not user_input:
                state.is_generating = False
                app.invalidate()
                return True

            # 🔥 只在需要时打印（initial_input 需要打印）
            if print_input:
                self.cli.console.print_user_input(user_input)

            # 🔥 全局 spinner 会自动显示，不需要单独的 spinner_task
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
            if not state_result.get("should_continue", True):
                return False
            return True

        async def _workflow_loop() -> None:
            if initial_input:
                state.is_generating = True
                state.loading_text = "运行中..."
                app.invalidate()
                cont = await _process_input(initial_input, print_input=True)  # 🔥 初始输入需要打印
                if not cont:
                    should_exit.set()
                    _safe_app_exit()
                    return
            while not should_exit.is_set():
                user_input = await input_queue.get()
                cont = await _process_input(user_input, print_input=False)  # 🔥 已在 _accept_input 中打印
                if not cont:
                    should_exit.set()
                    _safe_app_exit()
                    return
        
        def _update_progress(progress_data: dict) -> None:
            """更新进度信息到 TUI state"""
            wf_type = progress_data.pop("_workflow_type", "")
            is_reset = progress_data.pop("_reset", False)
            
            if is_reset:
                state.workflow_type = ""
                state.progress_info = {}
                state.progress_display = None
                state.evolve_state = None
                state.reset_timer = True
                app.invalidate()
                return
            
            if wf_type:
                state.workflow_type = wf_type
            
            if wf_type == "evolve":
                if state.evolve_state is None:
                    state.evolve_state = EvolveProgressState()
                state.evolve_state.update(progress_data)
            else:
                state.progress_info = progress_data
            app.invalidate()
        
        # 设置回调
        console_obj = getattr(self.cli, 'console', None)
        if console_obj:
            if hasattr(console_obj, '_progress_callback'):
                console_obj._progress_callback = _update_progress
        app_task = asyncio.create_task(app.run_async())
        workflow_task = asyncio.create_task(_workflow_loop())
        spinner_task = asyncio.create_task(_global_spinner())  # 🔥 启动全局 spinner
        
        done, _ = await asyncio.wait(
            {app_task, workflow_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if workflow_task in done and not app_task.done():
            _safe_app_exit()
        await app_task
        
        # 🔥 清理任务
        if not workflow_task.done():
            workflow_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await workflow_task
        
        if not spinner_task.done():
            spinner_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await spinner_task
        
        # 清理回调
        console_obj = getattr(self.cli, 'console', None)
        if console_obj:
            if hasattr(console_obj, '_progress_callback'):
                console_obj._progress_callback = None
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
        """渲染代理消息（opencode 风格：简洁缩进文本，无 Panel 边框）"""
        if not isinstance(state, dict):
            return

        # flush 可能还没关闭的 thinking/content 流
        if hasattr(self.cli, "console") and hasattr(self.cli.console, "_flush_all_streams"):
            try:
                self.cli.console._flush_all_streams()
            except Exception:
                pass

        display_message = str(state.get("display_message") or "").rstrip()
        if display_message:
            for line in display_message.split("\n"):
                try:
                    self.console.print(Text(f"  {line}"))
                except Exception:
                    self.console.print(f"  {line}", markup=False)

        hint_message = str(state.get("hint_message") or "").rstrip()
        if hint_message:
            try:
                self.console.print(Text(f"  {hint_message}", style="dim"))
            except Exception:
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
                self._print_session_id_on_exit()
                self.console.print(
                    f"\n[{DisplayStyle.YELLOW}]已退出[/{DisplayStyle.YELLOW}]"
                )
            except Exception as e:
                self.console.print(
                    f"[{DisplayStyle.RED}]错误:[/{DisplayStyle.RED}] {e}"
                )
                raise typer.Exit(code=1)
