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
import os
from pathlib import Path
from queue import Queue
from typing import Coroutine, Optional

from rich.console import Console
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button
from textual.widgets import Footer, Header, Static
from textual import log

from .chat_manager import ChatManager
from .i18n_manager import I18nManager
from .queue_processing import SplitViewQueueProcessor
from .session_store import (
    commit_session,
    discard_session,
    new_session_id,
    resolve_session_dir_for_resume,
    tmp_session_dir,
)
from .theme_manager import ThemeManager
from .widgets import (
    InteractiveInput,
    LogPane,
    TraceListItem,
    TraceListView,
    TaskTabs,
    ReactiveInfoPanel,
    ReactiveWorkflowPanel,
)
from ai_kernel_generator.cli.cli.ui.commands import UICommand
from ai_kernel_generator.cli.cli.ui.intents import (
    AppMounted,
    LangChanged,
    ThemeChanged,
    UIIntent,
    WatchNext,
    WatchSet,
    WriteMainContent,
)
from ai_kernel_generator.cli.cli.ui.types import MainContent
from ai_kernel_generator.cli.cli.constants import make_gradient_logo
from ai_kernel_generator.cli.cli.utils.i18n import t, toggle_lang


class _ConfirmSaveScreen(ModalScreen[bool]):
    """退出前确认是否保存会话。"""

    DEFAULT_CSS = """
    _ConfirmSaveScreen {
        align: center middle;
    }
    _ConfirmSaveScreen > Container {
        width: 60;
        padding: 1 2;
        border: round $border;
        background: $panel;
    }
    _ConfirmSaveScreen #buttons {
        height: auto;
        layout: horizontal;
        content-align: center middle;
        margin-top: 1;
    }
    _ConfirmSaveScreen Button {
        margin: 0 1;
    }
    """

    def compose(self) -> ComposeResult:
        with Container():
            yield Static(t("tui.msg.confirm_save_session"), id="msg")
            with Container(id="buttons"):
                yield Button(t("tui.btn.save"), id="save", variant="primary")
                yield Button(t("tui.btn.discard"), id="discard", variant="default")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = str(getattr(event.button, "id", "") or "")
        self.dismiss(bid == "save")


class SplitViewApp(App):
    """分屏应用主类。"""

    TITLE = "akg_cli"

    CSS = """
    Screen {
        layout: vertical;
        overflow-x: hidden;
        background: $background;
        color: $text;
    }

    Header, Footer {
        background: $background;
        color: $foreground;
        dock: none;
    }

    #tasks-bar {
        height: 3;
        min-height: 3;
        overflow-x: hidden;
        overflow-y: hidden;
        background: $background;
    }

    #main-container {
        height: 1fr;
        layout: horizontal;
        overflow-x: hidden;
        background: $background;
    }

    #chat-log {
        width: 6fr;
        min-width: 0;
        border: round $border;
        padding: 0 1;
        overflow-x: hidden;
        overflow-y: auto;
        background: $surface;
        scrollbar-size: 1 0; /* 纵向细滚动条；禁用横向滚动条 */
    }
    #chat-log:focus {
        background-tint: $foreground 0%;
        border: round $primary;
    }

    #right-pane {
        width: 4fr;
        min-width: 0;
        layout: vertical;
        overflow-x: hidden;
        background: $background;
    }

    #info-panel {
        height: 9;
        min-height: 7;
        border: round $border;
        padding: 0 1;
        overflow-x: hidden;
        overflow-y: hidden;
        background: $panel;
    }
    #info-panel:focus {
        border: round $primary;
    }

    #task-tabs {
        height: 3;
        min-height: 3;
        border: round $border;
        padding: 0 1;
        background: $panel;
        overflow-x: auto;
        overflow-y: hidden;
    }
    #task-tabs:focus {
        border: round $primary;
    }

    /* 选中 tab 更醒目：主题色背景 + 加粗 */
    #task-tabs Tab.-active {
        background: $primary;
        text-style: bold;
        color: $background;
    }
    #task-tabs Tab.-active:dark {
        color: $foreground;
    }
    #task-tabs Tab:hover {
        background: $primary-muted;
    }

    #workflow-panel {
        height: 1fr;
        width: 1fr;
        min-width: 0;
        border: round $border;
        padding: 0 1;
        background: $panel;
        text-wrap: wrap;
        overflow-x: hidden;
        overflow-y: auto;
        scrollbar-size: 1 0;
    }
    #workflow-panel:focus {
        border: round $primary;
    }

    #trace-panel {
        height: 1fr;
        width: 1fr;
        min-width: 0;
        border: round $border;
        padding: 0 1;
        background: $panel;
        overflow-x: hidden;
        overflow-y: auto;
        scrollbar-size: 1 0;
    }
    #trace-panel:focus {
        border: round $primary;
    }

    #input-container {
        height: 7;
        min-height: 5;
    }

    #user-input {
        border: round $border;
        padding: 0 1;
        width: 100%;
        height: 100%;
        overflow-x: hidden;
        overflow-y: auto;
        background: $surface;
    }
    #user-input:focus {
        border: round $primary;
    }

    /* 聚焦样式调优：避免 focused 时整体背景变暗过多 */
    #trace-panel:focus {
        background-tint: $foreground 0%;
    }
    #trace-panel > ListItem.-highlight {
        color: $block-cursor-blurred-foreground;
        background: $foreground 4%;
        text-style: $block-cursor-blurred-text-style;
    }
    #trace-panel:focus > ListItem.-highlight {
        color: $foreground;
        background: $foreground 8%;
        text-style: $block-cursor-text-style;
    }

    #user-input .text-area--cursor-line {
        background: $foreground 3%;
    }
    #user-input .text-area--cursor-gutter {
        background: $foreground 3%;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "退出", show=True, key_display="Ctrl+C"),
        Binding(
            "ctrl+s",
            "save_tui",
            t("tui.binding.save_tui"),
            show=False,
            key_display="Ctrl+S",
        ),
        Binding("ctrl+e", "scroll_end", "跳到底部", show=True, key_display="Ctrl+E"),
        Binding("f8", "watch_next", "下一个并发任务", show=True),
        # Binding("f9", "toggle_language", "切换语言", show=True),
        Binding("f10", "toggle_theme", "切换主题", show=True),
    ]

    def __init__(
        self,
        workflow_task: Optional[Coroutine] = None,
        *,
        resume_session_id: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.theme_manager = ThemeManager(self)
        self.theme_manager.register_builtin_themes()
        self.chat = ChatManager(self, self.theme_manager)
        self.i18n = I18nManager(self)

        self.theme = (
            "aikg-dark" if "aikg-dark" in self.available_themes else "textual-dark"
        )
        self.chat_log: Optional[LogPane] = None
        self.task_tabs: Optional[TaskTabs] = None
        self.info_panel: Optional[Static] = None
        self.workflow_panel: Optional[Static] = None
        self.trace_panel: Optional[TraceListView] = None
        self.user_input: Optional[InteractiveInput] = None

        # 响应式面板
        self.reactive_info_panel: ReactiveInfoPanel = ReactiveInfoPanel(
            t_func=t, empty_placeholder=t("tui.msg.wait_task_info")
        )
        self.reactive_workflow_panel: ReactiveWorkflowPanel = ReactiveWorkflowPanel(
            empty_placeholder=t("tui.msg.wait_workflow"),
            t_func=t,
        )

        self.workflow_task = workflow_task
        self.workflow_running = False
        self._workflow_runner_task: Optional[asyncio.Task] = None
        self._input_enabled = True
        self._input_block_reason: str = ""
        self._main_task_id: str = "main"
        self._active_task_id: str = ""

        # workflow 协程可 await
        self.input_queue: asyncio.Queue[str] = asyncio.Queue()
        # UI -> workflow/presenter 的事件队列（用于“丝滑切换观察任务”等）
        self.ui_event_queue: asyncio.Queue[UIIntent] = asyncio.Queue()

        # 线程安全队列（worker -> UI）
        self.output_queue: Queue[MainContent] = Queue()
        self.command_queue: Queue[UICommand] = Queue()
        # (task_id, event_idx) -> anchor_y（用于 Trace 点击后滚动）
        self.trace_anchors: dict[tuple[str, int], int] = {}
        # Tabs id <-> task_id 映射（修复 task_id 以数字开头导致 Tab.id 非法的问题）
        self._tab_id_to_task_id: dict[str, str] = {}
        self._task_id_to_tab_id: dict[str, str] = {}
        # 是否追底（Tail 模式）：默认开启；用户点 Trace/上滑后自动关闭
        self.follow_tail: bool = True
        # 当前视图位置与上次点击的 Trace 目标位置（用于定位确认）
        self._current_scroll_y: Optional[int] = None
        self._current_trace_anchor_y: Optional[int] = None
        self._chat_subtitle_cache: str = ""
        # 是否展示顶部任务 Tabs（仅 evolve 并发时需要）
        self.show_task_tabs: bool = self._env_bool("AIKG_TUI_TASK_TABS", True)
        self.logo_printed: bool = False
        # 是否开启鼠标（影响提示文案）
        self.mouse_enabled: bool = self._env_bool("AIKG_TUI_MOUSE", False)
        # 可恢复会话：默认写入临时目录；用户 Ctrl+C 退出时选择“保存/丢弃”
        self.resume_mode: bool = bool(resume_session_id)
        env_sid = (os.environ.get("AIKG_SESSION_ID") or "").strip()
        self.session_id: str = str(resume_session_id or env_sid or new_session_id())
        self.session_saved: bool = False

        if self.resume_mode:
            self.session_dir: Path = resolve_session_dir_for_resume(self.session_id)
        else:
            self.session_dir = tmp_session_dir(self.session_id)
            self.session_dir.mkdir(parents=True, exist_ok=True)
        self._queue_processor = SplitViewQueueProcessor(self)

    @staticmethod
    def _env_bool(name: str, default: bool) -> bool:
        raw = os.environ.get(name)
        if raw is None:
            return default
        return str(raw).strip().lower() in ["1", "true", "yes", "on", "y"]

    def set_main_task_id(self, task_id: str) -> None:
        tid = str(task_id or "").strip()
        self._main_task_id = tid or "main"
        self._apply_input_state()

    def set_active_task_id(self, task_id: str) -> None:
        self._active_task_id = str(task_id or "").strip()
        try:
            if self.task_tabs is not None:
                self.task_tabs.set_active_task_id(self._active_task_id)
        except Exception as e:
            log.debug("[TUI] task_tabs.set_active_task_id failed", exc_info=e)
        self._apply_input_state()

    def is_on_main_tab(self) -> bool:
        if not self.show_task_tabs or self.task_tabs is None:
            return True
        main_id = str(self._main_task_id or "").strip()
        if not main_id:
            return True
        active = str(self._active_task_id or "").strip()
        if not active:
            return True
        return active == main_id

    def set_input_enabled(self, enabled: bool) -> None:
        self._input_enabled = bool(enabled)
        self._apply_input_state()

    def _resolve_input_placeholder(self) -> str:
        if self.resume_mode:
            return t("tui.placeholder.resume_readonly")
        if self._input_block_reason == "non_main":
            return t("tui.placeholder.input_only_main", main=self._main_task_id or "main")
        if not self._input_enabled:
            return t("tui.placeholder.input_disabled")
        wf_done = bool(
            self.workflow_task is not None
            and self._workflow_runner_task is not None
            and getattr(self._workflow_runner_task, "done", lambda: False)()
        )
        if wf_done:
            return t("tui.placeholder.input_done")
        if self.workflow_running:
            return t("tui.placeholder.input_enabled_hint")
        return t("tui.placeholder.input_initial")

    def _apply_input_state(self) -> None:
        if self.user_input is None:
            return
        if self.resume_mode:
            self.user_input.disabled = True
            self.user_input.placeholder = self._resolve_input_placeholder()
            return
        is_main = self.is_on_main_tab()
        if not is_main:
            self._input_block_reason = "non_main"
        elif not self._input_enabled:
            self._input_block_reason = "disabled"
        else:
            self._input_block_reason = ""
        enabled = bool(self._input_enabled) and is_main
        self.user_input.disabled = not enabled
        self.user_input.placeholder = self._resolve_input_placeholder()
        if enabled:
            try:
                self.user_input.focus()
            except Exception as e:
                log.debug("[TUI] user_input.focus failed", exc_info=e)

    def refresh_input_placeholder(self) -> None:
        if self.user_input is None:
            return
        try:
            self.user_input.placeholder = self._resolve_input_placeholder()
        except Exception as e:
            log.debug("[TUI] refresh_input_placeholder failed", exc_info=e)

    def _tasks_bar_title(self) -> str:
        if self.mouse_enabled:
            return t("tui.title.tasks_bar_mouse")
        return t("tui.title.tasks_bar_nomouse")

    def _trace_title(self) -> str:
        if self.mouse_enabled:
            return t("tui.title.trace_mouse")
        return t("tui.title.trace_nomouse")

    def set_follow_tail(self, enabled: bool) -> None:
        self.follow_tail = bool(enabled)
        try:
            log.info("[TUI] follow_tail", enabled=bool(self.follow_tail))
        except Exception as e:
            log.debug("[TUI] log follow_tail failed", exc_info=e)
        # 给用户一个明确的可见状态：Chat 标题副标题显示 Tail 状态/位置
        self._update_chat_subtitle()

    def set_current_scroll_y(self, value: float | int | None) -> None:
        if value is None:
            new_value = None
        else:
            try:
                new_value = int(value)
            except (TypeError, ValueError):
                return
        if new_value == self._current_scroll_y:
            return
        self._current_scroll_y = new_value
        self._update_chat_subtitle()

    def set_trace_anchor_y(self, anchor_y: int | None) -> None:
        if anchor_y is None:
            new_value = None
        else:
            try:
                new_value = int(anchor_y)
            except (TypeError, ValueError):
                return
        if new_value == self._current_trace_anchor_y:
            return
        self._current_trace_anchor_y = new_value
        self._update_chat_subtitle()

    def _format_chat_subtitle(self) -> str:
        tail_text = t("tui.tail.on") if self.follow_tail else t("tui.tail.off")
        parts = [tail_text]
        if self._current_scroll_y is not None:
            parts.append(f"y:{self._current_scroll_y}")
        if self._current_trace_anchor_y is not None:
            parts.append(f"anchor_y:{self._current_trace_anchor_y}")
        return " | ".join(parts)

    def _update_chat_subtitle(self) -> None:
        try:
            if self.chat_log is None:
                return
            subtitle = self._format_chat_subtitle()
            if subtitle == self._chat_subtitle_cache:
                return
            self._chat_subtitle_cache = subtitle
            self.chat_log.border_subtitle = subtitle
            try:
                self.chat_log.refresh()
            except Exception as e:
                log.debug("[TUI] chat_log.refresh failed", exc_info=e)
        except Exception as e:
            log.debug("[TUI] set chat_log.border_subtitle failed", exc_info=e)

    def compose(self) -> ComposeResult:
        yield Header()

        if self.show_task_tabs:
            # 顶部 Tasks 一行（全宽）
            with Container(id="tasks-bar"):
                self.task_tabs = TaskTabs(id="task-tabs")
                self.task_tabs.border_title = self._tasks_bar_title()
                yield self.task_tabs

        with Container(id="main-container"):
            # 尽量保留历史，避免只看到最后一段输出
            chat_max_lines = 200_000
            self.chat_log = LogPane(
                t("tui.title.chat"), id="chat-log", max_lines=chat_max_lines
            )
            yield self.chat_log
            with Vertical(id="right-pane"):
                self.info_panel = Static(t("tui.msg.wait_task_info"), id="info-panel")
                self.info_panel.border_title = t("tui.title.task_info")
                self.workflow_panel = Static(
                    t("tui.msg.wait_workflow"), id="workflow-panel"
                )
                self.workflow_panel.border_title = t("tui.title.workflow")
                self.trace_panel = TraceListView(id="trace-panel")
                self.trace_panel.border_title = self._trace_title()
                yield self.info_panel
                yield self.workflow_panel
                yield self.trace_panel

        with Container(id="input-container"):
            self.user_input = InteractiveInput(
                placeholder=t("tui.placeholder.input_initial"),
                id="user-input",
                on_submit=self._handle_user_input,
            )
            yield self.user_input

        yield Footer()

    def on_mount(self) -> None:
        if not self.logo_printed:
            try:
                self.output_queue.put(make_gradient_logo())
                self.output_queue.put(Text("\n"))
                self.logo_printed = True
            except Exception as e:
                log.debug("[TUI] enqueue logo failed", exc_info=e)
        if self.user_input is not None:
            self.user_input.focus()
        self.i18n.apply()
        if self.chat_log is not None:
            try:
                self.set_current_scroll_y(self.chat_log.scroll_y)
            except Exception as e:
                log.debug("[TUI] init scroll_y failed", exc_info=e)

        # 通知 Presenter 应用已挂载（打印 Logo 等）
        try:
            self.ui_event_queue.put_nowait(AppMounted())
        except Exception as e:
            log.debug("[TUI] emit AppMounted failed", exc_info=e)

        try:
            log.info(
                "[TUI] on_mount",
                theme=str(getattr(self, "theme", "") or ""),
                workflow_task=bool(self.workflow_task),
            )
        except Exception as e:
            log.debug("[TUI] log on_mount failed", exc_info=e)

        if self.resume_mode:
            # resume 只读：messages.jsonl 的“协议回放”由外层命令驱动
            if self.user_input is not None:
                try:
                    self.user_input.disabled = True
                    # 回放模式只读：避免输入框（即使 disabled）仍占用焦点/拦截键盘事件
                    try:
                        self.user_input.can_focus = False
                    except Exception as e:
                        log.debug("[TUI] disable input focus failed", exc_info=e)
                    self.user_input.placeholder = t("tui.placeholder.resume_readonly")
                except Exception as e:
                    log.debug("[TUI] set resume readonly input failed", exc_info=e)
            # 回放模式默认让用户直接回看/滚动左侧日志
            try:
                self.set_follow_tail(False)
            except Exception as e:
                log.debug("[TUI] ignored exception", exc_info=e)
            if self.chat_log is not None:
                try:
                    # 用 set_focus 更强制，避免 focus() 被某些状态吞掉
                    try:
                        self.set_focus(self.chat_log)
                    except Exception as e:
                        log.debug(
                            "[TUI] set_focus failed; fallback focus()", exc_info=e
                        )
                        self.chat_log.focus()
                except Exception as e:
                    log.debug("[TUI] ignored exception", exc_info=e)

        # 连接响应式面板到 widget
        if self.info_panel is not None:
            self.reactive_info_panel.set_widget(self.info_panel)
        if self.workflow_panel is not None:
            self.reactive_workflow_panel.set_widget(self.workflow_panel)

        # 注入依赖到自定义组件
        if self.trace_panel is not None and self.chat_log is not None:
            self.trace_panel.set_chat_log(self.chat_log)
            self.trace_panel.set_on_item_selected_callback(
                lambda: self.set_follow_tail(False)
            )
        if self.task_tabs is not None:
            self.task_tabs.set_ui_event_queue(self.ui_event_queue)
            self.task_tabs.set_tab_mapping(self._tab_id_to_task_id)

        self.set_interval(0.1, self._queue_processor.process)

        if self.workflow_task:
            self.workflow_running = True
            if self.user_input is not None:
                self.set_input_enabled(False)
            self._workflow_runner_task = asyncio.create_task(self._run_workflow())

    def _deserialize_output_payload(
        self, payload: dict[str, object]
    ) -> MainContent | None:
        try:
            tp = str(payload.get("type") or "")
        except Exception as e:
            log.debug("[TUI] ignored exception; return", exc_info=e)
            return None
        if tp == "clear":
            return ClearMainContent()
        if tp == "syntax_block":
            try:
                from ai_kernel_generator.cli.cli.ui.types import SyntaxBlockMainContent

                return SyntaxBlockMainContent(
                    code=str(payload.get("code") or ""),
                    lexer_name=str(payload.get("lexer_name") or "text"),
                    add_end_separator=bool(payload.get("add_end_separator")),
                    line_number_start=payload.get("line_number_start"),  # type: ignore[arg-type]
                )
            except Exception as e:
                log.debug("[TUI] ignored exception; return", exc_info=e)
                return None
        if tp == "markup":
            return str(payload.get("markup") or "")
        if tp == "text":
            return Text(str(payload.get("text") or ""))
        if tp == "renderable_str":
            return Text(str(payload.get("text") or ""))
        return None

    def _deserialize_command_payload(self, payload: dict[str, object]) -> object | None:
        try:
            tp = str(payload.get("type") or "")
            data = payload.get("data") or {}
            if not isinstance(data, dict):
                data = {}
        except Exception as e:
            log.debug("[TUI] ignored exception; return", exc_info=e)
            return None

        # 显式映射可回放的 command（字段名按 dataclass 定义）
        try:
            from ai_kernel_generator.cli.cli.ui.commands import (
                AppendTrace,
                ClearTrace,
                Focus,
                PatchInfoState,
                PatchWorkflowState,
                SetActiveTaskTab,
                SetInput,
                SetMainTaskId,
                ResetTaskTabs,
                SetTaskTabs,
                SetTrace,
                SetTraceTitle,
                UpdateInfoState,
                UpdateWorkflowState,
            )
        except Exception as e:
            log.debug("[TUI] ignored exception; return", exc_info=e)
            return None

        mapping = {
            "SetInput": SetInput,
            "Focus": Focus,
            "UpdateInfoState": UpdateInfoState,
            "PatchInfoState": PatchInfoState,
            "UpdateWorkflowState": UpdateWorkflowState,
            "PatchWorkflowState": PatchWorkflowState,
            "SetTraceTitle": SetTraceTitle,
            "SetTrace": SetTrace,
            "AppendTrace": AppendTrace,
            "ClearTrace": ClearTrace,
            "SetTaskTabs": SetTaskTabs,
            "SetActiveTaskTab": SetActiveTaskTab,
            "SetMainTaskId": SetMainTaskId,
            "ResetTaskTabs": ResetTaskTabs,
        }
        cls = mapping.get(tp)
        if cls is None:
            return None
        try:
            return cls(**data)  # type: ignore[arg-type]
        except Exception as e:
            log.debug("[TUI] ignored exception; return", exc_info=e)
            return None

    async def _run_workflow(self) -> None:
        try:
            log.info("[TUI] workflow_task start")
        except Exception as e:
            log.debug("[TUI] ignored exception", exc_info=e)
        try:
            await self.workflow_task
        except asyncio.CancelledError:
            msg = t("tui.msg.workflow_cancelled")
            try:
                log.warning("[TUI] workflow_task cancelled")
            except Exception as e:
                log.debug("[TUI] ignored exception", exc_info=e)
            # Workflow 面板不再渲染“message 分支”；状态提示统一写入主输出。
            try:
                self._emit_main_content(
                    Text(
                        msg,
                        style=f"bold {self.theme_manager.theme_color('warning', 'blue')}",
                    )
                )
            except Exception as e:
                log.debug("[TUI] ignored exception", exc_info=e)
            raise
        except Exception as e:
            msg = t("tui.msg.workflow_error", error=str(e))
            try:
                import traceback

                log.error(
                    "[TUI] workflow_task error",
                    error=f"{type(e).__name__}: {e}",
                )
                log.error(traceback.format_exc())
            except Exception as e:
                log.debug("[TUI] ignored exception", exc_info=e)
            try:
                self._emit_main_content(
                    Text(
                        msg,
                        style=f"bold {self.theme_manager.theme_color('error', 'blue')}",
                    )
                )
            except Exception as e:
                log.debug("[TUI] ignored exception", exc_info=e)
            import traceback

            tb = traceback.format_exc()
            try:
                self._emit_main_content(Text(tb))
            except Exception as e:
                log.debug("[TUI] ignored exception", exc_info=e)
        finally:
            self.workflow_running = False
            if self.user_input is not None:
                self.set_input_enabled(True)
            try:
                log.info("[TUI] workflow_task done")
            except Exception as e:
                log.debug("[TUI] ignored exception", exc_info=e)
            done_msg = t("tui.msg.workflow_done")
            exit_msg = t("tui.msg.press_ctrl_c_exit")
            try:
                text = Text()
                text.append(
                    done_msg,
                    style=f"bold {self.theme_manager.theme_color('success', 'blue')}",
                )
                text.append("\n")
                text.append(exit_msg)
                self._emit_main_content(text)
            except Exception as e:
                log.debug(
                    "[TUI] output_queue.put(Text) failed; fallback str", exc_info=e
                )
                try:
                    self._emit_main_content(f"{done_msg}\n{exit_msg}")
                except Exception as e:
                    log.debug("[TUI] ignored exception", exc_info=e)

    def _handle_user_input(self, value: str) -> None:
        if not value:
            return
        if self.user_input is not None and self.user_input.disabled:
            try:
                log.warning(
                    "[TUI] input ignored (disabled)",
                    length=len(value or ""),
                    preview=(value or "")[:80],
                )
            except Exception as e:
                log.debug("[TUI] ignored exception", exc_info=e)
            if self.chat_log is not None:
                try:
                    warn_key = "tui.msg.input_disabled_warn"
                    warn_kwargs = {}
                    if self._input_block_reason == "non_main":
                        warn_key = "tui.msg.input_only_main_warn"
                        warn_kwargs = {"main": self._main_task_id or "main"}
                    self.ui_event_queue.put_nowait(
                        WriteMainContent(
                            content=self.chat.tagged_line(
                                "[WARN]",
                                t(warn_key, **warn_kwargs),
                                color_var="warning",
                            )
                        )
                    )
                except Exception as e:
                    log.debug("[TUI] emit WriteMainContent failed", exc_info=e)
            return
        if self.chat_log is not None:
            try:
                log.info(
                    "[TUI] input accepted",
                    length=len(value or ""),
                    preview=(value or "")[:80],
                )
            except Exception as e:
                log.debug("[TUI] ignored exception", exc_info=e)
            try:
                self.ui_event_queue.put_nowait(
                    WriteMainContent(
                        content=Text.assemble(
                            ("👤 ", f"bold {self.theme_manager.theme_color('success', 'blue')}"),
                            (value, "bold"),
                        )
                    )
                )
            except Exception as e:
                log.debug("[TUI] emit WriteMainContent failed", exc_info=e)
        try:
            self.input_queue.put_nowait(value)
        except Exception as e:
            log.debug("[TUI] ignored exception", exc_info=e)

    def _emit_main_content(self, content) -> None:
        """统一从 UI 侧写入主内容：优先走 UI intent。"""
        try:
            q = getattr(self, "ui_event_queue", None)
            if q is not None:
                q.put_nowait(WriteMainContent(content=content))
                return
        except Exception as e:
            log.debug("[TUI] enqueue WriteMainContent failed; drop", exc_info=e)

    def action_scroll_end(self) -> None:
        try:
            log.info("[TUI] action_scroll_end")
        except Exception as e:
            log.debug("[TUI] ignored exception", exc_info=e)
        # Ctrl+E 明确语义：跳到底并恢复追底
        self.set_follow_tail(True)
        if self.chat_log is not None:
            try:
                self.chat_log.scroll_end(animate=False)
            except Exception as e:
                log.debug("[TUI] ignored exception", exc_info=e)

    def action_toggle_language(self) -> None:
        """中英切换。"""
        try:
            log.info("[TUI] action_toggle_language")
        except Exception as e:
            log.debug("[TUI] ignored exception", exc_info=e)
        new_lang = toggle_lang()
        self.i18n.apply()

        # 通知 presenter 刷新右侧面板/Trace 标题等（best-effort）
        try:
            self.ui_event_queue.put_nowait(LangChanged(lang=new_lang))
        except Exception as e:
            log.debug("[TUI] ignored exception", exc_info=e)

    def action_toggle_theme(self) -> None:
        """切换 dark/light 主题。"""
        try:
            log.info("[TUI] action_toggle_theme")
        except Exception as e:
            log.debug("[TUI] ignored exception", exc_info=e)
        is_dark = True
        try:
            is_dark = bool(getattr(getattr(self, "current_theme", None), "dark", True))
        except Exception as e:
            log.debug("[TUI] read current_theme.dark failed; assume dark", exc_info=e)
            is_dark = True

        light_theme = (
            "aikg-light" if "aikg-light" in self.available_themes else "textual-light"
        )
        dark_theme = (
            "aikg-dark" if "aikg-dark" in self.available_themes else "textual-dark"
        )
        new_theme = light_theme if is_dark else dark_theme
        try:
            self.theme = new_theme
        except Exception as e:
            try:
                self.notify(
                    t("tui.msg.theme_switch_failed", error=str(e)),
                    severity="error",
                    markup=False,
                )
            except Exception as e:
                log.debug("[TUI] ignored exception", exc_info=e)
            return

        try:
            self.ui_event_queue.put_nowait(ThemeChanged(theme=str(new_theme)))
        except Exception as e:
            log.debug("[TUI] emit ThemeChanged failed", exc_info=e)

    async def action_quit(self) -> None:
        try:
            log.info("[TUI] action_quit")
        except Exception as e:
            log.debug("[TUI] ignored exception", exc_info=e)

        if not self.resume_mode:
            save = False
            try:
                if hasattr(self, "push_screen_wait"):
                    save = bool(await self.push_screen_wait(_ConfirmSaveScreen()))
                else:
                    # 兼容旧版本：没有 modal await 能力时，保守默认保存
                    save = True
            except Exception as e:
                log.debug("[TUI] confirm save failed; default save=True", exc_info=e)
                save = True

            if save:
                try:
                    commit_session(self.session_id)
                    self.session_saved = True
                except Exception as e:
                    log.debug("[TUI] commit_session failed", exc_info=e)
                    self.session_saved = False
            else:
                discard_session(self.session_id)
                self.session_saved = False

        if (
            self._workflow_runner_task is not None
            and not self._workflow_runner_task.done()
        ):
            try:
                self._workflow_runner_task.cancel()
            except Exception as e:
                log.debug("[TUI] ignored exception", exc_info=e)
        self.exit()

    async def _export_tui_text(self) -> str:
        """导出当前 TUI 的纯文本（优先导出当前渲染屏幕）。

        说明：你在 TUI 里看到的“布局”（多个面板、边框）只有在导出“整屏渲染”时才会保留；
        若导出失败则返回空字符串。
        """

        async def _call_maybe_await(fn):
            try:
                value = fn()
                if asyncio.iscoroutine(value):
                    value = await value
                return value
            except Exception as e:
                log.debug("[TUI] ignored exception; return", exc_info=e)
                return None

        # 0) Textual 6.x：没有 export_text API，直接用 compositor 的 LayoutUpdate 渲染到 Rich Console 后导出文本
        try:
            screen = getattr(self, "screen", None)
            compositor = getattr(screen, "_compositor", None)
            if compositor is not None:
                layout_update = compositor.render_full_update(simplify=True)
                width = int(getattr(getattr(screen, "size", None), "width", 0) or 0)
                height = int(getattr(getattr(screen, "size", None), "height", 0) or 0)
                console = Console(
                    record=True,
                    width=width or 120,
                    height=height or None,
                    force_terminal=True,
                    color_system=None,
                )
                console.print(layout_update)
                text = console.export_text(styles=False)
                if text and text.strip():
                    return text
        except Exception as e:
            log.debug("[TUI] ignored exception", exc_info=e)

        # 1) Textual 的整屏导出：不同版本 API 可能是 sync/async，也可能挂在 screen/_compositor 上
        exporters = [
            lambda: getattr(self, "export_text")(),
            lambda: getattr(getattr(self, "screen", None), "export_text")(),
            lambda: getattr(
                getattr(getattr(self, "screen", None), "_compositor", None),
                "export_text",
            )(),
        ]
        for exporter in exporters:
            text = await _call_maybe_await(exporter)
            try:
                if text:
                    return str(text)
            except Exception as e:
                log.debug("[TUI] ignored exception", exc_info=e)

        return ""

    def _default_tui_dump_path(self) -> Path:
        self.session_dir.mkdir(parents=True, exist_ok=True)
        idx = 1
        while True:
            candidate = self.session_dir / f"tui_{idx}.txt"
            if not candidate.exists():
                return candidate
            idx += 1

    async def action_save_tui(self) -> None:
        """保存当前 TUI 文本到文件（类似“截图”，但为可复制的纯文本）。"""
        path = self._default_tui_dump_path()
        try:
            content = await self._export_tui_text()
            if not content:
                self.notify(
                    t("tui.msg.save_tui_empty"), severity="warning", markup=False
                )
                return
            path.write_text(content, encoding="utf-8")
        except Exception as e:
            try:
                self.notify(
                    t("tui.msg.save_tui_failed", error=str(e)),
                    severity="error",
                    markup=False,
                )
            except Exception as e:
                log.debug("[TUI] ignored exception", exc_info=e)
            return

        try:
            self.notify(t("tui.msg.save_tui_ok", path=str(path)), markup=False)
        except Exception as e:
            log.debug("[TUI] ignored exception", exc_info=e)

    def action_watch_next(self) -> None:
        """切换观察目标：下一个并发任务（由 presenter 执行实际切换与回放）。"""
        try:
            log.info("[TUI] action_watch_next")
        except Exception as e:
            log.debug("[TUI] ignored exception", exc_info=e)
        try:
            self.ui_event_queue.put_nowait(WatchNext())
        except Exception as e:
            log.debug("[TUI] ignored exception; return", exc_info=e)
            return
