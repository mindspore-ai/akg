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

import logging
import os
import sys
import asyncio
from queue import Empty
from typing import Coroutine, Optional

from rich.text import Text
from textual import log

from .app import SplitViewApp
from ai_kernel_generator.cli.cli.ui.commands import (
    AppendTrace,
    ClearTrace,
    Focus,
    PatchInfoState,
    PatchWorkflowState,
    Quit,
    SetActiveTaskTab,
    SetInput,
    SetTaskTabs,
    SetTrace,
    SetTraceTitle,
    UpdateInfoState,
    UpdateWorkflowState,
)
from ai_kernel_generator.cli.cli.ui.intents import UIIntent, WriteMainContent
from ai_kernel_generator.cli.cli.ui.state import InfoPanelState, WorkflowPanelState
from ai_kernel_generator.cli.cli.ui.types import ClearMainContent, MainContent
from ai_kernel_generator.cli.cli.utils.paths import get_log_dir

logger = logging.getLogger(__name__)


class TextualLayoutManager:
    """Textual 布局管理器 - 外部接口（线程安全）。"""

    def __init__(self, app: Optional[SplitViewApp] = None):
        self.app = app
        self._started = False

    def set_app(self, app: SplitViewApp) -> None:
        self.app = app

    def is_ui_active(self) -> bool:
        """UI 是否已创建（best-effort）。"""
        return self.app is not None

    def get_session_dir(self):
        """获取当前会话落盘目录（用于录制 server->cli 消息）。"""
        try:
            return getattr(self.app, "session_dir", None)
        except Exception as e:
            log.debug("[TUI] get_session_dir failed", exc_info=e)
            return None

    def run(self, workflow_task: Coroutine) -> None:
        """运行 Textual 应用（必须在主线程调用）。"""
        # 关键：TUI 运行时把 stdout/stderr logging 重定向到文件，避免污染界面
        self._configure_logging_for_tui()
        self.app = SplitViewApp(workflow_task=workflow_task)

        # 说明：
        # - 许多 SSH 客户端（例如 Termius）在 Textual 捕获鼠标时，无法进行“拖拽选中即复制”。
        # - Textual 6.x 支持 run(mouse=..., inline=...)。
        #   - mouse=False：尽量把鼠标事件留给终端（便于选中复制）
        #   - inline=True：不进入 alternate screen（更利于复制/滚动回看，但 UI 观感会更像“刷新式”）
        term_program = (os.environ.get("TERM_PROGRAM") or "").lower()
        is_termius = "termius" in term_program
        # 绝大多数“无法选中复制”的场景发生在 SSH 客户端里（例如 Termius / iTerm 的远程会话等）
        is_ssh = bool(
            os.environ.get("SSH_CONNECTION")
            or os.environ.get("SSH_CLIENT")
            or os.environ.get("SSH_TTY")
        )

        def _env_bool(name: str, default: bool) -> bool:
            raw = os.environ.get(name)
            if raw is None:
                return default
            return str(raw).strip().lower() in ["1", "true", "yes", "on", "y"]

        # 重要：默认值必须"保持历史行为不变"，避免影响跑批/录屏/交互习惯：
        # - inline=False：使用 alternate screen（Textual 默认体验）
        # - mouse=True：默认开启鼠标支持（便于点击/滚动交互）
        #
        # 如果你明确需要"终端回滚/鼠标滚轮"，再用环境变量显式开启：
        # - AIKG_TUI_INLINE=on/off
        # - AIKG_TUI_MOUSE=on/off
        inline = _env_bool("AIKG_TUI_INLINE", False)
        mouse = _env_bool("AIKG_TUI_MOUSE", False)

        try:
            import textual.constants as textual_constants

            log_file = getattr(textual_constants, "LOG_FILE", None)
        except Exception as e:
            log.debug("[TUI] read textual.constants.LOG_FILE failed", exc_info=e)
            log_file = None

        try:
            log.info(
                "[TUI] run",
                inline=inline,
                mouse=mouse,
                is_ssh=is_ssh,
                is_termius=is_termius,
                term_program=term_program,
                textual_log_file=str(log_file or ""),
            )
        except Exception as e:
            log.debug("[TUI] log run info failed", exc_info=e)

        self.app.run(
            inline=inline,
            inline_no_clear=True,
            mouse=mouse,
        )
        self._started = True

        # 退出后把 session-id 打印到终端（用于 resume）
        try:
            sid = str(getattr(self.app, "session_id", "") or "")
            saved = bool(getattr(self.app, "session_saved", False))
            resume_mode = bool(getattr(self.app, "resume_mode", False))
            if sid and not resume_mode:
                if saved:
                    print(f"\nsession-id: {sid}\n可使用：akg_cli resume {sid}\n")
                else:
                    print("\n会话未保存（未生成可 resume 的 session-id）\n")
        except Exception as e:
            log.debug("[TUI] print session-id failed", exc_info=e)

    def run_resume(self, session_id: str) -> None:
        """以 resume 模式打开一个已保存的会话（只回放展示，不再驱动 workflow）。"""
        self._configure_logging_for_tui()
        self.app = SplitViewApp(workflow_task=None, resume_session_id=str(session_id))
        # 回放模式更偏向“查看/点击回看”，默认打开 mouse（便于点击 Footer/Trace）

        def _env_bool(name: str, default: bool) -> bool:
            raw = os.environ.get(name)
            if raw is None:
                return default
            return str(raw).strip().lower() in ["1", "true", "yes", "on", "y"]

        inline = _env_bool("AIKG_TUI_INLINE", False)
        mouse = _env_bool("AIKG_TUI_MOUSE", False)
        self.app.run(inline=inline, inline_no_clear=True, mouse=mouse)
        self._started = True

    def run_replay(self, *, session_id: str, workflow_task: Coroutine) -> None:
        """以“消息回放”模式打开会话：用同一路径重放 messages.jsonl。"""
        self._configure_logging_for_tui()
        # 使用已保存目录作为 base_dir（SplitViewApp 会以 resume_session_id 打开且不 truncate）
        self.app = SplitViewApp(
            workflow_task=workflow_task, resume_session_id=str(session_id)
        )

        def _env_bool(name: str, default: bool) -> bool:
            raw = os.environ.get(name)
            if raw is None:
                return default
            return str(raw).strip().lower() in ["1", "true", "yes", "on", "y"]

        inline = _env_bool("AIKG_TUI_INLINE", False)
        mouse = _env_bool("AIKG_TUI_MOUSE", False)
        self.app.run(inline=inline, inline_no_clear=True, mouse=mouse)
        self._started = True

    @staticmethod
    def _configure_logging_for_tui() -> None:
        """把常见 logger 的 StreamHandler 从控制台移走，避免破坏 Textual UI。"""
        try:
            root = logging.getLogger()
            if getattr(root, "_aikg_tui_logging_configured", False):
                # 补齐 Textual 的 log file（可能来自旧版本/外部初始化未覆盖）
                try:
                    log_dir = get_log_dir() / "internal"
                    log_dir.mkdir(parents=True, exist_ok=True)
                    import textual.constants as textual_constants

                    if not getattr(textual_constants, "LOG_FILE", None):
                        textual_constants.LOG_FILE = str(log_dir / "textual.log")
                        try:
                            log.info(
                                "[TUI] textual log enabled",
                                file=str(textual_constants.LOG_FILE or ""),
                            )
                        except Exception as e:
                            log.debug("[TUI] log enabled info failed", exc_info=e)
                except Exception as e:
                    log.debug("[TUI] enable textual log failed", exc_info=e)
                return

            # 统一 internal 日志目录（Textual log / python logging 共用）
            try:
                log_dir = get_log_dir() / "internal"
                log_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                log.debug("[TUI] create internal log dir failed", exc_info=e)
                log_dir = None

            # 文件日志（保留排查能力）
            try:
                if log_dir is None:
                    raise RuntimeError("internal log_dir unavailable")
                fh = logging.FileHandler(str(log_dir / "tui.log"), encoding="utf-8")
                fh.setFormatter(
                    logging.Formatter(
                        "%(asctime)s - %(levelname)s - [%(name)s] %(message)s"
                    )
                )
            except Exception as e:
                log.debug(
                    "[TUI] init FileHandler failed; continue without file log",
                    exc_info=e,
                )
                fh = None

            # Textual 内置日志：在没有 devtools 的情况下也可落盘
            try:
                import textual.constants as textual_constants

                if (log_dir is not None) and not getattr(
                    textual_constants, "LOG_FILE", None
                ):
                    textual_constants.LOG_FILE = str(log_dir / "textual.log")
                try:
                    log.info(
                        "[TUI] logging configured",
                        tui_log=str((log_dir / "tui.log"))
                        if log_dir is not None
                        else "",
                        textual_log=str(
                            getattr(textual_constants, "LOG_FILE", "") or ""
                        ),
                    )
                except Exception as e:
                    log.debug("[TUI] log configured info failed", exc_info=e)
            except Exception as e:
                log.debug(
                    "[TUI] configure textual.constants.LOG_FILE failed", exc_info=e
                )

            # 移除 root 上直接写 stdout/stderr 的 handler
            for h in list(root.handlers):
                try:
                    if isinstance(h, logging.StreamHandler) and getattr(
                        h, "stream", None
                    ) in (sys.stdout, sys.stderr):
                        root.removeHandler(h)
                except Exception as e:
                    log.debug(
                        "[TUI] remove root stream handler failed; continue", exc_info=e
                    )
                    continue

            # 确保有 handler，否则会走 lastResort 输出到 stderr
            if fh is not None:
                root.addHandler(fh)
                # 不改变用户自定义的 level，只确保不低于 INFO
                try:
                    if root.level == logging.NOTSET:
                        root.setLevel(logging.INFO)
                except Exception as e:
                    log.debug("[TUI] set root level failed; ignore", exc_info=e)

            # 重点屏蔽“刷屏”的 logger 向 root 传播（并可选写入文件）
            noisy = [
                "ai_kernel_generator.cli.client.client",
                "httpx",
                "httpcore",
                "websockets",
                "urllib3",
            ]
            for name in noisy:
                try:
                    lg = logging.getLogger(name)
                    lg.propagate = False
                    if fh is not None and not lg.handlers:
                        lg.addHandler(fh)
                except Exception as e:
                    log.debug(
                        "[TUI] configure noisy logger failed; continue",
                        logger=name,
                        exc_info=e,
                    )
                    continue

            setattr(root, "_aikg_tui_logging_configured", True)
        except Exception as e:
            log.warning("[TUI] configure_logging_for_tui failed", exc_info=e)
            return

    def stop(self) -> None:
        if self.app:
            try:
                self.app.command_queue.put(Quit())
            except Exception as e:
                log.warning("[TUI] enqueue Quit failed", exc_info=e)
        self._started = False

    def update_main_content(self, content: MainContent) -> None:
        if self.app:
            self.app.output_queue.put(content)

    # ========= 响应式面板 API =========

    def update_info_state(self, state: InfoPanelState) -> None:
        """更新 Info Panel 状态（响应式，线程安全）

        提供类型安全和自动差分更新。

        Args:
            state: InfoPanelState 对象

        Example:
            manager.update_info_state(InfoPanelState(
                framework="pytorch",
                backend="cuda",
                arch="x86_64"
            ))
        """
        if self.app:
            try:
                self.app.command_queue.put(UpdateInfoState(state))
            except Exception as e:
                log.warning("[TUI] enqueue UpdateInfoState failed", exc_info=e)

    def patch_info_state(self, **kwargs) -> None:
        """增量更新 Info Panel 状态（只更新指定字段）

        只更新指定字段，其他字段保持不变。自动差分，避免不必要的渲染。

        Args:
            **kwargs: 要更新的字段（如 framework="pytorch"）

        Example:
            manager.patch_info_state(framework="pytorch", backend="cuda")
        """
        if self.app:
            try:
                self.app.command_queue.put(PatchInfoState(dict(kwargs)))
            except Exception as e:
                log.warning("[TUI] enqueue PatchInfoState failed", exc_info=e)

    def update_workflow_state(self, state: WorkflowPanelState) -> None:
        """更新 Workflow Panel 状态（响应式，线程安全）。"""
        if self.app:
            try:
                self.app.command_queue.put(UpdateWorkflowState(state))
            except Exception as e:
                log.warning("[TUI] enqueue UpdateWorkflowState failed", exc_info=e)

    def patch_workflow_state(self, **kwargs) -> None:
        """增量更新 Workflow Panel 状态（只更新指定字段）。"""
        if self.app:
            try:
                self.app.command_queue.put(PatchWorkflowState(dict(kwargs)))
            except Exception as e:
                log.warning("[TUI] enqueue PatchWorkflowState failed", exc_info=e)

    def append_trace_item(self, text: str, *, task_id: str, event_idx: int = 0) -> None:
        """追加一条 Trace（线程安全）。"""
        if self.app:
            try:
                self.app.command_queue.put(
                    AppendTrace(str(text), str(task_id or ""), int(event_idx or 0))
                )
            except Exception as e:
                log.warning("[TUI] enqueue AppendTrace failed", exc_info=e)

    def clear_trace(self) -> None:
        """清空 Trace 面板（线程安全）。"""
        if self.app:
            try:
                self.app.command_queue.put(ClearTrace())
            except Exception as e:
                log.warning("[TUI] enqueue ClearTrace failed", exc_info=e)

    def set_trace_title(self, title: str) -> None:
        """设置 Trace 面板标题（线程安全）。"""
        if self.app:
            try:
                self.app.command_queue.put(SetTraceTitle(str(title)))
            except Exception as e:
                log.warning("[TUI] enqueue SetTraceTitle failed", exc_info=e)

    def set_trace_items(self, items: list[tuple[str, str, int]]) -> None:
        """整体替换 Trace 列表（线程安全）。"""
        if self.app:
            try:
                self.app.command_queue.put(SetTrace(list(items)))
            except Exception as e:
                log.warning("[TUI] enqueue SetTrace failed", exc_info=e)

    def set_task_tabs(
        self, items: list[tuple[str, str]], active_task_id: str = ""
    ) -> None:
        """设置任务 Tabs（线程安全）。items: [(task_id, label), ...]"""
        if self.app:
            try:
                self.app.command_queue.put(
                    SetTaskTabs(list(items), str(active_task_id or ""))
                )
            except Exception as e:
                log.warning("[TUI] enqueue SetTaskTabs failed", exc_info=e)

    def set_active_task_tab(self, task_id: str) -> None:
        """仅更新当前高亮的 task tab（线程安全）。"""
        if self.app:
            try:
                self.app.command_queue.put(SetActiveTaskTab(str(task_id or "")))
            except Exception as e:
                log.warning("[TUI] enqueue SetActiveTaskTab failed", exc_info=e)

    def set_input_enabled(self, enabled: bool) -> None:
        if not self.app or self.app.user_input is None:
            return
        try:
            self.app.command_queue.put(SetInput(bool(enabled)))
        except Exception as e:
            log.warning("[TUI] enqueue SetInput failed", exc_info=e)

    async def await_user_input(self, prompt: str) -> str:
        if not self.app:
            raise RuntimeError("Textual app is not running")

        content: MainContent
        try:
            content = Text.from_markup(f"[bold]{prompt}[/bold]")
        except Exception as e:
            log.debug(
                "[TUI] render prompt as markup failed; fallback plain", exc_info=e
            )
            content = str(prompt)

        app = self.app
        q = getattr(app, "ui_event_queue", None) if app is not None else None
        if q is not None:
            try:
                q.put_nowait(WriteMainContent(content=content))
            except Exception as e:
                log.debug(
                    "[TUI] enqueue WriteMainContent failed; fallback output_queue",
                    exc_info=e,
                )

        self.set_input_enabled(True)
        value = await self.app.input_queue.get()
        self.set_input_enabled(False)
        return (value or "").strip()

    def clear_log(self) -> None:
        if self.app:
            self.app.output_queue.put(ClearMainContent())

    def focus_chat(self) -> None:
        """把焦点切到 Chat（便于直接滚动回看）。"""
        if self.app:
            try:
                self.app.command_queue.put(Focus("chat"))
            except Exception as e:
                log.warning("[TUI] enqueue Focus(chat) failed", exc_info=e)

    def focus_input(self) -> None:
        """把焦点切回输入框。"""
        if self.app:
            try:
                self.app.command_queue.put(Focus("input"))
            except Exception as e:
                log.warning("[TUI] enqueue Focus(input) failed", exc_info=e)

    def focus_trace(self) -> None:
        """把焦点切到 Trace 列表。"""
        if self.app:
            try:
                self.app.command_queue.put(Focus("trace"))
            except Exception as e:
                log.warning("[TUI] enqueue Focus(trace) failed", exc_info=e)

    # ========= Panel width (best-effort) =========

    def get_panel_content_width(self, panel: str = "chat") -> Optional[int]:
        """
        尽力获取 Textual 面板“内容区”的宽度（字符数）。

        用途：让流式渲染/分割线的宽度与 RichLog 实际可用宽度一致，避免 RichLog 二次自动换行造成错位。
        注意：这是 best-effort；如果 app 未运行或无法访问尺寸，返回 None。
        """
        app = self.app
        if app is None:
            return None

        try:
            p = (panel or "").strip().lower()
            if p in ["chat", "main", "left"]:
                w = getattr(app, "chat_log", None)
            elif p in ["workflow", "task", "side", "right"]:
                w = getattr(app, "workflow_panel", None)
            elif p in ["info", "progress", "prog"]:
                w = getattr(app, "info_panel", None)
            else:
                w = getattr(app, "chat_log", None)

            if w is None:
                return None

            # 优先使用 content_size（排除 border/padding），更贴近渲染宽度
            width = getattr(getattr(w, "content_size", None), "width", None)
            if width is None:
                width = getattr(getattr(w, "size", None), "width", None)
            if width is None:
                return None
            width_i = int(width)
            return width_i if width_i > 0 else None
        except Exception as e:
            log.debug("[TUI] get_panel_content_width failed; return None", exc_info=e)
            return None

    # ========= UI intents (best-effort) =========

    def drain_ui_intents(self) -> list[UIIntent]:
        """尽力从 UI 侧取出用户意图事件（类型化）。"""
        app = self.app
        if app is None:
            return []

        q = getattr(app, "ui_event_queue", None)
        if q is None:
            return []

        intents: list[UIIntent] = []

        while True:
            try:
                evt = q.get_nowait()
            except Empty:
                break
            except asyncio.QueueEmpty:
                break

            if isinstance(evt, UIIntent):
                intents.append(evt)
                continue

            # 未识别事件：忽略但保留 debug 线索
            logger.debug("Unknown ui_event_queue item: %r", evt)

        return intents
