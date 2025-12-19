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

"""CLI 展示器 - 负责所有 UI 渲染和用户交互"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from textual import log

from ai_kernel_generator.cli.cli.constants import (
    AKG_CLI_LOGO,
    DisplayStyle,
    UISymbol,
    make_gradient_logo,
)
from ai_kernel_generator.cli.cli.presenter.stream import StreamRenderer
from ..ui import InfoPanelState, WorkflowPanelState, create_default_layout_manager
from ..ui.protocols import LayoutManager
from ..utils.i18n import t
from .evolve import EvolveTaskManager
from .event_handlers import PresenterEventHandlers

if TYPE_CHECKING:
    from ...messages import (
        LLMEndMessage,
        LLMStartMessage,
        LLMStreamMessage,
        NodeEndMessage,
        NodeStartMessage,
        ProgressMessage,
    )


class CLIPresenter:
    """CLI 展示器 - 负责 UI 渲染和显示。"""

    def __init__(
        self,
        console: Console,
        use_stream: bool = False,
        *,
        layout_manager: LayoutManager | None = None,
    ):
        self.console = console
        self.use_stream = use_stream

        # UI 布局管理器：Presenter 仅依赖抽象接口（默认使用 Textual 实现）
        self.layout_manager: LayoutManager = (
            layout_manager or create_default_layout_manager()
        )
        self.stream_renderer = StreamRenderer(
            console, layout_manager=self.layout_manager
        )
        self.llm_buffer = ""
        self.current_agent = ""
        self.current_model = ""
        self.llm_running = False
        self.op_name = ""  # 维护当前任务名
        self.last_prompt_tokens = None
        self.last_reasoning_tokens = None
        self.last_output_tokens = None
        self.last_total_tokens = None
        self.llm_records = []
        # 便于“复制”的缓存（Textual 下可一键复制）
        self.last_task_desc: str = ""
        self.last_kernel_code: str = ""
        self.last_job_id: str = ""

        # ========= 右侧面板状态（Task Info / Workflow）=========
        self._task_context: dict[str, str] = {
            "framework": "",
            "backend": "",
            "arch": "",
            "dsl": "",
            "workflow_name": "",
        }
        self._current_node: str = ""
        self._current_node_status: str = ""  # running/done/idle
        self._seen_nodes: list[str] = []
        # 当前 watch task 的“执行计数/序号/轨迹”（从 evolve manager 派生，用于渲染）
        self._node_run_counts: dict[str, int] = {}
        self._current_node_run_no: int = 0
        self._node_trace: list[str] = []

        # Node 耗时收集
        self.node_timings = []
        # 性能历史记录
        self.performance_history = []

        # 并发 task 管理（watch/trace/progress/replay + task 状态缓存）
        self.tasks = EvolveTaskManager(self)
        # 事件处理器（LLM/Node/Progress 等）
        self._handlers = PresenterEventHandlers(self)

    # ========= 右侧面板：Task Info / Workflow =========

    def set_task_context(
        self,
        *,
        framework: str,
        backend: str,
        arch: str,
        dsl: str,
        workflow_name: str = "",
    ) -> None:
        """设置本次任务的上下文信息（用于右侧 Task Info 面板）。"""
        self._task_context = {
            "framework": str(framework or ""),
            "backend": str(backend or ""),
            "arch": str(arch or ""),
            "dsl": str(dsl or ""),
            "workflow_name": str(workflow_name or ""),
        }
        self._refresh_info_panel()

    def _refresh_info_panel(self) -> None:
        """刷新 Task Info 面板（使用响应式 API）。"""
        ctx = dict(self._task_context or {})

        # 构建类型安全的状态对象
        state = InfoPanelState(
            framework=ctx.get("framework"),
            backend=ctx.get("backend"),
            arch=ctx.get("arch"),
            dsl=ctx.get("dsl"),
            watch_task_id=self.tasks.watch_task_id,
        )

        try:
            # 响应式更新（自动差分，只在变化时才渲染）
            self.layout_manager.update_info_state(state)
        except Exception as e:
            log.warning("[Presenter] update_info_state failed", exc_info=e)

    def _refresh_workflow_panel(self) -> None:
        """刷新 Workflow 面板（数据 -> state；渲染逻辑在 widget 内）。"""
        try:
            self.layout_manager.update_workflow_state(
                WorkflowPanelState(
                    workflow_name=str(
                        (self._task_context or {}).get("workflow_name") or ""
                    ),
                    current_node=str(self._current_node or ""),
                    current_node_status=str(self._current_node_status or ""),
                    seen_nodes=list(self._seen_nodes or []),
                    node_run_counts=dict(self._node_run_counts or {}),
                    current_node_run_no=int(self._current_node_run_no or 0),
                    progress=dict(self.tasks.latest_progress_data or {}),
                    watch_task_id=str(self.tasks.watch_task_id or ""),
                )
            )
        except Exception as e:
            log.warning("[Presenter] update_workflow_state failed", exc_info=e)

    # ========= 外部回调（由 client/session 驱动） =========

    def on_llm_start(self, message: "LLMStartMessage") -> None:
        self._handlers.on_llm_start(message)

    def on_llm_end(self, message: "LLMEndMessage") -> None:
        self._handlers.on_llm_end(message)

    def on_llm_stream(self, message: "LLMStreamMessage") -> None:
        self._handlers.on_llm_stream(message)

    def on_job_submitted(self, job_id: str) -> None:
        self._handlers.on_job_submitted(job_id)

    def on_progress(self, message: "ProgressMessage") -> None:
        self._handlers.on_progress(message)

    def on_node_start(self, message: "NodeStartMessage") -> None:
        self._handlers.on_node_start(message)

    def on_node_end(self, message: "NodeEndMessage") -> None:
        self._handlers.on_node_end(message)

    def display_summary(self, result: dict) -> None:
        self._handlers.display_summary(result)

    # ========= UI 输出（会话/runner 调用） =========

    def print_logo(self) -> None:
        """打印 AKG CLI Logo（进入主界面前展示一次）"""
        logo_panel = Panel.fit(
            make_gradient_logo(),
            border_style=DisplayStyle.CYAN,
            padding=(1, 2),
            title=f"[{DisplayStyle.BOLD_CYAN}]AKG CLI[/{DisplayStyle.BOLD_CYAN}]",
        )
        self.console.print(logo_panel)

    def print_welcome(self, use_mock: bool, server_url: str | None = None) -> None:
        """打印欢迎信息"""
        mode_text = (
            f"[{DisplayStyle.YELLOW}]Mock 模式[/{DisplayStyle.YELLOW}]"
            if use_mock
            else f"[{DisplayStyle.GREEN}]真实 AI 模式[/{DisplayStyle.GREEN}]"
        )
        execution_mode = f"[{DisplayStyle.MAGENTA}]远程 WebSocket 模式[/{DisplayStyle.MAGENTA}] - 服务器: {server_url}"

        welcome = Panel.fit(
            f"[{DisplayStyle.BOLD_CYAN}]AIKG Workflow - AI Kernel Generator[/{DisplayStyle.BOLD_CYAN}]\n\n"
            f"当前模式: {mode_text}\n"
            f"执行模式: {execution_mode}\n\n"
            "自动化 Kernel 代码生成工作流\n\n"
            "流程:\n"
            "  [1] TaskInit: 需求理解 → KernelBench 格式\n"
            "  [2] LangGraphTask: 生成优化 Kernel 代码\n\n"
            f"特性:\n"
            f"  {UISymbol.BULLET} 多 Agent 协作\n"
            f"  {UISymbol.BULLET} 实时进度展示\n"
            f"  {UISymbol.BULLET} 性能统计\n"
            f"  {UISymbol.BULLET} 端到端自动化",
            border_style=DisplayStyle.CYAN,
            padding=(1, 2),
        )
        self.console.print(welcome)

    def print_user_input(self, user_input: str) -> None:
        """打印用户输入"""
        output = f"\n[{DisplayStyle.BOLD}]{t('presenter.user_request')}:[/{DisplayStyle.BOLD}] {user_input}\n"
        self._handlers._emit_main_global(Text.from_markup(output))
        self._refresh_info_panel()
        self._refresh_workflow_panel()

    def print_workflow_start(self) -> None:
        """打印工作流开始信息"""
        self.llm_records = []
        self.last_prompt_tokens = None
        self.last_reasoning_tokens = None
        self.last_output_tokens = None
        self.last_total_tokens = None
        # 每次工作流独立统计
        self.node_timings = []
        self.performance_history = []
        # 刷新右侧面板：初始化 workflow 状态
        self._current_node = ""
        self._current_node_status = "idle"
        self._refresh_info_panel()
        self._refresh_workflow_panel()
        # 启动 UI 事件泵（保证 [ / ] 切换即时生效）
        self.tasks.start_ui_pump()

    def print_workflow_complete(self) -> None:
        """打印工作流完成信息"""
        # 不要在任务结束时停止 UI 事件泵：
        # 用户仍可能希望在 Summary 阶段用 F8/F9（或 [ / ]）切换观察并发任务并回放日志。
        self._current_node_status = "done"
        self._refresh_info_panel()
        self._refresh_workflow_panel()

    def print_goodbye(self) -> None:
        """打印再见信息"""
        self._handlers._emit_main_global(
            Text.from_markup(
                f"\n[{DisplayStyle.YELLOW}]{t('presenter.goodbye')}[/{DisplayStyle.YELLOW}]"
            )
        )

    def print_error(self, error: str, show_traceback: bool = False) -> None:
        """打印错误信息"""
        self.tasks.stop_ui_pump()
        self._handlers._emit_main_global(
            Text.from_markup(
                f"\n[{DisplayStyle.RED}]{t('presenter.error')}: {error}[/{DisplayStyle.RED}]"
            )
        )
        if show_traceback:
            import traceback

            self._handlers._emit_main_global(Text(traceback.format_exc()))
