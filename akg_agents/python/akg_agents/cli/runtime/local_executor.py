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

"""本地执行器 - 直接调用 KernelAgent，无需 server"""

import logging
import os
import uuid
from contextlib import redirect_stdout
from io import StringIO
from typing import Dict, Any, Optional

from akg_agents.op.config.config_validator import load_config
from akg_agents.cli.runtime.message_sender import (
    register_message_sender,
    unregister_message_sender,
)
from akg_agents.utils.stream_output import stream_output_override

logger = logging.getLogger(__name__)


class LocalExecutor:
    """本地执行器：CLI 侧基于 KernelAgent (V2) 的执行路径"""

    def __init__(
        self,
        console=None,
        *,
        session_id: str | None = None,
    ):
        self.session_id = str(session_id or uuid.uuid4())
        self.console = console
        self.current_job_id: Optional[str] = None

        # 兼容原 AIKGCLI 会话字段
        self.auto_yes: bool = False
        self.common_auto_approve_tools: bool = False
        self.default_user_input: str = ""
        self.use_stream: bool = False
        self.workflow_name: str = ""
        self.notifier = None

        # 会话状态管理
        self._main_agent_state: dict | None = None
        self._common_agent_state: dict | None = None
        # ReAct：跨轮次复用 executor（保留 memory/checkpointer）
        self._react_executor = None
        self._common_executor = None

    def _load_workflow_config(self, backend: str, dsl: str) -> Dict:
        """加载工作流配置"""
        config = load_config(dsl, backend=backend)

        # 兜底补齐 agent_model_config（使用 model_level）
        if "agent_model_config" not in config or not isinstance(
            config.get("agent_model_config"), dict
        ):
            config["agent_model_config"] = {}
        mc = config["agent_model_config"]

        base_default = mc.get("default") or "standard"
        mc.setdefault("default", base_default)
        if "log_dir" not in config or config["log_dir"] is None:
            config["log_dir"] = "~/akg_agents_logs"
        return config

    def _load_common_config(self) -> Dict:
        """加载 common 场景配置（最小化配置）。"""
        config: Dict[str, Any] = {
            "agent_model_config": {
                "default": "deepseek_r1_default",
            }
        }
        config.setdefault("enable_auto_compact", True)
        config.setdefault("auto_compact_ratio", 0.8)
        if "log_dir" not in config or config["log_dir"] is None:
            config["log_dir"] = "~/akg_agents_logs"
        return config

    def _build_response_state(self, state: dict) -> dict:
        """构建响应状态（ReAct 最小字段）"""

        payload = dict(state)
        payload.pop("config", None)
        payload.pop("conversation_history", None)

        if "should_continue" not in payload:
            payload["should_continue"] = True
        if "current_step" not in payload:
            payload["current_step"] = ""
        workflow_name = str(
            payload.get("workflow_name") or payload.get("sub_workflow") or ""
        )
        if workflow_name:
            payload["workflow_name"] = workflow_name
        payload.setdefault("display_message", "")
        payload.setdefault("hint_message", "")
        return payload

    def _ensure_common_executor(self):
        from akg_agents.cli.runtime.common_executor import CommonTurnExecutor

        base_config = None
        if self._common_agent_state and isinstance(self._common_agent_state, dict):
            base_config = self._common_agent_state.get("config")
        if base_config is not None:
            config = dict(base_config)
        else:
            config = self._load_common_config()

        config["session_id"] = self.session_id
        config.setdefault("workflow_name", "common")
        config["auto_approve_tools"] = bool(self.common_auto_approve_tools)

        if self._common_executor is None:
            self._common_executor = CommonTurnExecutor(
                session_id=self.session_id,
                config=config,
                thread_id=self.session_id,
            )

        self._common_agent_state = {"config": config}
        return self._common_executor

    def enter_common_plan_mode(self) -> dict:
        """Manually enter plan mode for common executor."""
        executor = self._ensure_common_executor()
        state = executor.enter_plan_mode()
        return self._build_response_state(state)

    async def execute_main_agent(
        self,
        *,
        user_input: str = "",
        framework: str = "torch",
        backend: str = "cuda",
        arch: str = "a100",
        dsl: str = "triton_cuda",
        use_stream: bool | None = None,
        rag: bool = False,
        output_path: str | None = None,
        task_file_content: str | None = None,
    ) -> Dict[str, Any]:
        """执行 KernelAgent 对话（start/continue），返回状态"""
        if not (user_input or "").strip():
            raise ValueError("user_input is required")

        effective_use_stream = (
            bool(use_stream) if use_stream is not None else bool(self.use_stream)
        )

        # 注册消息发送器（本地模式：直接调用 console）
        def _local_message_sender(message):
            """本地消息发送器：直接路由到 console"""

            # 直接路由到 console
            self._route_to_console(message)

        register_message_sender(self.session_id, _local_message_sender)

        try:
            user_input = (user_input or "").strip()
            
            # 使用 V2 执行器（基于 KernelAgent）
            from akg_agents.cli.runtime.react_executor_v2 import (
                ReactTurnExecutorV2,
                ReactTargetV2,
            )

            target = ReactTargetV2(
                framework=framework,
                backend=backend,
                arch=arch,
                dsl=dsl,
            )

            base_config = None
            if self._main_agent_state and isinstance(self._main_agent_state, dict):
                base_config = self._main_agent_state.get("config")
            if base_config is not None:
                config = dict(base_config)
            else:
                config = self._load_workflow_config(backend, dsl)

            config["session_id"] = self.session_id
            if output_path and "output_path" not in config:
                config["output_path"] = output_path
            config["rag"] = bool(rag)

            if (
                self._react_executor is None
                or getattr(self._react_executor, "target", None) != target
            ):
                self._react_executor = ReactTurnExecutorV2(
                    session_id=self.session_id,
                    config=config,
                    target=target,
                )

            # 保存最小 state（用于后续轮次复用 config）
            self._main_agent_state = {"config": config, "rag": bool(rag)}

            null_output = StringIO()
            with stream_output_override(bool(effective_use_stream)):
                with redirect_stdout(null_output):
                    state = await self._react_executor.run_turn(
                        user_input=user_input,
                        use_stream=effective_use_stream,
                    )
            # CLI 语义：finish 表示“一次任务完成”，不代表退出 akg_cli 会话。
            # 因此在 completed 后：
            # - 允许继续输入新需求（should_continue=True）
            # - 不重置 thread/executor：保留上下文，支持在同一会话里持续对话
            if isinstance(state, dict):
                cur = str(state.get("current_step") or "").strip().lower()
                if cur == "completed" or state.get("should_continue") is False:
                    state["should_continue"] = True
                    state.setdefault(
                        "hint_message",
                        "💡 本轮任务已完成。你可以继续输入新的需求（Ctrl+C 退出）。",
                    )
            return self._build_response_state(state)

        finally:
            unregister_message_sender(self.session_id)

    async def execute_common_agent(
        self,
        *,
        user_input: str = "",
        use_stream: bool | None = None,
    ) -> Dict[str, Any]:
        """执行 CommonAgent 对话（start/continue），返回状态"""
        if not (user_input or "").strip():
            raise ValueError("user_input is required")

        effective_use_stream = (
            bool(use_stream) if use_stream is not None else bool(self.use_stream)
        )

        # 注册消息发送器（本地模式：直接调用 console）
        def _local_message_sender(message):
            """本地消息发送器：直接路由到 console"""
            self._route_to_console(message)

        register_message_sender(self.session_id, _local_message_sender)

        try:
            user_input = (user_input or "").strip()

            from akg_agents.cli.runtime.common_executor import CommonTurnExecutor

            base_config = None
            if self._common_agent_state and isinstance(self._common_agent_state, dict):
                base_config = self._common_agent_state.get("config")
            if base_config is not None:
                config = dict(base_config)
            else:
                config = self._load_common_config()

            config["session_id"] = self.session_id
            config.setdefault("workflow_name", "common")
            config["auto_approve_tools"] = bool(self.common_auto_approve_tools)

            if self._common_executor is None:
                self._common_executor = CommonTurnExecutor(
                    session_id=self.session_id,
                    config=config,
                    thread_id=self.session_id,
                )

            # 保存最小 state（用于后续轮次复用 config）
            self._common_agent_state = {"config": config}

            null_output = StringIO()
            with stream_output_override(bool(effective_use_stream)):
                # with redirect_stdout(null_output):
                state = await self._common_executor.run_turn(
                    user_input=user_input,
                    use_stream=effective_use_stream,
                )

            # finish 后允许继续对话
            if isinstance(state, dict):
                cur = str(state.get("current_step") or "").strip().lower()
                if cur == "completed" or state.get("should_continue") is False:
                    state["should_continue"] = True
                    if not state.get("auto_input"):
                        state.setdefault(
                            "hint_message",
                            "💡 本轮任务已完成。你可以继续输入新的需求（Ctrl+C 退出）。",
                        )
            return self._build_response_state(state)

        finally:
            unregister_message_sender(self.session_id)

    def get_common_tools(self):
        """Return common tools list (without forcing model creation)."""
        if self._common_executor is not None:
            tools = getattr(self._common_executor, "tools", None)
            if tools is not None:
                return tools
        try:
            from akg_agents.cli.runtime.common_executor import (
                CommonToolState,
                build_common_tools,
            )
            return build_common_tools(CommonToolState(self.session_id))
        except Exception:
            return []

    def get_last_raw_llm_input(self) -> dict | None:
        """Return last raw LLM input snapshot for common executor."""
        if self._common_executor is None:
            return None
        return getattr(self._common_executor, "last_raw_llm_input", None)

    async def compact_common_history(self) -> dict:
        """Compact common conversation history if available."""
        if self._common_executor is None:
            return {"ok": False, "reason": "not_started"}
        return await self._common_executor.compact_history(
            reason="manual",
        )

    async def cancel_main_agent(self, reason: str = "cancelled by client") -> bool:
        """取消当前 main_agent"""
        try:
            cancelled = False
            for ex in (self._react_executor, self._common_executor):
                task = getattr(ex, "running_task", None) if ex is not None else None
                if task is not None and hasattr(task, "cancel"):
                    task.cancel()
                    cancelled = True
            if cancelled:
                logger.info(
                    f"[LocalExecutor] React cancellation requested for session {self.session_id}: {reason}"
                )
                return True
        except Exception as e:
            logger.warning(
                f"Cancel react executor failed: session_id={self.session_id}, err={e}"
            )
            return False
        return False

    def _route_to_console(self, message):
        """直接路由消息到 console"""

        if not self.console:
            logger.warning(
                f"[LocalExecutor] 没有设置 console，无法路由消息: type={type(message).__name__}"
            )
            return

        msg_type = type(message).__name__

        logger.debug(
            f"[LocalExecutor] session_id={self.session_id} - 收到消息: type={msg_type}"
        )

        # 直接根据消息类型路由
        try:
            from akg_agents.cli.messages import (
                DisplayMessage,
                LLMStreamMessage,
                PanelDataMessage,
                AgentHeaderMessage,
                ToolStartMessage,
                ToolResultMessage,
            )

            if isinstance(message, AgentHeaderMessage):
                self.console.on_agent_header(message)
            elif isinstance(message, ToolStartMessage):
                self.console.on_tool_start(message)
            elif isinstance(message, ToolResultMessage):
                self.console.on_tool_result(message)
            elif isinstance(message, DisplayMessage):
                self.console.on_display_message(message)
            elif isinstance(message, LLMStreamMessage):
                self.console.on_llm_stream(message)
            elif isinstance(message, PanelDataMessage):
                self.console.on_panel_data(message)
            else:
                logger.warning(f"[LocalExecutor] 未知消息类型: {msg_type}")

            logger.debug(
                f"[LocalExecutor] session_id={self.session_id} - 消息路由成功: type={msg_type}"
            )

        except Exception as e:
            logger.error(
                f"[LocalExecutor] session_id={self.session_id} - 消息路由失败: type={msg_type}, error={e}",
                exc_info=True,
            )

    @classmethod
    def create_for_cli(
        cls,
        console,
        *,
        auto_yes: bool = False,
        auto_approve_tools: bool = False,
        use_stream: bool = False,
        session_id: str | None = None,
    ) -> "LocalExecutor":
        """创建用于 CLI 的本地执行器"""
        from akg_agents.cli.constants import Defaults
        from akg_agents.cli.console import AKGConsole

        if isinstance(console, AKGConsole):
            akg_console = console
        else:
            akg_console = AKGConsole(
                console,
                use_stream=use_stream,
            )

        # 会话 id：用于 TUI resume + 录制目录
        sid = str(session_id or uuid.uuid4())
        os.environ["AKG_AGENTS_SESSION_ID"] = sid

        executor = cls(
            console=akg_console,
            session_id=sid,
        )
        try:
            akg_console.set_cancel_handler(executor.cancel_main_agent)
        except Exception:
            pass
        executor.auto_yes = bool(auto_yes)
        executor.common_auto_approve_tools = bool(auto_approve_tools)
        executor.use_stream = bool(use_stream)
        executor.workflow_name = Defaults.WORKFLOW_NAME
        return executor
