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

"""本地执行器 - 直接调用 MainOpAgent，无需 server"""

import logging
import os
import sys
import uuid
from contextlib import redirect_stdout
from io import StringIO
from typing import Dict, Any, Optional

from ai_kernel_generator.core.agent.main_op_agent import MainOpAgent
from ai_kernel_generator.config.config_validator import load_config
from ai_kernel_generator.cli.runtime.message_sender import (
    register_message_sender,
    unregister_message_sender,
)
from ai_kernel_generator.utils.main_op_agent_display import (
    format_display_message,
    get_hint_message,
    is_simple_command,
)

logger = logging.getLogger(__name__)


class LocalExecutor:
    """本地执行器，直接调用 MainOpAgent"""

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
        self.default_user_input: str = ""
        self.use_stream: bool = False
        self.workflow_name: str = ""
        self.notifier = None

        # 会话状态管理
        self._main_agent_state: dict | None = None
        self._main_agent: MainOpAgent | None = None

    def _load_workflow_config(self, backend: str, dsl: str) -> Dict:
        """加载工作流配置"""
        config = load_config(dsl, backend=backend)

        # 兜底补齐 agent_model_config
        if "agent_model_config" not in config or not isinstance(
            config.get("agent_model_config"), dict
        ):
            config["agent_model_config"] = {}
        mc = config["agent_model_config"]

        base_default = mc.get("default") or "deepseek_r1_default"
        mc.setdefault("default", base_default)
        if "log_dir" not in config or config["log_dir"] is None:
            config["log_dir"] = "~/aikg_logs"
        return config

    def _create_main_op_agent(
        self,
        framework: str,
        backend: str,
        arch: str,
        dsl: str,
        rag: bool = False,
        output_path: str | None = None,
    ) -> MainOpAgent:
        """创建 MainOpAgent 实例"""
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

        # RAG 参数处理：优先使用 state 中的值
        if self._main_agent_state and isinstance(self._main_agent_state, dict):
            if "rag" in self._main_agent_state:
                config["rag"] = self._main_agent_state.get("rag", False)
                logger.info(
                    f"[RAG] Using rag={config['rag']} from state for session {self.session_id}"
                )
            else:
                config["rag"] = bool(rag)
                logger.info(
                    f"[RAG] Using rag={config['rag']} from request for session {self.session_id}"
                )
        else:
            config["rag"] = bool(rag)
            logger.info(
                f"[RAG] Using rag={config['rag']} from request (default) for session {self.session_id}"
            )

        return MainOpAgent(
            config=config,
            framework=framework,
            backend=backend,
            arch=arch,
            dsl=dsl,
        )

    def _build_response_state(self, state: dict) -> dict:
        """构建响应状态"""
        if not state.get("display_message"):
            state["display_message"] = format_display_message(state)
        if not state.get("hint_message"):
            state["hint_message"] = get_hint_message(state)

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
    ) -> Dict[str, Any]:
        """执行 MainOpAgent 对话（start/continue），返回状态"""
        if not (user_input or "").strip():
            raise ValueError("user_input is required")

        effective_use_stream = (
            bool(use_stream) if use_stream is not None else bool(self.use_stream)
        )

        # 设置 Stream 环境变量
        os.environ["AIKG_STREAM_OUTPUT"] = "on" if effective_use_stream else "off"

        # 注册消息发送器（本地模式：直接调用 console）
        def _local_message_sender(message):
            """本地消息发送器：直接路由到 console"""

            # 直接路由到 console
            self._route_to_console(message)

        register_message_sender(self.session_id, _local_message_sender)

        try:
            # 判断是 start 还是 continue
            prev_state = self._main_agent_state
            should_start = not isinstance(prev_state, dict)

            # 创建 agent
            agent = self._create_main_op_agent(
                framework=framework,
                backend=backend,
                arch=arch,
                dsl=dsl,
                rag=rag,
                output_path=output_path,
            )
            self._main_agent = agent

            # 重定向 stdout 以避免 main_op_agent 执行过程中的 print 输出干扰 prompt_toolkit
            # 使用 StringIO 捕获输出，但不显示（或者可以记录到日志）
            null_output = StringIO()
            
            if should_start:
                # start action: 开启新对话
                user_input = user_input or ""
                if not user_input.strip():
                    raise ValueError("user_input is required for start action")

                self._main_agent_state = None
                # 重定向 stdout 以避免 print 输出到控制台
                with redirect_stdout(null_output):
                    state = await agent.start_conversation(user_request=user_input)
                # 确保将 rag 值保存到 state 中
                if not isinstance(state, dict):
                    state = {}
                state["rag"] = bool(rag)
                logger.info(
                    f"[RAG] Saved rag={state['rag']} to state for session {self.session_id} (start action)"
                )
                self._main_agent_state = state
            else:
                # continue action: 继续对话
                prev = self._main_agent_state
                if not isinstance(prev, dict):
                    raise ValueError(
                        "no main agent state found for this session; call start first"
                    )

                user_input = user_input or ""
                if not user_input.strip():
                    raise ValueError("user_input is required for continue action")

                is_cmd, command_type = is_simple_command(user_input)
                if is_cmd and command_type == "exit":
                    self._main_agent_state = None
                    return {
                        "current_step": "cancelled",
                        "should_continue": False,
                        "display_message": "Conversation ended.",
                        "hint_message": "",
                    }

                if "config" not in prev:
                    prev["config"] = agent.config
                # 重定向 stdout 以避免 print 输出到控制台
                with redirect_stdout(null_output):
                    state = await agent.continue_conversation(
                        current_state=prev,
                        user_input=user_input,
                        action="auto",
                    )
                self._main_agent_state = state

            # 更新 workflow_name
            try:
                wf_name = str(
                    state.get("workflow_name") or state.get("sub_workflow") or ""
                ).strip()
                if wf_name:
                    self.workflow_name = wf_name
            except Exception as e:
                logger.debug(
                    "[LocalExecutor] update workflow_name from metadata failed",
                    exc_info=e,
                )

            return self._build_response_state(state)

        finally:
            unregister_message_sender(self.session_id)

    async def cancel_main_agent(self, reason: str = "cancelled by client") -> bool:
        """取消当前 main_agent"""
        if self._main_agent:
            try:
                self._main_agent._cancellation_requested = True
                logger.info(
                    f"[LocalExecutor] Cancellation requested for session {self.session_id}: {reason}"
                )
                return True
            except Exception as e:
                logger.warning(
                    f"Cancel main_agent failed: session_id={self.session_id}, err={e}"
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
            from ai_kernel_generator.cli.messages import (
                DisplayMessage,
                LLMStreamMessage,
                PanelDataMessage,
            )

            if isinstance(message, DisplayMessage):
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
        use_stream: bool = False,
        session_id: str | None = None,
    ) -> "LocalExecutor":
        """创建用于 CLI 的本地执行器"""
        from ai_kernel_generator.cli.constants import Defaults
        from ai_kernel_generator.cli.console import AKGConsole

        if isinstance(console, AKGConsole):
            akg_console = console
        else:
            akg_console = AKGConsole(
                console,
                use_stream=use_stream,
            )

        # 会话 id：用于 TUI resume + 录制目录
        sid = str(session_id or uuid.uuid4())
        os.environ["AIKG_SESSION_ID"] = sid

        # 设置 Stream 环境变量
        os.environ["AIKG_STREAM_OUTPUT"] = "on" if use_stream else "off"

        executor = cls(
            console=akg_console,
            session_id=sid,
        )
        try:
            akg_console.set_cancel_handler(executor.cancel_main_agent)
        except Exception:
            pass
        executor.auto_yes = bool(auto_yes)
        executor.use_stream = bool(use_stream)
        executor.workflow_name = Defaults.WORKFLOW_NAME
        return executor
