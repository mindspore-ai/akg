# Copyright 2026 Huawei Technologies Co., Ltd
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

"""
ReactTurnExecutorV2 - 基于 KernelAgent (core_v2) 的 CLI 执行器

与 V1 的核心区别：
- V1 使用 LangGraph 的 MainOpAgent，通过 astream(stream_mode="updates") 流式获取事件
- V2 使用 KernelAgent (ReActAgent)，直接调用 agent.run(user_input)，
  返回结构化 dict（status, output, plan_list, history 等）
- KernelAgent 内部自带 TraceSystem 管理历史，不依赖 LangGraph checkpointer
- 流式输出由 AgentBase._stream_enabled() 控制（环境变量/settings.json/ContextVar）
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Optional

from akg_agents.cli.messages import PanelDataMessage
from akg_agents.cli.runtime.message_sender import send_message

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReactTargetV2:
    """算子目标硬件配置"""
    framework: str
    backend: str
    arch: str
    dsl: str


class ReactTurnExecutorV2:
    """
    基于 KernelAgent 的 CLI 执行器

    - 每个 executor 实例持有一个 KernelAgent（含 TraceSystem，跨轮次保持历史）
    - run_turn() 执行一轮用户输入，返回 CLI 标准状态字典
    - 支持取消（通过 asyncio.Task.cancel）
    - 通过 send_message 推送面板/流式消息给 CLI
    """

    def __init__(
        self,
        *,
        session_id: str,
        config: dict,
        target: ReactTargetV2,
    ) -> None:
        self.session_id = str(session_id)
        self.config = dict(config or {})
        self.target = target

        self.running_task: Optional[asyncio.Task] = None
        self._last_panel_phase: str = ""

        # 创建 KernelAgent
        self._agent = self._create_kernel_agent()

    def _create_kernel_agent(self):
        """创建 KernelAgent 实例"""
        cfg = dict(self.config)

        # 从配置中获取 model_level
        model_level = (
            (cfg.get("agent_model_config") or {}).get("default") or "standard"
        )

        # task_id: 使用 session_id 作为任务 ID，保持跨轮次一致
        task_id = f"cli_{self.session_id}"

        from akg_agents.op.agents.kernel_agent import KernelAgent

        agent = KernelAgent(
            task_id=task_id,
            model_level=model_level,
            config=cfg,
            framework=self.target.framework,
            backend=self.target.backend,
            arch=self.target.arch,
            dsl=self.target.dsl,
        )

        # 注入 session_id 到 agent context，使得 run_llm() 创建的 LLMClient
        # 能通过 send_message 把流式 chunk 推送到 CLI
        agent.context["session_id"] = self.session_id

        logger.info(
            f"[ReactTurnExecutorV2] KernelAgent 已创建: task_id={task_id}, "
            f"session_id={self.session_id}, target={self.target}"
        )
        return agent

    def _panel_update_current(self, *, phase: str, op_name: str = "") -> None:
        """更新面板当前状态"""
        phase = str(phase or "").strip()
        op_name = str(op_name or "").strip()
        if not phase and not op_name:
            return
        # 降频：避免同一 phase 重复刷面板
        if phase and phase == self._last_panel_phase:
            return
        self._last_panel_phase = phase
        send_message(
            self.session_id,
            PanelDataMessage(
                action="update_current",
                data={"task_name": op_name, "phase": phase},
            ),
        )

    async def run_turn(self, user_input: str, use_stream: bool) -> dict:
        """
        执行一轮用户输入。

        KernelAgent.run() 返回的结构:
        - status: "success" | "error" | "waiting_for_user"
        - output: str (主要输出文本)
        - message: str (ask_user 时的提问)
        - error_information: str (错误时)
        - plan_list: list (执行计划)
        - history: list (执行历史)
        - total_actions: int (总动作数)
        - current_node: str (当前节点)

        映射为 CLI 标准状态:
        - current_step / should_continue / display_message / hint_message / workflow_name
        """
        user_input = (user_input or "").strip()
        if not user_input:
            raise ValueError("user_input is required")

        async def _run() -> dict:
            # 通知面板：开始处理
            self._panel_update_current(phase="thinking")

            # 调用 KernelAgent.run()
            result = await self._agent.run(user_input)

            if not isinstance(result, dict):
                logger.error(
                    f"[ReactTurnExecutorV2] KernelAgent.run() 返回非 dict: {type(result)}"
                )
                return {
                    "current_step": "error",
                    "should_continue": True,
                    "display_message": "Agent 返回了异常结果",
                    "hint_message": "",
                    "workflow_name": "kernel_agent",
                }

            status = str(result.get("status") or "").strip().lower()
            output = str(result.get("output") or "").strip()
            error_info = str(result.get("error_information") or "").strip()
            message = str(result.get("message") or "").strip()
            plan_list = result.get("plan_list") or []
            history = result.get("history") or []
            total_actions = result.get("total_actions", 0)

            # 发送面板计划信息
            if plan_list:
                self._panel_update_current(phase="executing_plan")
                send_message(
                    self.session_id,
                    PanelDataMessage(
                        action="update_plan",
                        data={"plan_list": plan_list},
                    ),
                )

            # 发送执行历史摘要
            if history:
                recent_tools = [
                    h.get("tool_name", "?") for h in history[-5:]
                ]
                logger.info(
                    f"[ReactTurnExecutorV2] 执行历史 ({len(history)} 条): "
                    f"最近工具: {recent_tools}"
                )

            # ====== 映射 status 到 CLI 标准状态 ======
            #
            # 流式输出说明：
            #   KernelAgent 内部 run_llm() 在流式模式下，LLMClient._generate_stream()
            #   会通过 send_message 把每个 chunk 以 LLMStreamMessage 推到 CLI
            #   的 stream_renderer（带行号的编号框），并在结束时发送 DisplayMessage("")
            #   触发 stream_renderer.finish()。
            #
            #   因此本层（run_turn）不需要再发送 LLMStreamMessage 或 DisplayMessage。
            #
            #   但注意：LLM 流式输出的是原始 JSON 指令（如 {"tool_name": "ask_user", ...}），
            #   而 result 中的 output/message 是 agent 提取后的纯文本。
            #   对于 waiting_for_user 和 error，这些文本没有经过流式通道，
            #   仍需通过 display_message 传给 runner 渲染。
            #
            #   对于 success（finish），LLM 的最后一次输出是 {"tool_name": "finish", ...}
            #   的 JSON，output 是 agent 构建的摘要，也没有被流式显示为纯文本。
            #   所以 success 时也需要传递 display_message。
            #

            if status == "success":
                self._panel_update_current(phase="completed")

                return {
                    "current_step": "completed",
                    "should_continue": False,
                    "display_message": output,
                    "hint_message": f"共执行 {total_actions} 个动作",
                    "workflow_name": "kernel_agent",
                    "plan_list": plan_list,
                }

            elif status == "waiting_for_user":
                # ask_user：Agent 需要用户回答问题
                display = message or output

                return {
                    "current_step": "waiting_for_user_input",
                    "should_continue": True,
                    "display_message": display,
                    "hint_message": "",
                    "workflow_name": "kernel_agent",
                }

            elif status == "error":
                self._panel_update_current(phase="error")

                error_display = error_info or output or "未知错误"

                return {
                    "current_step": "error",
                    "should_continue": True,
                    "display_message": error_display,
                    "hint_message": "",
                    "workflow_name": "kernel_agent",
                }

            else:
                # 未知状态，作为 "进行中" 处理
                logger.warning(
                    f"[ReactTurnExecutorV2] 未知 status: {status}, result={result}"
                )
                display = output or message or f"Agent 返回状态: {status}"

                return {
                    "current_step": status or "unknown",
                    "should_continue": True,
                    "display_message": display,
                    "hint_message": "",
                    "workflow_name": "kernel_agent",
                }

        # 让 cancel_main_agent 能取消到这一轮
        try:
            self.running_task = asyncio.create_task(_run())
            state = await self.running_task
            return state
        except asyncio.CancelledError:
            return {
                "current_step": "cancelled_by_user",
                "should_continue": True,
                "display_message": "⚠️ 操作已被用户取消",
                "hint_message": "",
                "workflow_name": "kernel_agent",
            }
        finally:
            self.running_task = None
