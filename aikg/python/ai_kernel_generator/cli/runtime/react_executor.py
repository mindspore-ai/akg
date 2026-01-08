from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, Optional

from langgraph.types import Command

from ai_kernel_generator.cli.messages import LLMStreamMessage, PanelDataMessage
from ai_kernel_generator.cli.runtime.message_sender import send_message

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReactTarget:
    framework: str
    backend: str
    arch: str
    dsl: str


class ReactTurnExecutor:
    """
    ReAct 模式下的一轮对话执行器：
    - 复用同一个 react_agent（含 checkpointer）实现跨轮次记忆
    - 将 astream event 映射为 akg_cli 的消息体系（LLMStreamMessage/PanelDataMessage）
    """

    def __init__(
        self,
        *,
        session_id: str,
        config: dict,
        target: ReactTarget,
        thread_id: Optional[str] = None,
    ) -> None:
        self.session_id = str(session_id)
        self.thread_id = str(thread_id or session_id)
        self.config = dict(config or {})
        self.target = target

        self.running_task: Optional[asyncio.Task] = None

        self._react_agent = self._create_react_agent()
        self._last_panel_phase: str = ""
        self._awaiting_resume: bool = False

    def _create_react_agent(self):
        cfg = dict(self.config)

        model_preset = (
            (cfg.get("agent_model_config") or {}).get("default") or "deepseek_r1_default"
        )

        from ai_kernel_generator.core.llm.model_loader import create_langchain_chat_model
        from ai_kernel_generator.core.agent.react_agent import MainOpAgent as ReactMainOpAgent

        model = create_langchain_chat_model(model_preset)
        return ReactMainOpAgent(
            config=cfg,
            model=model,
            framework=self.target.framework,
            backend=self.target.backend,
            arch=self.target.arch,
            dsl=self.target.dsl,
        )

    @staticmethod
    def _extract_reasoning_content(msg: Any) -> str:
        try:
            kwargs = getattr(msg, "additional_kwargs", None) or {}
            return str(kwargs.get("reasoning_content") or "")
        except Exception:
            return ""

    @staticmethod
    def _safe_json_loads(s: str) -> Any:
        s = (s or "").strip()
        if not s:
            return None
        if not (s.startswith("{") or s.startswith("[")):
            return None
        try:
            return json.loads(s)
        except Exception:
            return None

    def _panel_update_current(self, *, phase: str, op_name: str = "") -> None:
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

        返回最小状态（供 CLI runner 渲染）：
        - current_step / should_continue / display_message / hint_message / workflow_name
        """
        user_input = (user_input or "").strip()
        if not user_input:
            raise ValueError("user_input is required")

        stream_config = {"configurable": {"thread_id": self.thread_id}}

        display_message = ""
        hint_message = ""
        should_continue = True
        finished = False
        interrupted = False

        collected_content: list[str] = []
        collected_reasoning: list[str] = []

        async def _run() -> None:
            nonlocal display_message, hint_message, should_continue, finished, interrupted
            # 如果上一轮触发了 interrupt，则这一轮必须用 Command(resume=...) 恢复
            if self._awaiting_resume:
                invoke_input: Any = Command(resume=user_input)
                self._awaiting_resume = False
            else:
                invoke_input = {"messages": [{"role": "user", "content": user_input}]}

            async for event in self._react_agent.agent.astream(
                invoke_input, stream_config, stream_mode="updates"
            ):
                if not isinstance(event, dict):
                    continue

                # interrupt：ask_user 会触发 GraphInterrupt，stream 中会出现 __interrupt__
                if "__interrupt__" in event:
                    interrupted = True
                    intrs = event.get("__interrupt__") or ()
                    value = ""
                    try:
                        if intrs and len(intrs) > 0:
                            value = getattr(intrs[0], "value", "")  # Interrupt.value
                    except Exception:
                        value = ""
                    display_message = str(value or "").strip()
                    should_continue = True
                    self._awaiting_resume = True
                    # interrupt 后本轮结束，等待用户下一次输入（resume）
                    continue

                # model/agent 节点输出：LLM 内容 + tool_calls
                node_output = None
                if "model" in event:
                    node_output = event.get("model")
                elif "agent" in event:
                    node_output = event.get("agent")

                if isinstance(node_output, dict):
                    messages = node_output.get("messages", []) or []
                    for msg in messages:
                        # reasoning_content（DeepSeek reasoner 的 think）
                        reasoning = self._extract_reasoning_content(msg)
                        if reasoning:
                            if use_stream:
                                send_message(
                                    self.session_id,
                                    LLMStreamMessage(
                                        agent="react",
                                        chunk=reasoning,
                                        is_reasoning=True,
                                    ),
                                )
                            else:
                                collected_reasoning.append(reasoning)

                        # assistant content
                        content = str(getattr(msg, "content", "") or "")
                        if content:
                            if use_stream:
                                send_message(
                                    self.session_id,
                                    LLMStreamMessage(
                                        agent="react",
                                        chunk=content,
                                        is_reasoning=False,
                                    ),
                                )
                            else:
                                collected_content.append(content)

                        # tool calls（用于面板 phase）
                        tool_calls = getattr(msg, "tool_calls", None) or []
                        for tc in tool_calls:
                            if isinstance(tc, dict):
                                tool_name = str(tc.get("name") or "").strip()
                                tool_args = tc.get("args") or {}
                            else:
                                tool_name = str(getattr(tc, "name", "") or "").strip()
                                tool_args = getattr(tc, "args", {}) or {}

                            op_name = ""
                            try:
                                if isinstance(tool_args, dict):
                                    op_name = str(tool_args.get("op_name") or "").strip()
                            except Exception:
                                op_name = ""

                            if tool_name:
                                self._panel_update_current(phase=tool_name, op_name=op_name)

                            # 如果模型直接调用 finish，通常最终答案在 tool output 中
                            if tool_name == "finish":
                                # 不在这里置 finished：以 tools event 为准
                                pass

                # tools 节点输出：Observation（含 finish 的最终答案）
                tools_output = event.get("tools")
                if isinstance(tools_output, dict):
                    messages = tools_output.get("messages", []) or []
                    for msg in messages:
                        tool_name = str(getattr(msg, "name", "") or "").strip()
                        content = str(getattr(msg, "content", "") or "")

                        if tool_name == "finish":
                            # basic_tools.finish 返回 final_answer（string）
                            display_message = content.strip() if content else ""
                            finished = True
                            should_continue = False
                            # 不 break：让流式事件自然收尾

                        # 有些 tool 可能返回结构化 JSON 字符串；这里不强依赖，只尽量提取
                        if (not finished) and (not use_stream) and content:
                            parsed = self._safe_json_loads(content)
                            if isinstance(parsed, dict) and "final_answer" in parsed:
                                display_message = str(parsed.get("final_answer") or "").strip()
                                finished = True
                                should_continue = False

        # 让 cancel_main_agent 能取消到这一轮
        try:
            self.running_task = asyncio.create_task(_run())
            await self.running_task
        except asyncio.CancelledError:
            return {
                "current_step": "cancelled_by_user",
                "should_continue": True,
                "display_message": "⚠️ 操作已被用户取消",
                "hint_message": "",
                "workflow_name": "react",
            }
        finally:
            self.running_task = None

        if not finished and not interrupted:
            # 非 finish：视为等待用户下一轮输入
            should_continue = True
            if not use_stream:
                content_text = "".join(collected_content).strip()
                reasoning_text = "".join(collected_reasoning).strip()
                if content_text:
                    display_message = content_text
                    hint_message = reasoning_text
                else:
                    display_message = reasoning_text
                    hint_message = ""

        # 流式收尾：触发 AKGConsole.stream_renderer.finish()
        if use_stream:
            from ai_kernel_generator.cli.messages import DisplayMessage

            send_message(self.session_id, DisplayMessage(text=""))

        return {
            "current_step": "waiting_for_user_input" if interrupted else "react",
            "should_continue": bool(should_continue),
            "display_message": str(display_message or ""),
            "hint_message": str(hint_message or ""),
            "workflow_name": "react",
        }

