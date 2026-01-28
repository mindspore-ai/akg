from __future__ import annotations

import asyncio
import json
import logging
import uuid
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
        
        # 如果提供了 task_file_content，标记为跳过 OpTaskBuilder
        self._task_file_content: str | None = self.config.get("task_file_content")
        self._task_file_used: bool = False

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

    async def _rollback_incomplete_tool_calls(self) -> None:
        """
        回滚取消后不完整的对话历史。
        
        当 LLM 生成了 tool_calls 但工具还没执行完就被取消时，
        删除最后那条带有未完成 tool_calls 的 AIMessage，回到 tool 调用前的状态。
        """
        try:
            from langchain_core.messages import AIMessage, ToolMessage, RemoveMessage
            
            config = {"configurable": {"thread_id": self.thread_id}}
            
            # 获取当前状态
            state = await self._react_agent.agent.aget_state(config)
            if not state or not state.values:
                return
            
            messages = state.values.get("messages", [])
            if not messages:
                return
            
            # 收集已回复的 tool_call_id
            answered_tool_call_ids = set()
            for msg in messages:
                if isinstance(msg, ToolMessage):
                    tool_call_id = getattr(msg, "tool_call_id", None)
                    if tool_call_id:
                        answered_tool_call_ids.add(tool_call_id)
            
            # 从后往前找需要删除的消息（带有未完成 tool_calls 的 AIMessage）
            messages_to_remove = []
            for msg in reversed(messages):
                if isinstance(msg, AIMessage):
                    tool_calls = getattr(msg, "tool_calls", None) or []
                    if tool_calls:
                        # 检查是否有未完成的 tool_call
                        has_pending = False
                        for tc in tool_calls:
                            tc_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
                            if tc_id and tc_id not in answered_tool_call_ids:
                                has_pending = True
                                break
                        
                        if has_pending:
                            msg_id = getattr(msg, "id", None)
                            if msg_id:
                                messages_to_remove.append(RemoveMessage(id=msg_id))
                                logger.info(f"[ReactTurnExecutor] Removing incomplete AIMessage: {msg_id}")
                        break  # 只处理最后一个带 tool_calls 的 AIMessage
            
            if messages_to_remove:
                await self._react_agent.agent.aupdate_state(
                    config,
                    {"messages": messages_to_remove}
                )
                logger.info(f"[ReactTurnExecutor] Rolled back {len(messages_to_remove)} incomplete message(s)")
            
        except Exception as e:
            # 如果回滚失败，只记录警告，保持 thread_id 不变以保留对话历史
            logger.warning(f"[ReactTurnExecutor] Rollback failed: {e}, but keeping thread_id to preserve conversation history")
            # 注释掉重置 thread_id，以保留多轮对话历史
            # self.thread_id = f"{self.session_id}_{uuid.uuid4().hex[:8]}"

    async def run_turn(self, user_input: str, use_stream: bool) -> dict:
        """
        执行一轮用户输入。

        返回最小状态（供 CLI runner 渲染）：
        - current_step / should_continue / display_message / hint_message / workflow_name
        """
        user_input = (user_input or "").strip()
        if not user_input:
            raise ValueError("user_input is required")
        # 清理之前会话遗留的不完整状态（跨会话恢复、CLI 重启）
        await self._rollback_incomplete_tool_calls()

        # 如果提供了 task_file_content 且尚未使用，将其作为特殊指令注入
        if self._task_file_content and not self._task_file_used:
            self._task_file_used = True
            # 构建特殊指令：告诉 ReAct Agent 直接使用提供的 task_desc，跳过 OpTaskBuilder
            task_file_instruction = f"""用户已提供完整的 KernelBench 格式 task_desc 文件，请直接使用以下代码作为 task_code，**不要**调用 call_op_task_builder 进行转换。

用户的原始请求：{user_input}

已提供的 task_desc 代码：
```python
{self._task_file_content}
```

请直接调用 call_codeonly 或其他代码生成工具，使用上述 task_code 进行算子代码生成。"""
            user_input = task_file_instruction
            logger.info("[ReactTurnExecutor] Injected task_file_content, skipping OpTaskBuilder")

        stream_config = {"configurable": {"thread_id": self.thread_id}}

        display_message = ""
        hint_message = ""
        should_continue = True
        finished = False
        interrupted = False

        collected_content: list[str] = []
        collected_reasoning: list[str] = []
        # 允许“子 Agent 自己的 LLMStreamMessage”流式输出的工具集合：
        # - codeonly：单并发执行，允许流式
        # - op_task_builder：单线程，允许流式
        allow_tool_streaming = {"call_codeonly", "call_op_task_builder"}

        async def _run() -> None:
            nonlocal display_message, hint_message, should_continue, finished, interrupted
            # 如果上一轮触发了 interrupt，则这一轮必须用 Command(resume=...) 恢复
            if self._awaiting_resume:
                invoke_input: Any = Command(resume=user_input)
                self._awaiting_resume = False
            else:
                invoke_input = {"messages": [{"role": "user", "content": user_input}]}

            def _extract_kind_payload(evt: dict[str, Any]) -> tuple[str, Any]:
                """严格解析 updates event；不符合形态直接报错。"""
                if "__interrupt__" in evt:
                    if len(evt) != 1:
                        raise RuntimeError(
                            f"Unexpected interrupt event shape: event={evt!r}"
                        )
                    return "__interrupt__", evt.get("__interrupt__")
                if len(evt) != 1:
                    raise RuntimeError(
                        f"Unexpected updates event shape (expect single-key dict): event={evt!r}"
                    )
                return next(iter(evt.items()))

            def _should_ignore_node(kind: str) -> bool:
                # middleware(before_model) 节点：本项目 react_agent.py 使用了 @before_model(trim_messages)，
                # 因此 updates 流里会出现类似 "trim_messages.before_model" 的节点更新；CLI 不消费其更新。
                return str(kind).endswith(".before_model")

            def _emit_llm_chunk(*, chunk: str, is_reasoning: bool) -> None:
                if not chunk:
                    return
                if use_stream:
                    send_message(
                        self.session_id,
                        LLMStreamMessage(
                            agent="react",
                            chunk=chunk,
                            is_reasoning=bool(is_reasoning),
                        ),
                    )
                else:
                    if is_reasoning:
                        collected_reasoning.append(chunk)
                    else:
                        collected_content.append(chunk)

            def _handle_tool_call(tc: Any) -> None:
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

                    # 关键：如果即将调用“允许子 Agent 自己 stream”的工具，
                    # 先结束当前 ReAct 的 stream，避免两个来源的 LLMStreamMessage 混到同一个 StreamRenderer。
                    if use_stream and tool_name in allow_tool_streaming:
                        from ai_kernel_generator.cli.messages import DisplayMessage

                        send_message(self.session_id, DisplayMessage(text=""))

                # 如果模型直接调用 finish，通常最终答案在 tool output 中
                if tool_name == "finish":
                    # 不在这里置 finished：以 tools event 为准
                    return

            def _handle_model_payload(payload: Any) -> None:
                node_output = payload
                if not isinstance(node_output, dict):
                    raise TypeError(f"Invalid 'model' payload type: {type(node_output)}")

                messages = node_output.get("messages", []) or []
                for msg in messages:
                    # reasoning_content（DeepSeek reasoner 的 think）
                    reasoning = self._extract_reasoning_content(msg)
                    if reasoning:
                        _emit_llm_chunk(chunk=reasoning, is_reasoning=True)

                    # assistant content
                    content = str(getattr(msg, "content", "") or "")
                    if content:
                        _emit_llm_chunk(chunk=content, is_reasoning=False)

                    # tool calls（用于面板 phase）
                    tool_calls = getattr(msg, "tool_calls", None) or []
                    for tc in tool_calls:
                        _handle_tool_call(tc)

            def _handle_tools_payload(payload: Any) -> bool:
                """返回 True 表示本轮应立即结束（例如 finish）。"""
                nonlocal display_message, finished, should_continue

                tools_output = payload
                if not isinstance(tools_output, dict):
                    raise TypeError(f"Invalid 'tools' payload type: {type(tools_output)}")

                messages = tools_output.get("messages", []) or []
                for msg in messages:
                    tool_name = str(getattr(msg, "name", "") or "").strip()
                    content = str(getattr(msg, "content", "") or "")

                    if tool_name == "finish":
                        # basic_tools.finish 返回 final_answer（string）
                        display_message = content.strip() if content else ""
                        finished = True
                        should_continue = False
                        # 关键：立即结束本轮 astream，避免 finish 之后又触发一轮新的 model 调用而“卡住”
                        return True

                    # 有些 tool 可能返回结构化 JSON 字符串；这里不强依赖，只尽量提取
                    if (not finished) and (not use_stream) and content:
                        parsed = self._safe_json_loads(content)
                        if isinstance(parsed, dict) and "final_answer" in parsed:
                            display_message = str(parsed.get("final_answer") or "").strip()
                            finished = True
                            should_continue = False
                            return True

                return False

            async for event in self._react_agent.agent.astream(
                invoke_input, stream_config, stream_mode="updates"
            ):
                if not isinstance(event, dict):
                    continue

                kind, payload = _extract_kind_payload(event)

                match kind:
                    # interrupt：ask_user 会触发 GraphInterrupt，stream 中会出现 __interrupt__
                    case "__interrupt__":
                        interrupted = True
                        intrs = payload or ()
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
                        return

                    # model 节点输出：LLM 内容 + tool_calls
                    case "model":
                        _handle_model_payload(payload)

                    # tools 节点输出：Observation（含 finish 的最终答案）
                    case "tools":
                        if _handle_tools_payload(payload):
                            return

                    case _ if _should_ignore_node(str(kind)):
                        continue
                    case _:
                        raise RuntimeError(
                            f"Unexpected updates event node: {kind!r}, event={event!r}"
                        )

        # 让 cancel_main_agent 能取消到这一轮
        try:
            self.running_task = asyncio.create_task(_run())
            await self.running_task
        except asyncio.CancelledError:
            # 取消后回滚：删除最后那条带有未完成 tool_calls 的消息，回到 tool 调用前
            self._awaiting_resume = False
            await self._rollback_incomplete_tool_calls()
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
            "current_step": (
                "waiting_for_user_input"
                if interrupted
                else ("completed" if finished else "react")
            ),
            "should_continue": bool(should_continue),
            "display_message": str(display_message or ""),
            "hint_message": str(hint_message or ""),
            "workflow_name": "react",
        }

