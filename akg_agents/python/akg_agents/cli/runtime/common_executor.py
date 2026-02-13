from __future__ import annotations

import asyncio
import json
import logging
import uuid
from pathlib import Path
from typing import Any, Optional

from langgraph.types import Command
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from akg_agents.cli.messages import DisplayMessage, LLMStreamMessage
from akg_agents.cli.runtime.message_sender import send_message
from akg_agents.cli.runtime.react_utils import (
    DEFAULT_MAX_MESSAGES,
    create_checkpointer,
    trim_messages,
)
from akg_agents.core.llm.model_loader import create_langchain_chat_model

from akg_agents.cli.runtime.common_constants import (
    COMMON_SYSTEM_PROMPT,
    COMPACTION_SYSTEM_PROMPT,
    COMPACTION_USER_PROMPT,
    PLAN_MODE_SUFFIX,
)
from akg_agents.cli.runtime.common_skills import build_common_skills_metadata
from akg_agents.cli.runtime.common_support import DeltaFormatter, WorkspacePaths
from akg_agents.cli.runtime.common_tools import CommonToolState, build_common_tools

logger = logging.getLogger(__name__)

__all__ = ["CommonToolState", "CommonTurnExecutor", "build_common_tools"]


class RunTurnProcessor:
    def __init__(self, executor: "CommonTurnExecutor", user_input: str, use_stream: bool):
        self.executor = executor
        self.user_input = (user_input or "").strip()
        self.use_stream = bool(use_stream)
        self.pending_tool_calls: dict[str, dict] = {}
        self.display_message = ""
        self.hint_message = ""
        self.should_continue = True
        self.finished = False
        self.interrupted = False
        self.auto_input: Optional[str] = None
        self.collected_content: list[str] = []
        self.collected_reasoning: list[str] = []

    async def run(self) -> dict:
        self._validate_input()
        self.executor._refresh_agent_if_needed()
        await self._prepare_state()
        auto_token = self._enable_auto_approve()
        try:
            await self._stream_events()
        except asyncio.CancelledError:
            return await self._handle_cancel()
        finally:
            self._reset_auto_approve(auto_token)
        self._finalize_non_stream_output()
        self._emit_stream_end()
        return self._result_payload()

    def _validate_input(self) -> None:
        if not self.user_input:
            raise ValueError("user_input is required")

    async def _prepare_state(self) -> None:
        if self.executor._awaiting_resume:
            return
        await self.executor._rollback_incomplete_tool_calls()
        await self.executor._maybe_auto_compact()

    def _enable_auto_approve(self) -> Optional[str]:
        from akg_agents.core.tools.basic_tools import set_tool_auto_approve

        return set_tool_auto_approve(self.executor._auto_approve_tools)

    def _reset_auto_approve(self, token: Optional[str]) -> None:
        from akg_agents.core.tools.basic_tools import reset_tool_auto_approve

        reset_tool_auto_approve(token)

    async def _handle_cancel(self) -> dict:
        self.executor._awaiting_resume = False
        await self.executor._rollback_incomplete_tool_calls()
        return {
            "current_step": "cancelled_by_user",
            "should_continue": True,
            "display_message": "Operation cancelled by user",
            "hint_message": "",
            "workflow_name": "react",
        }

    def _emit_stream_end(self) -> None:
        if self.use_stream:
            send_message(self.executor.session_id, DisplayMessage(text=""))

    def _result_payload(self) -> dict:
        return {
            "current_step": self._current_step(),
            "should_continue": bool(self.should_continue),
            "display_message": str(self.display_message or ""),
            "hint_message": str(self.hint_message or ""),
            "workflow_name": "react",
            **({"auto_input": self.auto_input} if self.auto_input else {}),
        }

    def _current_step(self) -> str:
        if self.interrupted:
            return "waiting_for_user_input"
        return "completed" if self.finished else "react"

    async def _stream_events(self) -> None:
        invoke_input = self._build_invoke_input()
        self._set_last_raw_input(invoke_input)
        stream_config = {"configurable": {"thread_id": self.executor.thread_id}}
        async for event in self.executor._agent.astream(
            invoke_input, stream_config, stream_mode="updates"
        ):
            await self._handle_event(event)
            if self.finished or self.interrupted:
                return

    async def _handle_event(self, event: dict[str, Any]) -> None:
        if not isinstance(event, dict):
            return
        kind, payload = self._extract_kind_payload(event)
        if kind == "__interrupt__":
            self._handle_interrupt(payload)
            return
        if kind == "model":
            self._handle_model_payload(payload)
            await self.executor._emit_context_usage("model")
            return
        if kind == "tools":
            self._handle_tools_payload(payload)
            return
        if self._should_ignore_node(kind):
            return
        raise RuntimeError(f"Unexpected updates event node: {kind!r}, event={event!r}")

    def _build_invoke_input(self) -> Any:
        if self.executor._awaiting_resume:
            self.executor._awaiting_resume = False
            return Command(resume=self.user_input)
        return self._build_message_input()

    def _build_message_input(self) -> dict:
        summary = self.executor._pending_compaction_summary
        if summary:
            self.executor._pending_compaction_summary = None
            return {
                "messages": [
                    {"role": "assistant", "content": f"Summary of previous session:\n{summary}"},
                    {"role": "user", "content": self.user_input},
                ]
            }
        return {"messages": [{"role": "user", "content": self.user_input}]}

    def _set_last_raw_input(self, invoke_input: Any) -> None:
        self.executor.last_raw_llm_input = {
            "system_prompt": self.executor._system_prompt,
            "model_preset": self.executor._model_preset,
            "tools": [
                {"name": getattr(tool, "name", ""), "description": getattr(tool, "description", "")}
                for tool in (self.executor.tools or [])
            ],
            "invoke_input": invoke_input,
            "thread_id": self.executor.thread_id,
        }

    def _extract_kind_payload(self, event: dict[str, Any]) -> tuple[str, Any]:
        if "__interrupt__" in event:
            if len(event) != 1:
                raise RuntimeError(f"Unexpected interrupt event shape: event={event!r}")
            return "__interrupt__", event.get("__interrupt__")
        if len(event) != 1:
            raise RuntimeError(f"Unexpected updates event shape (expect single-key dict): event={event!r}")
        return next(iter(event.items()))

    @staticmethod
    def _should_ignore_node(kind: str) -> bool:
        return str(kind).endswith(".before_model")

    def _handle_interrupt(self, payload: Any) -> None:
        self.interrupted = True
        value = self._interrupt_value(payload)
        self.display_message = str(value or "").strip()
        self.should_continue = True
        self.executor._awaiting_resume = True

    @staticmethod
    def _interrupt_value(payload: Any) -> str:
        try:
            if payload and len(payload) > 0:
                return getattr(payload[0], "value", "")
        except Exception:
            return ""
        return ""

    def _handle_model_payload(self, payload: Any) -> None:
        if not isinstance(payload, dict):
            raise TypeError(f"Invalid 'model' payload type: {type(payload)}")
        messages = payload.get("messages", []) or []
        self.executor._update_last_usage(messages)
        for msg in messages:
            self._emit_reasoning(msg)
            self._emit_content(msg)
            self._collect_tool_calls(msg)

    def _emit_reasoning(self, msg: Any) -> None:
        reasoning = self.executor._extract_reasoning_content(msg)
        if reasoning:
            self._emit_llm_chunk(reasoning, is_reasoning=True)

    def _emit_content(self, msg: Any) -> None:
        content = str(getattr(msg, "content", "") or "")
        if content:
            self._emit_llm_chunk(content, is_reasoning=False)

    def _collect_tool_calls(self, msg: Any) -> None:
        for tc in getattr(msg, "tool_calls", None) or []:
            tc_id, tc_name, tc_args = self._tool_call_parts(tc)
            if tc_id:
                self.pending_tool_calls[tc_id] = {"name": tc_name, "args": tc_args}

    @staticmethod
    def _tool_call_parts(tc: Any) -> tuple[str, str, Any]:
        if isinstance(tc, dict):
            return tc.get("id") or "", tc.get("name") or "", tc.get("args")
        return getattr(tc, "id", "") or "", getattr(tc, "name", "") or "", getattr(tc, "args", None)

    def _emit_llm_chunk(self, chunk: str, is_reasoning: bool) -> None:
        if not chunk:
            return
        if self.use_stream:
            self._stream_chunk(chunk, is_reasoning)
            return
        target = self.collected_reasoning if is_reasoning else self.collected_content
        target.append(chunk)

    def _stream_chunk(self, chunk: str, is_reasoning: bool) -> None:
        send_message(
            self.executor.session_id,
            LLMStreamMessage(agent="react", chunk=chunk, is_reasoning=bool(is_reasoning)),
        )
        send_message(self.executor.session_id, DisplayMessage(text=""))

    def _handle_tools_payload(self, payload: Any) -> None:
        if not isinstance(payload, dict):
            raise TypeError(f"Invalid 'tools' payload type: {type(payload)}")
        for msg in payload.get("messages", []) or []:
            if self._handle_tool_message(msg):
                return

    def _handle_tool_message(self, msg: Any) -> bool:
        tool_name = str(getattr(msg, "name", "") or "").strip()
        content = str(getattr(msg, "content", "") or "")
        content = self._format_tool_content(tool_name, content)
        if tool_name == "finish":
            return self._handle_finish(content)
        if tool_name in {"plan-enter", "plan-exit"}:
            return self._handle_plan_switch(tool_name, content)
        self._emit_tool_output(msg, tool_name, content)
        return self._handle_final_answer(content)

    def _format_tool_content(self, tool_name: str, content: str) -> str:
        if tool_name == "todowrite":
            return self.executor._format_todos_output(content) or content
        if tool_name in {"edit", "multiedit", "write", "apply_patch"}:
            return self.executor._delta_formatter.format(content)
        return content

    def _handle_finish(self, content: str) -> bool:
        self.display_message = content.strip() if content else ""
        self.finished = True
        self.should_continue = False
        return True

    def _handle_plan_switch(self, tool_name: str, content: str) -> bool:
        self.display_message = content.strip() if content else ""
        self.finished = True
        self.should_continue = True
        content_prefix = (content or "").lstrip()
        if content_prefix.startswith("[NOOP]"):
            self.auto_input = None
        else:
            self.auto_input = self._plan_auto_input(tool_name)
        return True

    def _plan_auto_input(self, tool_name: str) -> str | None:
        plan_path = self.executor._plan_display_path()
        if tool_name == "plan-enter" and self.executor._tool_state.mode == "plan":
            return (
                "You are now in plan mode. "
                f"The plan file is at {plan_path}. "
                "Do not call plan-enter again. Begin planning."
            )
        if tool_name == "plan-exit" and self.executor._tool_state.mode == "build":
            return (
                "The plan at "
                f"{plan_path} "
                "has been approved. You can now edit files. Execute the plan."
            )
        return None

    def _emit_tool_output(self, msg: Any, tool_name: str, content: str) -> None:
        if not content:
            return
        tool_label = f"[tool:{tool_name or 'unknown'}]"
        args_line = self._tool_args_line(msg, tool_name)
        payload_text = f"{tool_label}\n{args_line}{content}".rstrip()
        send_message(self.executor.session_id, DisplayMessage(text=payload_text))

    def _tool_args_line(self, msg: Any, tool_name: str) -> str:
        if tool_name in {"edit", "multiedit", "write", "apply_patch"}:
            return ""
        tool_call_id = str(getattr(msg, "tool_call_id", "") or "")
        tool_args = self.pending_tool_calls.get(tool_call_id, {}).get("args") if tool_call_id else None
        args_text = self._format_args(tool_args)
        return f"args: {args_text}\n" if args_text else ""

    @staticmethod
    def _format_args(tool_args: Any) -> str:
        if tool_args is None:
            return ""
        if isinstance(tool_args, str):
            try:
                return json.dumps(json.loads(tool_args), ensure_ascii=False)
            except Exception:
                return tool_args
        try:
            return json.dumps(tool_args, ensure_ascii=False)
        except Exception:
            return str(tool_args)

    def _handle_final_answer(self, content: str) -> bool:
        if self.finished or self.use_stream or not content:
            return False
        parsed = self.executor._safe_json_loads(content)
        if isinstance(parsed, dict) and "final_answer" in parsed:
            self.display_message = str(parsed.get("final_answer") or "").strip()
            self.finished = True
            self.should_continue = False
            return True
        return False

    def _finalize_non_stream_output(self) -> None:
        if self.finished or self.interrupted or self.use_stream:
            return
        content_text = "".join(self.collected_content).strip()
        reasoning_text = "".join(self.collected_reasoning).strip()
        if content_text:
            self.display_message = content_text
            self.hint_message = reasoning_text
        else:
            self.display_message = reasoning_text
            self.hint_message = ""


class CommonTurnExecutor:
    """ReAct executor for common tasks (no target config)."""

    def __init__(
        self,
        *,
        session_id: str,
        config: dict,
        thread_id: Optional[str] = None,
    ) -> None:
        self.session_id = str(session_id)
        self.thread_id = str(thread_id or session_id)
        self.config = dict(config or {})
        self._init_state()
        self._agent = self._create_agent()
        self.last_raw_llm_input: Optional[dict] = None

    def _init_state(self) -> None:
        self.running_task: Optional[asyncio.Task] = None
        self._awaiting_resume = False
        self._auto_approve_tools = bool(self.config.get("auto_approve_tools"))
        self._mode = str(self.config.get("mode") or "build")
        self._tool_state = CommonToolState(self.session_id)
        self._tool_state.mode = self._mode
        self._last_usage: dict[str, Optional[int]] | None = None
        self._checkpointer = None
        self._middleware = None
        self._delta_formatter = DeltaFormatter()
        self._init_compaction_settings()
        self._model = None
        self._pending_compaction_summary: str | None = None

    def _init_compaction_settings(self) -> None:
        self._max_messages = self._coerce_int(self.config.get("max_messages")) or DEFAULT_MAX_MESSAGES
        self._auto_compact_ratio = self._coerce_ratio(self.config.get("auto_compact_ratio"), 0.8)
        self._enable_auto_compact = bool(self.config.get("enable_auto_compact", True))
        self._compaction_system_prompt = str(
            self.config.get("compaction_system_prompt") or COMPACTION_SYSTEM_PROMPT
        )
        self._compaction_user_prompt = str(
            self.config.get("compaction_user_prompt") or COMPACTION_USER_PROMPT
        )

    def _create_agent(self):
        cfg = dict(self.config)
        self._model_preset = self._get_model_preset(cfg)
        self._system_prompt = self._build_system_prompt()
        self._inject_skills_metadata()
        self._model = create_langchain_chat_model(self._model_preset)
        self.tools = build_common_tools(self._tool_state)
        agent_kwargs = self._build_agent_kwargs(cfg)
        return create_agent(**agent_kwargs)

    def _get_model_preset(self, cfg: dict) -> str:
        return (cfg.get("agent_model_config") or {}).get("default") or "deepseek_r1_default"

    def _inject_skills_metadata(self) -> None:
        skills_metadata, skills_count = build_common_skills_metadata()
        if skills_metadata:
            self._system_prompt = f"{self._system_prompt}\n\n{skills_metadata}"
            logger.info("Injected %d common skills into system prompt", skills_count)

    def _build_agent_kwargs(self, cfg: dict) -> dict[str, Any]:
        if self._checkpointer is None and cfg.get("enable_memory", True):
            backend = str(cfg.get("memory_backend") or "memory")
            self._checkpointer = create_checkpointer(backend)
        if self._middleware is None:
            self._middleware = [trim_messages] if cfg.get("enable_trim", True) else None
        agent_kwargs = {
            "model": self._model,
            "tools": self.tools,
            "system_prompt": self._system_prompt,
        }
        if self._checkpointer is not None:
            agent_kwargs["checkpointer"] = self._checkpointer
        if self._middleware:
            agent_kwargs["middleware"] = self._middleware
        return agent_kwargs

    def _build_system_prompt(self) -> str:
        if str(self._mode).lower() != "plan":
            return COMMON_SYSTEM_PROMPT
        plan_path = self._tool_state.plan_path or str((WorkspacePaths.from_cwd().root / "plan.md").resolve())
        return COMMON_SYSTEM_PROMPT + PLAN_MODE_SUFFIX + f"\nPlan file: {plan_path}\n"

    def _refresh_agent_if_needed(self) -> None:
        desired_mode = str(getattr(self._tool_state, "mode", "build") or "build")
        if desired_mode != self._mode:
            self._mode = desired_mode
        desired_prompt = self._build_system_prompt()
        if desired_prompt != self._system_prompt:
            self._system_prompt = desired_prompt
            self._agent = self._create_agent()

    def _plan_display_path(self) -> str:
        plan_path = self._tool_state.plan_path or str((WorkspacePaths.from_cwd().root / "plan.md").resolve())
        try:
            return str(Path(plan_path).resolve().relative_to(WorkspacePaths.from_cwd().root))
        except Exception:
            return plan_path

    def _ensure_plan_path(self) -> str:
        if not self._tool_state.plan_path:
            self._tool_state.plan_path = str((WorkspacePaths.from_cwd().root / "plan.md").resolve())
        return self._tool_state.plan_path

    def enter_plan_mode(self) -> dict:
        plan_path = self._ensure_plan_path()
        self._tool_state.mode = "plan"
        self._refresh_agent_if_needed()
        display_message = (
            f"User confirmed switch to plan mode. Plan file: {plan_path}. "
            "Begin planning and call plan-exit when ready."
        )
        auto_input = (
            "You are now in plan mode. "
            f"The plan file is at {self._plan_display_path()}. "
            "Do not call plan-enter again. Begin planning."
        )
        return {
            "current_step": "completed",
            "should_continue": True,
            "display_message": display_message,
            "hint_message": "",
            "workflow_name": "react",
            "auto_input": auto_input,
        }

    @staticmethod
    def _extract_reasoning_content(msg: Any) -> str:
        try:
            kwargs = getattr(msg, "additional_kwargs", None) or {}
            return str(kwargs.get("reasoning_content") or "")
        except Exception:
            return ""

    @staticmethod
    def _safe_json_loads(s: str) -> Any:
        text = (s or "").strip()
        if not text or not (text.startswith("{") or text.startswith("[")):
            return None
        try:
            return json.loads(text)
        except Exception:
            return None

    def _format_todos_output(self, content: str) -> str | None:
        items = self._extract_todo_items(content)
        if not isinstance(items, list):
            return None
        lines = [line for line in (self._format_todo_item(i) for i in items) if line]
        if not lines:
            return "Todos: [empty]"
        return "\n".join([f"Todos ({len(lines)}):", *lines])

    def _extract_todo_items(self, content: str) -> list | None:
        parsed = self._safe_json_loads(content)
        if parsed is None:
            return None
        if isinstance(parsed, dict):
            return parsed.get("todos") or parsed.get("items") or parsed.get("tasks")
        return parsed if isinstance(parsed, list) else None

    def _format_todo_item(self, item: Any) -> str | None:
        text, status, priority, item_id = self._todo_fields(item)
        if not text:
            return None
        box = self._todo_box(status)
        suffix = self._todo_suffix(priority, item_id)
        return f"{box} {text}{suffix}"

    @staticmethod
    def _todo_fields(item: Any) -> tuple[str, str, str, str]:
        if not isinstance(item, dict):
            return str(item).strip(), "pending", "", ""
        text = str(
            item.get("content")
            or item.get("name")
            or item.get("title")
            or item.get("text")
            or item.get("description")
            or ""
        ).strip()
        status = str(item.get("status") or "pending").strip().lower()
        priority = str(item.get("priority") or "").strip().lower()
        item_id = str(item.get("id") or "").strip()
        return text, status, priority, item_id

    @staticmethod
    def _todo_box(status: str) -> str:
        if status in {"done", "completed", "complete", "finished"}:
            return "[x]"
        if status in {"in_progress", "doing", "active"}:
            return "[-]"
        return "[ ]"

    @staticmethod
    def _todo_suffix(priority: str, item_id: str) -> str:
        extras = [f"prio={priority}"] if priority else []
        if item_id:
            extras.append(f"id={item_id}")
        return f" ({', '.join(extras)})" if extras else ""

    @staticmethod
    def _content_to_text(content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [CommonTurnExecutor._part_text(p) for p in content]
            return "\n".join([p for p in parts if p])
        return str(content)

    @staticmethod
    def _part_text(part: Any) -> str:
        if not isinstance(part, dict):
            return str(part)
        text = part.get("text")
        if text is not None:
            return str(text)
        try:
            return json.dumps(part, ensure_ascii=False)
        except Exception:
            return str(part)

    def _messages_to_transcript(self, messages: list[Any]) -> str:
        lines: list[str] = []
        for msg in messages or []:
            label, text = self._message_label_and_text(msg)
            lines.append(f"{label}: {text}" if text else f"{label}:")
            self._append_tool_calls(lines, msg)
        return "\n".join([line for line in lines if line])

    def _message_label_and_text(self, msg: Any) -> tuple[str, str]:
        role, content, tool_name = self._message_parts(msg)
        label = f"Tool[{tool_name or 'unknown'}]" if role == "tool" else str(role or "message").capitalize()
        text = self._content_to_text(content).strip()
        return label, text

    @staticmethod
    def _message_parts(msg: Any) -> tuple[str | None, Any, str]:
        if isinstance(msg, dict):
            return msg.get("role"), msg.get("content"), ""
        if isinstance(msg, HumanMessage):
            return "user", msg.content, ""
        if isinstance(msg, SystemMessage):
            return "system", msg.content, ""
        if isinstance(msg, AIMessage):
            return "assistant", msg.content, ""
        if isinstance(msg, ToolMessage):
            return "tool", msg.content, str(getattr(msg, "name", "") or "")
        role = getattr(msg, "type", None) or "message"
        return role, getattr(msg, "content", None), ""

    @staticmethod
    def _append_tool_calls(lines: list[str], msg: Any) -> None:
        if not isinstance(msg, AIMessage):
            return
        for tc in getattr(msg, "tool_calls", None) or []:
            name, args = CommonTurnExecutor._tool_call_info(tc)
            call_line = f"Assistant tool_call: {name}"
            if args:
                call_line += f" args={args}"
            lines.append(call_line)

    @staticmethod
    def _tool_call_info(tc: Any) -> tuple[str, str]:
        if isinstance(tc, dict):
            tc_name = tc.get("name") or ""
            tc_args = tc.get("args")
        else:
            tc_name = getattr(tc, "name", "") or ""
            tc_args = getattr(tc, "args", None)
        return tc_name, CommonTurnExecutor._tool_args_text(tc_args)

    @staticmethod
    def _tool_args_text(tc_args: Any) -> str:
        if tc_args is None:
            return ""
        if isinstance(tc_args, str):
            return tc_args
        try:
            return json.dumps(tc_args, ensure_ascii=False)
        except Exception:
            return str(tc_args)

    async def _summarize_messages(self, messages: list[Any]) -> tuple[str, str]:
        transcript = self._messages_to_transcript(messages)
        if not transcript.strip():
            return "", "empty_transcript"
        if self._model is None:
            return "", "model_unavailable"
        return await self._request_summary(transcript)

    async def _request_summary(self, transcript: str) -> tuple[str, str]:
        try:
            summary_messages = self._summary_messages(transcript)
            response = await self._model.ainvoke(summary_messages)
            summary = self._content_to_text(getattr(response, "content", "")).strip()
            return (summary, "") if summary else ("", "empty_summary")
        except Exception as exc:
            logger.warning("[CommonTurnExecutor] Compaction summary failed: %s", exc)
            return "", str(exc)

    def _summary_messages(self, transcript: str) -> list[Any]:
        return [
            SystemMessage(content=self._compaction_system_prompt),
            HumanMessage(content=self._compaction_user_prompt.format(transcript=transcript)),
        ]

    @staticmethod
    def _coerce_int(value: Any) -> Optional[int]:
        try:
            return int(value) if value is not None else None
        except Exception:
            return None

    @staticmethod
    def _coerce_ratio(value: Any, default: float) -> float:
        if value is None:
            return float(default)
        try:
            ratio = float(value)
        except Exception:
            return float(default)
        if ratio > 1.0 and ratio <= 100.0:
            ratio = ratio / 100.0
        if ratio <= 0.0:
            return float(default)
        return min(1.0, float(ratio))

    def _update_last_usage(self, messages: list[Any]) -> None:
        for msg in reversed(messages or []):
            usage = self._extract_usage_from_message(msg)
            if usage:
                normalized = self._normalize_usage(usage)
                if any(value is not None for value in normalized.values()):
                    self._last_usage = normalized
                return

    def _extract_usage_from_message(self, msg: Any) -> Optional[dict]:
        usage = self._usage_from_attrs(msg)
        if usage:
            return usage
        usage = self._usage_from_dict(msg)
        return usage or None

    @staticmethod
    def _usage_from_attrs(msg: Any) -> dict | None:
        usage = getattr(msg, "usage_metadata", None)
        if isinstance(usage, dict) and usage:
            return usage
        for attr_name in ("response_metadata", "additional_kwargs"):
            attr = getattr(msg, attr_name, None)
            usage = CommonTurnExecutor._usage_from_container(attr)
            if usage:
                return usage
        return None

    @staticmethod
    def _usage_from_container(container: Any) -> dict | None:
        if not isinstance(container, dict):
            return None
        for key in ("usage", "token_usage", "usage_metadata"):
            usage = container.get(key)
            if isinstance(usage, dict) and usage:
                return usage
        return None

    @staticmethod
    def _usage_from_dict(msg: Any) -> dict | None:
        if not isinstance(msg, dict):
            return None
        return CommonTurnExecutor._usage_from_container(msg)

    @classmethod
    def _normalize_usage(cls, usage: dict) -> dict[str, Optional[int]]:
        prompt = usage.get("prompt_tokens") or usage.get("input_tokens") or usage.get("prompt")
        completion = usage.get("completion_tokens") or usage.get("output_tokens") or usage.get("completion")
        total = usage.get("total_tokens") or usage.get("total")
        prompt_val = cls._coerce_int(prompt)
        completion_val = cls._coerce_int(completion)
        total_val = cls._coerce_int(total)
        if prompt_val is None and total_val is not None and completion_val is not None:
            derived = total_val - completion_val
            prompt_val = derived if derived >= 0 else None
        return {
            "prompt_tokens": prompt_val,
            "completion_tokens": completion_val,
            "total_tokens": total_val,
        }

    async def _emit_context_usage(self, label: str) -> None:
        usage = self._last_usage or {}
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        total_tokens = usage.get("total_tokens")
        suffix = f" after {label}" if label else ""
        if prompt_tokens is None and completion_tokens is None and total_tokens is None:
            text = f"[INFO] Context tokens{suffix}: N/A (usage not available)"
        else:
            text = self._usage_text(prompt_tokens, completion_tokens, total_tokens, suffix)
        send_message(self.session_id, DisplayMessage(text=text))

    @staticmethod
    def _usage_text(prompt_tokens: Any, completion_tokens: Any, total_tokens: Any, suffix: str) -> str:
        in_text = "N/A" if prompt_tokens is None else str(prompt_tokens)
        out_text = "N/A" if completion_tokens is None else str(completion_tokens)
        total_text = "N/A" if total_tokens is None else str(total_tokens)
        return f"[INFO] Context tokens{suffix}: in={in_text}, out={out_text}, total={total_text}"

    async def compact_history(self, *, reason: str = "manual") -> dict[str, Any]:
        messages, error_reason = await self._compaction_messages()
        if not messages:
            return {"ok": False, "reason": error_reason}
        summary, error = await self._summarize_messages(messages)
        if not summary:
            return {"ok": False, "reason": "summary_failed", "error": error}
        return self._store_compaction_summary(summary, len(messages), reason)

    async def _compaction_messages(self) -> tuple[list[Any] | None, str]:
        config = {"configurable": {"thread_id": self.thread_id}}
        state = await self._agent.aget_state(config)
        if not state or not state.values:
            return None, "no_state"
        messages = state.values.get("messages", []) or []
        if not messages:
            return None, "no_messages"
        return messages, ""

    def _store_compaction_summary(self, summary: str, total: int, reason: str) -> dict[str, Any]:
        self._pending_compaction_summary = summary
        self.thread_id = f"{self.session_id}_{uuid.uuid4().hex[:8]}"
        self._awaiting_resume = False
        logger.info("[CommonTurnExecutor] %s compact: %d -> %d", reason, total, 1)
        return {"ok": True, "changed": True, "before": total, "after": 1, "summary": summary}

    async def _maybe_auto_compact(self) -> None:
        if not self._enable_auto_compact:
            return
        threshold = self._auto_compact_threshold()
        if threshold is None:
            return
        state = await self._get_agent_state()
        messages = state.get("messages", []) if state else []
        if len(messages) < threshold:
            return
        result = await self.compact_history(reason="auto")
        summary = result.get("summary") if isinstance(result, dict) else None
        if summary:
            send_message(self.session_id, DisplayMessage(text=f"[INFO] Auto-compact summary:\n{summary}"))

    def _auto_compact_threshold(self) -> Optional[int]:
        if self._max_messages <= 1:
            return None
        return max(2, int(self._max_messages * self._auto_compact_ratio))

    async def _get_agent_state(self) -> dict[str, Any] | None:
        config = {"configurable": {"thread_id": self.thread_id}}
        state = await self._agent.aget_state(config)
        return state.values if state and state.values else None

    async def _rollback_incomplete_tool_calls(self) -> None:
        try:
            messages = await self._get_messages_for_rollback()
            if not messages:
                return
            answered = self._answered_tool_calls(messages)
            to_remove = self._messages_to_remove(messages, answered)
            if not to_remove:
                return
            await self._apply_message_removals(to_remove)
        except Exception as exc:
            logger.warning("[CommonTurnExecutor] Rollback failed: %s, resetting thread_id", exc)
            self.thread_id = f"{self.session_id}_{uuid.uuid4().hex[:8]}"

    async def _get_messages_for_rollback(self) -> list[Any]:
        state = await self._get_agent_state()
        return state.get("messages", []) if state else []

    @staticmethod
    def _answered_tool_calls(messages: list[Any]) -> set[str]:
        ids: set[str] = set()
        for msg in messages:
            if isinstance(msg, ToolMessage):
                tool_call_id = getattr(msg, "tool_call_id", None)
                if tool_call_id:
                    ids.add(tool_call_id)
        return ids

    def _messages_to_remove(self, messages: list[Any], answered: set[str]) -> list[Any]:
        from langchain_core.messages import RemoveMessage

        for msg in reversed(messages):
            if not isinstance(msg, AIMessage):
                continue
            pending = self._has_pending_tool_calls(msg, answered)
            if pending:
                msg_id = getattr(msg, "id", None)
                if msg_id:
                    logger.info("[CommonTurnExecutor] Removing incomplete AIMessage: %s", msg_id)
                    return [RemoveMessage(id=msg_id)]
            break
        return []

    @staticmethod
    def _has_pending_tool_calls(msg: AIMessage, answered: set[str]) -> bool:
        for tc in getattr(msg, "tool_calls", None) or []:
            tc_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
            if tc_id and tc_id not in answered:
                return True
        return False

    async def _apply_message_removals(self, messages_to_remove: list[Any]) -> None:
        config = {"configurable": {"thread_id": self.thread_id}}
        await self._agent.aupdate_state(config, {"messages": messages_to_remove})
        logger.info("[CommonTurnExecutor] Rolled back %d incomplete message(s)", len(messages_to_remove))

    async def run_turn(self, user_input: str, use_stream: bool) -> dict:
        self.running_task = asyncio.current_task()
        try:
            processor = RunTurnProcessor(self, user_input, use_stream)
            return await processor.run()
        finally:
            self.running_task = None
