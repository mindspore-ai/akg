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

import os

import asyncio
import logging
from typing import Dict

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ai_kernel_generator.cli.messages import (
    ErrorMessage,
    FinalResultMessage,
    pack_message,
)
from ai_kernel_generator.cli.server.config import get_server_config
from ai_kernel_generator.cli.server.message_sender import (
    register_message_sender,
    send_message,
    unregister_message_sender,
)
from ai_kernel_generator.cli.server.models import (
    CliMainAgentRequest,
    ServerStatusResponse,
)
from ai_kernel_generator.config.config_validator import load_config
from ai_kernel_generator.utils.main_op_agent_display import (
    format_display_message,
    get_hint_message,
    is_simple_command,
)

logger = logging.getLogger(__name__)

router = APIRouter()

_main_agent_states: dict[str, dict] = {}


def _load_workflow_config(backend: str, dsl: str) -> Dict:
    """加载工作流配置"""
    # 设置环境变量
    config = load_config(dsl, backend=backend)

    # 兜底补齐 agent_model_config，避免某些 agent 使用 `.get(x, "default")` 时触发 preset=default 报错
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


def _sanitize_state_for_client(state: dict) -> dict:
    payload = dict(state) if isinstance(state, dict) else {}
    payload.pop("config", None)
    payload.pop("conversation_history", None)
    return payload


def _ensure_display_messages(state: dict) -> None:
    if not state.get("display_message"):
        state["display_message"] = format_display_message(state)
    if not state.get("hint_message"):
        state["hint_message"] = get_hint_message(state)


def _build_response_state(state: dict) -> dict:
    _ensure_display_messages(state)
    payload = _sanitize_state_for_client(state)
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


def _resolve_log_dir(state: dict) -> str:
    config = state.get("config") if isinstance(state, dict) else None
    if isinstance(config, dict):
        log_dir = config.get("log_dir")
    else:
        log_dir = None
    if not log_dir:
        log_dir = "~/aikg_logs"
    return os.path.expanduser(str(log_dir))


def _build_command_response(
    state: dict,
    *,
    command: str,
    message: str,
    save_path: str | None = None,
) -> dict:
    payload = _sanitize_state_for_client(state)
    payload["current_step"] = "saved" if command == "save" else "cancelled"
    payload["should_continue"] = False
    payload["display_message"] = message
    payload["hint_message"] = ""
    if save_path:
        payload["saved_path"] = save_path
    return payload


@router.get("/api/v1/workflow/status", response_model=ServerStatusResponse)
async def get_workflow_status():
    try:
        backend, arch, devices = get_server_config()
        return ServerStatusResponse(
            status="ready",
            version="1.0.0",
            backend=backend,
            arch=arch,
            devices=devices,
        )
    except Exception as e:
        logger.error(f"Failed to get workflow status: {e}", exc_info=True)
        return ServerStatusResponse(
            status="error",
            version="1.0.0",
            backend="unknown",
            arch="unknown",
            devices=[],
        )


async def _safe_send_json(websocket: WebSocket, payload: dict) -> None:
    try:
        await websocket.send_json(payload)
    except (WebSocketDisconnect, RuntimeError) as e:
        logger.debug(f"WebSocket send skipped: {e}", exc_info=True)


def _create_main_op_agent(request: CliMainAgentRequest, session_id: str):
    """创建 MainOpAgent 实例。"""
    from ai_kernel_generator.core.agent.main_op_agent import MainOpAgent

    config = _load_workflow_config(  # type: ignore[attr-defined]
        request.backend, request.dsl
    )
    config["session_id"] = session_id
    config["task_label"] = "main"

    return MainOpAgent(
        config=config,
        framework=request.framework,
        backend=request.backend,
        arch=request.arch,
        dsl=request.dsl,
    )


async def _handle_start_action(
    request: CliMainAgentRequest, session_id: str
) -> dict:
    """start action: 开启新对话"""
    user_input = request.user_input or ""
    if not user_input.strip():
        raise ValueError("user_input is required for start action")

    _main_agent_states.pop(session_id, None)
    agent = _create_main_op_agent(request, session_id)
    state = await agent.start_conversation(
        user_request=user_input, task_id=session_id
    )
    _main_agent_states[session_id] = state
    return _build_response_state(state)


async def _handle_continue_action(
    request: CliMainAgentRequest, session_id: str
) -> dict:
    """continue action: 继续对话（所有逻辑由 server 统一处理）"""
    prev = _main_agent_states.get(session_id)
    if not isinstance(prev, dict):
        raise ValueError("no main agent state found for this session; call start first")

    user_input = request.user_input or ""
    if not user_input.strip():
        raise ValueError("user_input is required for continue action")

    is_cmd, command_type = is_simple_command(user_input)
    if is_cmd and command_type == "exit":
        _main_agent_states.pop(session_id, None)
        return _build_command_response(
            prev, command="exit", message="Conversation ended."
        )
    if is_cmd and command_type == "save":
        agent = _create_main_op_agent(request, session_id)
        log_dir = _resolve_log_dir(prev)
        os.makedirs(log_dir, exist_ok=True)
        task_id = str(prev.get("task_id") or session_id)
        save_path = os.path.join(log_dir, f"conversation_{task_id}.json")
        agent.save_conversation(prev, save_path)
        _main_agent_states.pop(session_id, None)
        return _build_command_response(
            prev,
            command="save",
            message=f"Conversation saved to: {save_path}",
            save_path=save_path,
        )

    agent = _create_main_op_agent(request, session_id)
    if "config" not in prev:
        prev["config"] = agent.config
    state = await agent.continue_conversation(
        current_state=prev,
        user_input=user_input,
        action="auto",
    )
    _main_agent_states[session_id] = state
    return _build_response_state(state)


@router.websocket("/api/v1/main_agent/stream")
async def main_agent_stream(websocket: WebSocket):
    """MainOpAgent 多轮对话接口（用于 CLI 侧与 server 端 MainAgent 交互）。"""
    await websocket.accept()

    session_id: str | None = None

    try:
        data = await websocket.receive_json()
        request = CliMainAgentRequest(**data)

        session_id = request.session_id
        if not session_id:
            raise ValueError("session_id is required")

        register_message_sender(
            session_id,
            lambda message: asyncio.create_task(
                _safe_send_json(websocket, pack_message(message))
            ),
        )

        # 与 JobManager 保持一致：用环境变量控制 stream
        os.environ["AIKG_STREAM_OUTPUT"] = "on" if request.use_stream else "off"

        action = (request.action or "").strip().lower()
        allowed_actions = {"start", "continue"}
        if action not in allowed_actions:
            raise ValueError("action must be start/continue")

        if action == "start":
            result = await _handle_start_action(request, session_id)
        else:
            result = await _handle_continue_action(request, session_id)

        await _safe_send_json(
            websocket, pack_message(FinalResultMessage(result=result))
        )

    except WebSocketDisconnect:
        if session_id:
            logger.info(
                f"[WebSocket] session_id={session_id} - 客户端断开连接(MainAgent)"
            )
    except Exception as e:
        logger.error(
            f"[WebSocket] session_id={session_id if session_id else 'unknown'} - MainAgent 错误: {e}",
            exc_info=True,
        )
        if session_id:
            send_message(session_id, ErrorMessage(error=str(e)))

        error_result = {
            "error": str(e),
            "current_step": "error",
            "should_continue": False,
            "display_message": "",
            "hint_message": "",
        }
        await _safe_send_json(
            websocket, pack_message(FinalResultMessage(result=error_result))
        )
    finally:
        if session_id:
            unregister_message_sender(session_id)
        try:
            await websocket.close()
        except RuntimeError as e:
            logger.debug(f"WebSocket close skipped: {e}", exc_info=True)
