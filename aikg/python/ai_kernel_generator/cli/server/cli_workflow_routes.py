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
from datetime import datetime
from typing import Dict

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

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
    CliExecuteResponse,
    CliMainAgentRequest,
    ServerStatusResponse,
)
from ai_kernel_generator.config.config_validator import load_config

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


def _build_main_agent_task_init_payload(state: dict) -> dict:
    """构造与 TaskInit 元数据对齐的 payload（供 CLI 复用现有展示逻辑）。"""
    status = str(state.get("task_init_status") or "").strip()
    return {
        "status": status,
        "op_name": state.get("op_name") or "",
        "generated_task_desc": state.get("task_code") or "",
        "agent_message": state.get("op_description") or "",
        "clarification_question": state.get("clarification_question") or "",
        "modification_suggestion": state.get("modification_suggestion") or "",
        "agent_reasoning": state.get("task_reasoning") or "",
        "static_check_passed": bool(state.get("static_check_passed", False)),
        "static_check_error": state.get("static_check_error") or "",
    }


def _workflow_name_from_sub_workflow(name: str) -> str:
    n = (name or "").strip().lower()
    if n == "taskopcreate":
        return "default_workflow"
    if n == "evolve":
        return "evolve_workflow"
    return "coder_only_workflow"


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


async def _handle_cancel_action(session_id: str, websocket: WebSocket) -> None:
    """处理 cancel action：清理状态并返回取消响应。"""
    _main_agent_states.pop(session_id, None)
    result = CliExecuteResponse(
        success=False,
        op_name="",
        task_init_status="cancelled",
        task_desc="",
        kernel_code="",
        verification_result=False,
        step_timings=[
            {
                "stage": "MainAgent",
                "duration": 0.0,
                "timestamp": datetime.now().isoformat(),
            }
        ],
        total_time=0.0,
        error="cancelled by client",
        metadata={},
    )
    await _safe_send_json(
        websocket, pack_message(FinalResultMessage(result=result.model_dump()))
    )


def _create_main_op_agent(request: CliMainAgentRequest, session_id: str):
    """创建 MainOpAgent 实例。"""
    from ai_kernel_generator.core.agent.main_op_agent import MainOpAgent

    config = _load_workflow_config(  # type: ignore[attr-defined]
        request.backend, request.dsl
    )
    config["session_id"] = session_id

    return MainOpAgent(
        config=config,
        framework=request.framework,
        backend=request.backend,
        arch=request.arch,
        dsl=request.dsl,
    )


def _build_confirm_response(
    state: dict, state2: dict, task_init_payload: dict, dt: float
) -> CliExecuteResponse:
    """构建 confirm action 的响应。"""
    sub_workflow = (
        str(state2.get("sub_workflow") or state.get("sub_workflow") or "")
    ).strip() or "codeonly"
    workflow_name = _workflow_name_from_sub_workflow(sub_workflow)

    return CliExecuteResponse(
        success=bool(state2.get("generation_success", False)),
        op_name=str(state2.get("op_name") or task_init_payload.get("op_name") or ""),
        task_init_status=str(task_init_payload.get("status") or ""),
        task_desc=str(task_init_payload.get("generated_task_desc") or ""),
        kernel_code=str(state2.get("generated_code") or ""),
        verification_result=bool(state2.get("verification_result", False)),
        step_timings=[
            {
                "stage": "MainAgent.Confirm",
                "duration": round(dt, 2),
                "timestamp": datetime.now().isoformat(),
            }
        ],
        total_time=round(dt, 2),
        error=(
            str(
                state2.get("verification_error") or state2.get("generation_error") or ""
            )
        )
        or None,
        metadata={
            "task_init": task_init_payload,
            "main_agent": {
                "action": "confirm",
                "current_step": state2.get("current_step", ""),
                "available_workflows": state2.get("available_workflows", []),
                "sub_workflow": sub_workflow,
                "workflow_name": workflow_name,
            },
        },
    )


def _build_task_init_response(
    state: dict, action: str, task_init_payload: dict, dt: float
) -> CliExecuteResponse:
    """构建 start/revise action 的响应。"""
    status = task_init_payload.get("status", "")
    generated = task_init_payload.get("generated_task_desc", "") or ""
    ok = bool(status == "ready" and isinstance(generated, str) and generated.strip())

    step_timings = [
        {
            "stage": "MainAgent.TaskInit",
            "duration": round(dt, 2),
            "timestamp": datetime.now().isoformat(),
        }
    ]

    return CliExecuteResponse(
        success=ok,
        op_name=task_init_payload.get("op_name") or "",
        task_init_status=status or "",
        task_desc=generated or "",
        kernel_code="",
        verification_result=False,
        step_timings=step_timings,
        total_time=round(dt, 2),
        error=None,
        metadata={
            "task_init": task_init_payload,
            "main_agent": {
                "action": action,
                "current_step": state.get("current_step", ""),
                "available_workflows": state.get("available_workflows", []),
                "sub_workflow": state.get("sub_workflow", ""),
            },
        },
    )


async def _handle_confirm_action(
    request: CliMainAgentRequest, session_id: str, websocket: WebSocket
) -> None:
    """处理 confirm action：执行选中的 sub-agent 并返回结果。"""
    state = _main_agent_states.get(session_id)
    if not isinstance(state, dict):
        raise ValueError("no main agent state found for this session; call start first")

    agent = _create_main_op_agent(request, session_id)

    t0 = asyncio.get_event_loop().time()
    state2 = await agent.continue_conversation(
        current_state=state,
        user_input="",
        action="confirm",
    )
    dt = asyncio.get_event_loop().time() - t0

    if isinstance(state2, dict):
        state2.pop("config", None)
    _main_agent_states[session_id] = state2

    # task_init 元数据沿用 start/revise 生成的（便于 CLI 展示对齐）
    task_init_payload = _build_main_agent_task_init_payload(state)

    result = _build_confirm_response(state, state2, task_init_payload, dt)

    await _safe_send_json(
        websocket, pack_message(FinalResultMessage(result=result.model_dump()))
    )


async def _handle_start_revise_action(
    request: CliMainAgentRequest, session_id: str, action: str, websocket: WebSocket
) -> None:
    """处理 start/revise action：生成或修改 task_desc。"""
    agent = _create_main_op_agent(request, session_id)

    t0 = asyncio.get_event_loop().time()
    if action == "start":
        state = await agent.start_conversation(
            user_request=request.user_input, task_id=session_id
        )
    else:
        prev = _main_agent_states.get(session_id)
        if not isinstance(prev, dict):
            raise ValueError(
                "no main agent state found for this session; call start first"
            )
        state = await agent.continue_conversation(
            current_state=prev,
            user_input=request.user_input,
            action="revise",
        )
    dt = asyncio.get_event_loop().time() - t0

    if isinstance(state, dict):
        state.pop("config", None)
    _main_agent_states[session_id] = state

    task_init_payload = _build_main_agent_task_init_payload(state)
    result = _build_task_init_response(state, action, task_init_payload, dt)

    await _safe_send_json(
        websocket, pack_message(FinalResultMessage(result=result.model_dump()))
    )


@router.websocket("/api/v1/main_agent/stream")
async def main_agent_stream(websocket: WebSocket):
    """MainOpAgent 多轮对话接口（用于 CLI 侧与 server 端 MainAgent 交互）。"""
    await websocket.accept()

    session_id: str | None = None
    job_id: str | None = None

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
        import os

        os.environ["AIKG_STREAM_OUTPUT"] = "on" if request.use_stream else "off"

        action = (request.action or "").strip().lower()
        if action not in {"start", "revise", "confirm", "cancel"}:
            raise ValueError("action must be one of: start/revise/confirm/cancel")

        # 根据 action 调用相应的处理函数
        if action == "cancel":
            await _handle_cancel_action(session_id, websocket)
        elif action == "confirm":
            await _handle_confirm_action(request, session_id, websocket)
        else:  # start or revise
            await _handle_start_revise_action(request, session_id, action, websocket)

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

        error_result = CliExecuteResponse(
            success=False,
            op_name="",
            task_init_status="error",
            task_desc="",
            kernel_code="",
            verification_result=False,
            step_timings=[],
            total_time=0,
            error=str(e),
            metadata={},
        )
        await _safe_send_json(
            websocket,
            pack_message(FinalResultMessage(result=error_result.model_dump())),
        )
    finally:
        if session_id:
            unregister_message_sender(session_id)
        try:
            await websocket.close()
        except RuntimeError as e:
            logger.debug(f"WebSocket close skipped: {e}", exc_info=True)
