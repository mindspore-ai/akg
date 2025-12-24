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

"""CliClient - 流式客户端，负责与远程 server 通信"""

import logging
import json
import os
from typing import Dict, Any, Optional
import asyncio
import httpx
import websockets
import uuid
from pathlib import Path

from textual import log

from ..messages import (
    unpack_message,
    NodeStartMessage,
    NodeEndMessage,
    LLMStartMessage,
    LLMEndMessage,
    LLMStreamMessage,
    FinalResultMessage,
    ErrorMessage,
    JobSubmittedMessage,
    ProgressMessage,
)


logger = logging.getLogger(__name__)


class CliClient:
    """远程客户端（使用 WebSocket）"""

    def __init__(
        self,
        server_url: str = "http://localhost:9002",
        timeout: float = 600.0,
        presenter=None,
        *,
        session_id: str | None = None,
        record_path: str | None = None,
        console=None,
    ):
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        self.session_id = str(session_id or uuid.uuid4())
        self.presenter = presenter  # 直接持有 presenter
        self.console = console
        self.current_job_id: Optional[str] = None
        self._status_checked = False

        # 兼容原 AIKGCLI 会话字段（供 runners/orchestrator 使用）
        self.auto_yes: bool = False
        self.default_user_input: str = ""
        self.use_stream: bool = False
        self.workflow_name: str = ""
        self.notifier = None

        self._recorder = None
        self._recorder_disabled = False
        if record_path:
            try:
                from ai_kernel_generator.cli.cli.utils.message_recording import (
                    MessageRecorder,
                )

                self._recorder = MessageRecorder(Path(record_path))
            except Exception as e:
                log.warning(
                    "[Client] init MessageRecorder failed; disabled", exc_info=e
                )
                self._recorder = None

    def set_record_path(self, record_path: str | None) -> None:
        """为当前 session 启用/更新 server->cli 消息录制（jsonl）。"""
        if not record_path:
            return
        try:
            from ai_kernel_generator.cli.cli.utils.message_recording import (
                MessageRecorder,
            )

            self._recorder = MessageRecorder(Path(record_path))
            self._recorder_disabled = False
        except Exception as e:
            log.warning("[Client] set_record_path failed; disabled", exc_info=e)
            self._recorder = None
            self._recorder_disabled = True

    def set_presenter(self, presenter):
        """设置 presenter"""
        self.presenter = presenter

    async def check_status(self) -> Dict[str, Any]:
        """检查服务器状态"""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.server_url}/api/v1/workflow/status")
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"Failed to check server status: {e}")
            raise ConnectionError(f"无法连接到工作流服务器: {self.server_url}") from e

    @classmethod
    def create_for_cli(
        cls,
        console,
        *,
        auto_yes: bool = False,
        server_url: Optional[str] = None,
        timeout: float = 600.0,
        use_stream: bool = False,
        default_user_input: Optional[str] = None,
        notify: bool = True,
        bark_key: str = "",
        record_path: str | None = None,
        session_id: str | None = None,
    ) -> "CliClient":
        from ai_kernel_generator.cli.cli.constants import Defaults
        from ai_kernel_generator.cli.cli.presenter import CLIPresenter
        from ai_kernel_generator.cli.cli.service.notify import BarkNotifier

        server_url0 = (server_url or Defaults.SERVER_URL).rstrip("/")

        presenter = CLIPresenter(
            console,
            use_stream=use_stream,
        )

        # 会话 id：用于 TUI resume + client 录制目录
        sid = str(session_id or uuid.uuid4())
        os.environ["AIKG_SESSION_ID"] = sid

        # 设置 Stream 环境变量
        os.environ["AIKG_STREAM_OUTPUT"] = "on" if use_stream else "off"

        client = cls(
            server_url=server_url0,
            timeout=timeout,
            presenter=presenter,
            session_id=sid,
            record_path=record_path,
            console=console,
        )
        client.auto_yes = bool(auto_yes)
        client.default_user_input = (
            (default_user_input or Defaults.DEFAULT_USER_INPUT)
            if default_user_input is not None
            else Defaults.DEFAULT_USER_INPUT
        )
        client.use_stream = bool(use_stream)
        client.workflow_name = Defaults.WORKFLOW_NAME
        client.notifier = BarkNotifier(console, bark_key=bark_key, enabled=notify)
        return client

    async def execute_main_agent(
        self,
        *,
        action: str,
        user_input: str = "",
        framework: str = "torch",
        backend: str = "cuda",
        arch: str = "a100",
        dsl: str = "triton_cuda",
        use_stream: bool | None = None,
        rag: bool = False,
    ) -> Dict[str, Any]:
        """执行 MainOpAgent 对话（start/continue），返回 server 侧状态。"""
        from ai_kernel_generator.cli.cli.constants import DisplayStyle

        action_l = (action or "").strip().lower()
        if action_l not in ["start", "continue"]:
            raise ValueError("action must be start/continue")
        if not (user_input or "").strip():
            raise ValueError(f"action={action_l} requires non-empty user_input")

        effective_use_stream = (
            bool(use_stream) if use_stream is not None else bool(self.use_stream)
        )

        try:
            if (self.console is not None) and (not self._status_checked):
                self.console.print(
                    f"[{DisplayStyle.DIM}]正在连接服务器: {self.server_url}...[/{DisplayStyle.DIM}]"
                )
                status = await self.check_status()
                self.console.print(
                    f"[{DisplayStyle.GREEN}]服务器状态: {status.get('status')}[/{DisplayStyle.GREEN}]\n"
                )
                self._status_checked = True
        except Exception as exc:
            log.warning(
                "[Client] connect server failed",
                server_url=str(self.server_url or ""),
                exc_info=exc,
            )
            raise ConnectionError(f"无法连接到服务器 {self.server_url}: {exc}")

        request_data = {
            "session_id": self.session_id,
            "action": action_l,
            "user_input": user_input,
            "framework": framework,
            "backend": backend,
            "arch": arch,
            "dsl": dsl,
            "use_stream": effective_use_stream,
            "rag": rag,
        }

        ws_url = self._make_ws_url("/api/v1/main_agent/stream")
        logger.info(
            "[Client] main_agent action=%s session_id=%s input_len=%s",
            str(action or ""),
            str(self.session_id),
            len(user_input) if isinstance(user_input, str) else 0,
        )
        result = await self._execute_ws(ws_url, request_data)
        if not isinstance(result, dict):
            raise RuntimeError("invalid main agent response")

        # 用 server 返回的 workflow_name 更新本地
        try:
            wf_name = (
                str(result.get("workflow_name") or result.get("sub_workflow") or "")
                .strip()
            )
            if wf_name:
                self.workflow_name = wf_name
                try:
                    if self.presenter and hasattr(self.presenter, "set_task_context"):
                        self.presenter.set_task_context(
                            framework=framework,
                            backend=backend,
                            arch=arch,
                            dsl=dsl,
                            workflow_name=wf_name,
                        )
                except Exception as e:
                    log.debug(
                        "[Client] presenter.set_task_context failed after workflow_name update",
                        exc_info=e,
                    )
        except Exception as e:
            log.debug("[Client] update workflow_name from metadata failed", exc_info=e)
        return result

    def _make_ws_url(self, path: str) -> str:
        # 将 http:// 替换为 ws://
        ws_url = self.server_url.replace("http://", "ws://").replace(
            "https://", "wss://"
        )
        return f"{ws_url.rstrip('/')}{path}"

    async def _execute_ws(
        self, ws_url: str, request_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """底层 WebSocket 执行器：发送请求并等待 FinalResult。"""

        try:
            logger.info(f"Connecting to WebSocket: {ws_url}")

            # 调试 server 时事件循环可能被阻塞，禁用客户端 keepalive ping 避免超时断连
            async with websockets.connect(ws_url, ping_interval=None) as websocket:
                # 发送请求
                await websocket.send(json.dumps(request_data))

                # 接收消息
                final_result = None
                async for message in websocket:
                    data = json.loads(message)
                    if (
                        self._recorder is not None
                        and not self._recorder_disabled
                        and isinstance(data, dict)
                    ):
                        try:
                            self._recorder.record(data)
                        except Exception as e:
                            self._recorder_disabled = True
                            log.warning(
                                "[Client] recorder failed; disabled for this session",
                                exc_info=e,
                            )

                    # 统一解包消息
                    message = unpack_message(data)
                    if not message:
                        logger.warning(f"无法解包消息: {data}")
                        continue

                    # 记录 job_id（用于取消/跟踪）
                    if isinstance(message, JobSubmittedMessage):
                        self.current_job_id = message.job_id
                        logger.info(
                            f"[Client] session_id={self.session_id} - job_id={self.current_job_id}"
                        )
                        try:
                            if self.presenter and hasattr(
                                self.presenter, "on_job_submitted"
                            ):
                                self.presenter.on_job_submitted(self.current_job_id)
                        except Exception as e:
                            log.warning(
                                "[Client] presenter.on_job_submitted failed",
                                job_id=str(self.current_job_id or ""),
                                exc_info=e,
                            )
                        continue

                    # 处理最终结果
                    if isinstance(message, FinalResultMessage):
                        final_result = message.result
                        break

                    # 处理错误
                    if isinstance(message, ErrorMessage):
                        raise RuntimeError(f"服务器错误: {message.error}")

                    # 路由到 presenter
                    self._route_to_presenter(message)

                if final_result is None:
                    raise RuntimeError("未收到最终结果")

                logger.info(
                    "Main agent response received: step=%s",
                    str(final_result.get("current_step", "")),
                )
                return final_result

        except asyncio.CancelledError:
            # asyncio.run() 在 Ctrl+C 时可能会取消当前协程
            await self.cancel_current_job(reason="client cancelled (asyncio)")
            raise
        except KeyboardInterrupt:
            await self.cancel_current_job(reason="client cancelled (keyboardinterrupt)")
            raise
        except websockets.exceptions.WebSocketException as e:
            logger.error(f"WebSocket error: {e}")
            raise ConnectionError(f"WebSocket 连接失败: {str(e)}") from e
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            raise RuntimeError(f"工作流执行失败: {str(e)}") from e

    async def cancel_current_job(self, reason: str = "cancelled by client") -> bool:
        """尽力取消当前 job（需要 server 支持 /api/v1/jobs/{job_id}/cancel）"""
        if not self.current_job_id:
            return False
        try:
            url = f"{self.server_url}/api/v1/jobs/{self.current_job_id}/cancel"
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.post(url, params={"reason": reason})
                resp.raise_for_status()
            return True
        except Exception as e:
            logger.warning(f"Cancel job failed: job_id={self.current_job_id}, err={e}")
            return False

    def _route_to_presenter(self, message):
        """
        直接路由消息到 presenter

        Args:
            message: Message 对象
        """
        if hasattr(message, "task_id"):
            # 只有当 AIKG_ONLY_USE_MAIN_TAB=1 时才重置 task_id，默认为 0
            if os.getenv("AIKG_ONLY_USE_MAIN_TAB", "0") == "1":
                message.task_id = ""
        if not self.presenter:
            logger.warning(
                f"[Client] 没有设置 presenter，无法路由消息: type={type(message).__name__}"
            )
            return
        msg_type = type(message).__name__
        msg_info = self._format_message_info(message)

        logger.debug(
            f"[Client] session_id={self.session_id} - 收到消息: type={msg_type}, {msg_info}"
        )

        # 直接根据消息类型路由
        try:
            if isinstance(message, NodeStartMessage):
                self.presenter.on_node_start(message)
            elif isinstance(message, NodeEndMessage):
                self.presenter.on_node_end(message)
            elif isinstance(message, LLMStartMessage):
                self.presenter.on_llm_start(message)
            elif isinstance(message, LLMEndMessage):
                self.presenter.on_llm_end(message)
            elif isinstance(message, LLMStreamMessage):
                self.presenter.on_llm_stream(message)
            elif isinstance(message, ProgressMessage):
                data = getattr(message, "data", {}) or {}
                logger.info(
                    "[Client] progress scope=%s round=%s done=%s total=%s ok=%s fail=%s",
                    str(getattr(message, "scope", "") or ""),
                    str(getattr(data, "get", lambda *_: "")("round", "")),
                    str(getattr(data, "get", lambda *_: "")("done", "")),
                    str(getattr(data, "get", lambda *_: "")("total", "")),
                    str(getattr(data, "get", lambda *_: "")("ok", "")),
                    str(getattr(data, "get", lambda *_: "")("fail", "")),
                )
                if hasattr(self.presenter, "on_progress"):
                    self.presenter.on_progress(message)
            else:
                logger.warning(f"[Client] 未知消息类型: {msg_type}")
                return

            logger.debug(
                f"[Client] session_id={self.session_id} - 消息路由成功: type={msg_type}"
            )

        except Exception as e:
            logger.error(
                f"[Client] session_id={self.session_id} - 消息路由失败: type={msg_type}, error={e}",
                exc_info=True,
            )

    def _format_message_info(self, message):
        """
        格式化消息信息用于日志输出

        Args:
            message: Message 对象

        Returns:
            格式化的消息信息字符串
        """
        if isinstance(message, NodeStartMessage):
            return f"node={message.node}"
        elif isinstance(message, NodeEndMessage):
            return f"node={message.node}, duration={message.duration:.2f}s"
        elif isinstance(message, LLMStartMessage):
            return f"agent={message.agent}, model={message.model}"
        elif isinstance(message, LLMEndMessage):
            token_parts = []
            if getattr(message, "prompt_tokens", None) is not None:
                token_parts.append(f"in={message.prompt_tokens}")
            if getattr(message, "reasoning_tokens", None) is not None:
                token_parts.append(f"reasoning={message.reasoning_tokens}")
            if getattr(message, "output_tokens", None) is not None:
                token_parts.append(f"output={message.output_tokens}")
            if getattr(message, "total_tokens", None) is not None:
                token_parts.append(f"total={message.total_tokens}")
            token_text = f", tokens({'; '.join(token_parts)})" if token_parts else ""
            return (
                f"agent={message.agent}, duration={message.duration:.2f}s{token_text}"
            )
        elif isinstance(message, LLMStreamMessage):
            return f"agent={message.agent}, chunk_len={len(message.chunk)}"
        else:
            return ""
