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

"""全局消息发送器 - 简化版，替代 AgentHookManager。

仅负责按 session_id 路由并发送 Message，不对 task_id 做任何推断或过滤。
"""

import logging
import threading
from typing import Callable, Dict
from ..messages import Message
from textual import log

logger = logging.getLogger(__name__)


# Session 级别的消息发送器（线程安全）
_session_senders: Dict[str, Callable[[Message], None]] = {}
_session_lock = threading.Lock()


def register_message_sender(session_id: str, sender: Callable[[Message], None]):
    """
    注册消息发送器（由 executor 初始化时调用）

    Args:
        session_id: 会话 ID
        sender: 消息发送函数，接收 Message 对象
    """
    with _session_lock:
        _session_senders[session_id] = sender
        logger.info(f"[MessageSender] 注册 session_id={session_id} 的消息发送器")


def unregister_message_sender(session_id: str):
    """
    注销消息发送器

    Args:
        session_id: 会话 ID
    """
    with _session_lock:
        if session_id in _session_senders:
            del _session_senders[session_id]
            logger.info(f"[MessageSender] 注销 session_id={session_id} 的消息发送器")


def send_message(session_id: str, message: Message):
    """
    发送消息到客户端

    Args:
        session_id: 会话 ID
        message: Message 对象
    """
    # 统一检查 session_id
    if not session_id:
        logger.warning("[MessageSender] session_id 为空，无法发送消息")
        return

    # 获取消息类型和基本信息（不应因 get_type 异常而影响主流程）
    get_type = getattr(message, "get_type", None)
    if callable(get_type):
        try:
            msg_type = get_type()
        except Exception as e:
            log.warning(
                "[MessageSender] message.get_type failed; fallback to class name",
                exc_info=e,
            )
            msg_type = message.__class__.__name__
    else:
        msg_type = message.__class__.__name__

    # llm_stream消息非常频繁：过滤掉该类 debug，避免刷屏
    with _session_lock:
        sender = _session_senders.get(session_id)

    if sender:
        if msg_type not in ("llm_stream",) and logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[MessageSender] session_id=%s - 准备发送消息: type=%s, %s",
                session_id,
                msg_type,
                _format_message_info(message),
            )
        try:
            sender(message)
            if msg_type not in ("llm_stream",) and logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "[MessageSender] session_id=%s - 消息发送成功: type=%s",
                    session_id,
                    msg_type,
                )
        except Exception as e:
            log.warning(
                "[MessageSender] send failed",
                session_id=str(session_id or ""),
                msg_type=str(msg_type or ""),
                exc_info=e,
            )
            logger.warning(
                "[MessageSender] session_id=%s - 发送消息失败: type=%s, error=%s",
                session_id,
                msg_type,
                e,
            )
    else:
        if msg_type not in ("llm_stream",):
            logger.warning(
                "[MessageSender] session_id=%s - 没有注册消息发送器，无法发送: type=%s",
                session_id,
                msg_type,
            )


def _format_message_info(message: Message) -> str:
    """
    格式化消息信息用于日志输出

    Args:
        message: Message 对象

    Returns:
        格式化的消息信息字符串
    """
    from ..messages import (
        NodeStartMessage,
        NodeEndMessage,
        LLMStartMessage,
        LLMEndMessage,
        LLMStreamMessage,
    )

    tid = getattr(message, "task_id", "") if hasattr(message, "task_id") else ""
    tid_txt = f", task_id={tid}" if tid else ""

    if isinstance(message, (NodeStartMessage, NodeEndMessage)):
        node = getattr(message, "node", "unknown")
        if isinstance(message, NodeEndMessage):
            duration = getattr(message, "duration", 0)
            return f"node={node}, duration={duration:.2f}s{tid_txt}"
        return f"node={node}{tid_txt}"
    elif isinstance(message, (LLMStartMessage, LLMEndMessage)):
        agent = getattr(message, "agent", "unknown")
        if isinstance(message, LLMEndMessage):
            duration = getattr(message, "duration", 0)
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
            return f"agent={agent}, duration={duration:.2f}s{tid_txt}{token_text}"
        # LLMStartMessage 不再携带 prompt_tokens（以 llm_end 的 LLM API 统计为准）
        return f"agent={agent}{tid_txt}"
    elif isinstance(message, LLMStreamMessage):
        agent = getattr(message, "agent", "unknown")
        chunk_len = len(getattr(message, "chunk", ""))
        return f"agent={agent}, chunk_len={chunk_len}{tid_txt}"
    else:
        return f"message_class={message.__class__.__name__}"


def clear_all_senders():
    """清除所有消息发送器"""
    with _session_lock:
        _session_senders.clear()
        logger.info("[MessageSender] 清除所有消息发送器")
