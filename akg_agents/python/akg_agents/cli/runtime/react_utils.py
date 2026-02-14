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

"""
ReAct 对话管理辅助工具

提供 checkpointer 创建、消息裁剪等功能，供 CLI executor 使用。
（从原 core/agent/react_agent.py 迁移而来）
"""

import logging
from typing import Any, Optional

from langchain.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langchain.agents import AgentState
from langchain.agents.middleware import before_model
from langgraph.runtime import Runtime

logger = logging.getLogger(__name__)

DEFAULT_MAX_MESSAGES = 100


def create_checkpointer(backend: str = "memory", db_path: Optional[str] = None) -> Any:
    """
    创建 checkpointer 用于保存对话历史
    """
    if backend == "memory":
        from langgraph.checkpoint.memory import InMemorySaver
        logger.info("Created InMemorySaver for short-term memory")
        return InMemorySaver()
    # 可在一个.db的文件中持久化
    elif backend == "sqlite":
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
            path = db_path or ":memory:"
            logger.info(f"Created SqliteSaver at: {path}")
            return SqliteSaver.from_conn_string(path)
        except ImportError:
            logger.warning("langgraph-checkpoint-sqlite not installed, falling back to InMemorySaver")
            from langgraph.checkpoint.memory import InMemorySaver
            return InMemorySaver()
    else:
        logger.warning(f"Unknown checkpointer backend '{backend}', using InMemorySaver")
        from langgraph.checkpoint.memory import InMemorySaver
        return InMemorySaver()


def _find_safe_trim_index(messages: list, target_keep: int) -> int:
    """
    为了保证截取之后的message是能够配对的，因为aimessage(有tool_Call)后面必须紧跟对应的toolmessage
    """
    from langchain_core.messages import AIMessage, ToolMessage

    if len(messages) <= target_keep:
        return 0

    target_start = len(messages) - target_keep
    safe_start = target_start

    if safe_start > 0 and safe_start < len(messages):
        start_msg = messages[safe_start]

        if isinstance(start_msg, ToolMessage):
            for i in range(safe_start - 1, -1, -1):
                msg = messages[i]
                if isinstance(msg, AIMessage):
                    tool_calls = getattr(msg, 'tool_calls', None)
                    if tool_calls:
                        safe_start = i
                        break
                elif not isinstance(msg, ToolMessage):
                    break
    if safe_start > 0:
        prev_msg = messages[safe_start - 1]
        if isinstance(prev_msg, AIMessage):
            tool_calls = getattr(prev_msg, 'tool_calls', None)
            if tool_calls:
                if safe_start < len(messages) and isinstance(messages[safe_start], ToolMessage):
                    safe_start -= 1

    logger.debug(f"Safe trim: target_start={target_start}, safe_start={safe_start}")
    return safe_start


@before_model
def trim_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """
    保存对话历史
    """
    messages = state["messages"]
    max_messages = DEFAULT_MAX_MESSAGES

    if len(messages) <= max_messages:
        return None

    first_msg = messages[0]
    remaining_messages = messages[1:]

    target_keep = max_messages - 1

    safe_start = _find_safe_trim_index(remaining_messages, target_keep)
    recent_messages = remaining_messages[safe_start:]

    logger.info(f"Trimming messages: {len(messages)} -> {1 + len(recent_messages)} (safe_start={safe_start})")

    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            first_msg,
            *recent_messages
        ]
    }
