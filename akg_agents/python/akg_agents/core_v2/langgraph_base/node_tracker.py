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

"""通用 LangGraph 节点追踪装饰器

提供节点执行追踪功能，支持：
- 执行时间记录
- 可选的流式消息发送
- 灵活的配置选项
"""

import functools
import logging
import time
from typing import Any, Callable, Dict

logger = logging.getLogger(__name__)


def track_node(node_name: str, require_session: bool = False, require_task_label: bool = False):
    """节点追踪装饰器（通用版本）
    
    为 LangGraph 节点函数添加追踪功能：
    - 记录执行时间
    - 可选的 session_id 校验
    - 可选的流式消息发送
    
    Args:
        node_name: 节点名称，用于日志和消息
        require_session: 是否要求 session_id（默认 False）
        require_task_label: 是否要求 task_label（默认 False）
    
    Example:
        @track_node("my_node")
        async def my_node(state: MyState) -> dict:
            # 节点逻辑
            return {"result": "done"}
    """
    def decorator(node_fn: Callable):
        @functools.wraps(node_fn)
        async def wrapped(state: Dict[str, Any]):
            session_id = str(state.get("session_id") or "").strip()
            task_label = str(state.get("task_label") or "").strip()
            
            # 可选的校验
            if require_session and not session_id:
                raise ValueError(f"[{node_name}] state 中必须包含 session_id")
            
            if require_task_label and not task_label:
                raise ValueError(f"[{node_name}] state 中必须包含 task_label")
            
            start = time.time()
            
            # 发送开始消息（如果有 session）
            if session_id:
                _safe_send_start(session_id, node_name)
            
            try:
                result = await node_fn(state)
                return result
            finally:
                elapsed = time.time() - start
                logger.debug(f"[{node_name}] completed in {elapsed:.2f}s")
        
        return wrapped
    return decorator


def _safe_send_start(session_id: str, node_name: str):
    """安全发送开始消息
    
    发送失败不影响主流程，仅记录警告日志。
    """
    if not session_id:
        return
    try:
        from akg_agents.cli.runtime.message_sender import send_message
        from akg_agents.cli.messages import DisplayMessage
        send_message(session_id, DisplayMessage(text=f"▶ {node_name}"))
    except Exception as e:
        logger.warning(f"[{node_name}] send_message failed: {e}")

