"""
LangGraph 节点埋点（给 CLI 的 WS 流式输出）。

本模块的职责：
- 将 LangGraph 的 node 函数包装为"带开始/结束事件"的版本；
- 通过 `send_message(session_id, Message)` 向当前 WebSocket session 推送：
  - `NodeStartMessage`
  - `NodeEndMessage`

触发策略：
- 当 `AIKG_STREAM_OUTPUT=on` 时，要求 `state` 中必须包含非空 `session_id`；
  否则会拒绝执行（避免事件无法路由到 CLI）。
- `AIKG_STREAM_OUTPUT` 关闭时，不强制要求 `session_id`；但如果提供了也会尝试发送事件。

注意：
- 发送消息失败不会影响主流程，会记录日志并继续（避免"过程展示"导致任务失败）。

使用方式：
```python
@track_node("designer")
async def designer_node(state):
    # 节点逻辑
    return state
```
"""

import functools
import time
from typing import Any, Callable, Dict

from textual import log as textual_log

from ai_kernel_generator.cli.messages import NodeEndMessage, NodeStartMessage
from ai_kernel_generator.cli.server.message_sender import send_message


def _stream_enabled() -> bool:
    import os

    return os.getenv("AIKG_STREAM_OUTPUT", "off").lower() == "on"


def _safe_send(session_id: str, message) -> None:
    """安全发送消息，失败不影响主流程"""
    if not session_id:
        return
    try:
        send_message(session_id, message)
    except Exception as e:
        textual_log.warning("[Node] send_message failed; ignored", exc_info=e)


def track_node(node_name: str):
    """
    节点函数装饰器，追踪节点执行并发送事件。

    Args:
        node_name: 节点名称，如 "designer", "coder", "verifier" 等

    Returns:
        装饰器函数

    Example:
        @track_node("designer")
        async def designer_node(state):
            # 节点逻辑
            return state
    """
    def decorator(node_fn: Callable):
        @functools.wraps(node_fn)
        async def wrapped(state: Dict[str, Any]):
            session_id = str(state.get("session_id") or "").strip()
            if _stream_enabled() and not session_id:
                raise ValueError(f"[{node_name}_node] state 中必须包含 session_id（AIKG_STREAM_OUTPUT=on）")

            task_id = str(state.get("task_id") or "")
            start = time.time()
            _safe_send(session_id, NodeStartMessage(node=node_name, task_id=task_id, state=state))

            try:
                result = await node_fn(state)
                _safe_send(
                    session_id,
                    NodeEndMessage(node=node_name, duration=time.time() - start, task_id=task_id, result=result),
                )
                return result
            except Exception as e:
                _safe_send(
                    session_id,
                    NodeEndMessage(
                        node=node_name,
                        duration=time.time() - start,
                        task_id=task_id,
                        result={"error": str(e)},
                    ),
                )
                raise

        return wrapped
    return decorator
