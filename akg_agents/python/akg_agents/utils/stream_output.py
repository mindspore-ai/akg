from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator, Optional, Union

# 说明：
# - os.environ 是进程全局，异步并发/嵌套调用下“临时改 env”会互相覆盖，导致流式消息乱序或 UI 冲突。
# - ContextVar 是协程上下文级别的开关，适合在 async 任务里做“临时覆盖且可恢复”。

_STREAM_OUTPUT_OVERRIDE: ContextVar[Optional[bool]] = ContextVar(
    "AKG_AGENTS_STREAM_OUTPUT_OVERRIDE", default=None
)


def get_stream_output_override() -> Optional[bool]:
    """返回当前协程上下文中的 stream 开关覆盖值；None 表示未覆盖。"""
    return _STREAM_OUTPUT_OVERRIDE.get()


@contextmanager
def stream_output_override(value: Union[bool, None]) -> Iterator[None]:
    """
    临时覆盖“是否允许发送 LLMStreamMessage”（影响 AgentBase._stream_enabled 等读取点）。

    - value=True：强制视为 stream on
    - value=False：强制视为 stream off
    - value=None：取消覆盖，回退到 os.environ["AKG_AGENTS_STREAM_OUTPUT"]
    """
    token = _STREAM_OUTPUT_OVERRIDE.set(value)
    try:
        yield
    finally:
        _STREAM_OUTPUT_OVERRIDE.reset(token)

