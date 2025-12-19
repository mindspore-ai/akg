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
统一的事件消息类型定义

使用 Message 类型体系，在末端进行装包/拆包，中间层只传递消息对象。
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from datetime import datetime
from abc import ABC, abstractmethod


@dataclass
class Message(ABC):
    """事件消息基类"""

    # 使用 init=False 让 timestamp 不参与 __init__，而是在 __post_init__ 中设置
    timestamp: str = field(init=False)

    def __post_init__(self):
        """在初始化后设置 timestamp"""
        if not hasattr(self, "timestamp") or self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

    @abstractmethod
    def get_type(self) -> str:
        """获取消息类型标识"""
        pass

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典（用于网络传输）"""
        result = {"type": self.get_type(), "timestamp": self.timestamp}
        result.update(self._get_payload())
        return result

    @abstractmethod
    def _get_payload(self) -> Dict[str, Any]:
        """获取消息负载（子类实现）"""
        pass


# ==================== Node 事件 ====================


@dataclass
class NodeStartMessage(Message):
    """节点开始消息"""

    node: str
    # 任务ID（用于 evolve 并发时区分不同子任务；非 evolve 可为空）
    task_id: str = ""
    state: Dict = field(default_factory=dict)

    def get_type(self) -> str:
        return "node_start"

    def _get_payload(self) -> Dict[str, Any]:
        return {"node": self.node, "task_id": self.task_id}


@dataclass
class NodeEndMessage(Message):
    """节点结束消息"""

    node: str
    duration: float
    # 任务ID（用于 evolve 并发时区分不同子任务；非 evolve 可为空）
    task_id: str = ""
    result: Any = None

    def get_type(self) -> str:
        return "node_end"

    def _get_payload(self) -> Dict[str, Any]:
        return {
            "node": self.node,
            "duration": self.duration,
            "task_id": self.task_id,
            "result": self.result,
        }


# ==================== LLM 事件 ====================


@dataclass
class LLMStartMessage(Message):
    """LLM 开始调用消息"""

    agent: str
    model: str
    # 任务ID（用于 evolve 并发时区分不同子任务；非 evolve 可为空）
    task_id: str = ""

    def get_type(self) -> str:
        return "llm_start"

    def _get_payload(self) -> Dict[str, Any]:
        return {
            "agent": self.agent,
            "model": self.model,
            "task_id": self.task_id,
        }


@dataclass
class LLMEndMessage(Message):
    """LLM 结束调用消息"""

    agent: str
    model: str
    response: str
    duration: float
    # 任务ID（用于 evolve 并发时区分不同子任务；非 evolve 可为空）
    task_id: str = ""
    prompt_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None
    total_tokens: Optional[int] = None

    def get_type(self) -> str:
        return "llm_end"

    def _get_payload(self) -> Dict[str, Any]:
        return {
            "agent": self.agent,
            "model": self.model,
            "response": self.response,
            "duration": self.duration,
            "task_id": self.task_id,
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class LLMStreamMessage(Message):
    """LLM 流式输出消息"""

    agent: str
    chunk: str
    # 任务ID（用于 evolve 并发时区分不同子任务；非 evolve 可为空）
    task_id: str = ""

    def get_type(self) -> str:
        return "llm_stream"

    def _get_payload(self) -> Dict[str, Any]:
        return {
            "agent": self.agent,
            "chunk": self.chunk,
            "task_id": self.task_id,
        }


# ==================== 结果和错误消息 ====================


@dataclass
class FinalResultMessage(Message):
    """最终结果消息"""

    result: Dict

    def get_type(self) -> str:
        return "final_result"

    def _get_payload(self) -> Dict[str, Any]:
        return {"result": self.result}


@dataclass
class ErrorMessage(Message):
    """错误消息"""

    error: str

    def get_type(self) -> str:
        return "error"

    def _get_payload(self) -> Dict[str, Any]:
        return {"error": self.error}


@dataclass
class JobSubmittedMessage(Message):
    """Job 已提交消息（用于客户端获取 job_id，以支持取消/跟踪）"""

    job_id: str

    def get_type(self) -> str:
        return "job_submitted"

    def _get_payload(self) -> Dict[str, Any]:
        return {"job_id": self.job_id}


@dataclass
class ProgressMessage(Message):
    """进度消息（用于 search/evolve 多并发任务的进度展示）"""

    scope: str  # e.g. "evolve"
    data: Dict[str, Any] = field(default_factory=dict)

    def get_type(self) -> str:
        return "progress"

    def _get_payload(self) -> Dict[str, Any]:
        return {"scope": self.scope, "data": self.data}


# ==================== 消息注册表和工具函数 ====================

# 消息类型注册表：type -> Message 类
MESSAGE_REGISTRY: Dict[str, type] = {
    "node_start": NodeStartMessage,
    "node_end": NodeEndMessage,
    "llm_start": LLMStartMessage,
    "llm_end": LLMEndMessage,
    "llm_stream": LLMStreamMessage,
    "final_result": FinalResultMessage,
    "error": ErrorMessage,
    "job_submitted": JobSubmittedMessage,
    "progress": ProgressMessage,
}


def pack_message(msg: Message) -> Dict[str, Any]:
    """
    打包消息为字典（用于网络传输）

    Args:
        msg: Message 对象

    Returns:
        字典形式的消息
    """
    return msg.to_dict()


def unpack_message(data: Dict[str, Any]) -> Optional[Message]:
    """
    从字典解包为 Message 对象

    Args:
        data: 字典形式的消息

    Returns:
        Message 对象，如果类型未知则返回 None
    """
    msg_type = data.get("type")
    msg_class = MESSAGE_REGISTRY.get(msg_type)

    if not msg_class:
        return None

    # 从字典构造消息对象（不包括 type 和 timestamp）
    payload = {k: v for k, v in data.items() if k not in ("type", "timestamp")}
    message = msg_class(**payload)

    # 手动设置 timestamp（如果提供的话）
    if "timestamp" in data:
        message.timestamp = data["timestamp"]

    return message
