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


# ==================== 文本显示消息 ====================


@dataclass
class DisplayMessage(Message):
    """纯文本显示消息"""

    text: str

    def get_type(self) -> str:
        return "display"

    def _get_payload(self) -> Dict[str, Any]:
        return {"text": self.text}


# ==================== LLM 流式消息 ====================


@dataclass
class LLMStreamMessage(Message):
    """LLM 流式输出消息"""

    agent: str
    chunk: str
    # 是否为推理内容（由输出协议标识；UI 不应猜测）
    is_reasoning: bool = False

    def get_type(self) -> str:
        return "llm_stream"

    def _get_payload(self) -> Dict[str, Any]:
        return {
            "agent": self.agent,
            "chunk": self.chunk,
            "is_reasoning": self.is_reasoning,
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


# ==================== 面板数据消息 ====================


@dataclass
class PanelDataMessage(Message):
    """面板数据更新消息"""

    action: str
    data: Dict[str, Any] = field(default_factory=dict)

    def get_type(self) -> str:
        return "panel_data"

    def _get_payload(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "data": self.data,
        }


# ==================== 消息注册表和工具函数 ====================

# 消息类型注册表：type -> Message 类
MESSAGE_REGISTRY: Dict[str, type] = {
    "display": DisplayMessage,
    "llm_stream": LLMStreamMessage,
    "final_result": FinalResultMessage,
    "error": ErrorMessage,
    "panel_data": PanelDataMessage,
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
