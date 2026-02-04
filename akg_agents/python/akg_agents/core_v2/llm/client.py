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

"""
LLMClient - LLM 客户端（业务封装层）

封装 LLMProvider，提供：
- Token 统计
- 流式 UI 发送（通过 session_id）
- 统一的调用接口
"""

import logging
from typing import AsyncIterator, Dict, Any, List, Optional
from .providers.openai_provider import LLMProvider

logger = logging.getLogger(__name__)


class LLMClient:
    """
    LLM 客户端 - 业务层封装
    
    封装 LLMProvider，提供统一的调用接口和额外功能：
    - Token 统计
    - 流式 UI 发送（通过 session_id + send_message）
    - 自动处理 reasoning_content
    """
    
    def __init__(
        self,
        provider: LLMProvider,
        session_id: str = None,
        temperature: float = 0.2,
        max_tokens: int = 8192,
        top_p: float = 0.9,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        **kwargs
    ):
        """
        初始化 LLM 客户端
        
        Args:
            provider: LLM 提供者实例
            session_id: UI 会话 ID（流式输出时用于发送消息到 UI）
            temperature: 温度参数
            max_tokens: 最大 token 数
            top_p: Top-p 采样
            frequency_penalty: 频率惩罚（可选，来自配置文件）
            presence_penalty: 存在惩罚（可选，来自配置文件）
            **kwargs: 其他默认配置
        """
        self.provider = provider
        self.session_id = session_id
        self.default_config = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            **kwargs
        }
        # 可选参数仅在设置时包含
        if frequency_penalty is not None:
            self.default_config["frequency_penalty"] = frequency_penalty
        if presence_penalty is not None:
            self.default_config["presence_penalty"] = presence_penalty
        
        # Token 统计
        self._total_tokens = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0
        
        logger.info(f"Initialized LLMClient with provider={provider.model_name}, session_id={bool(session_id)}")
    
    async def generate(
        self,
        messages: List[Dict[str, str]],
        stream: bool = False,
        agent_name: str = "",
        tools: Optional[List[Dict]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        统一生成接口
        
        Args:
            messages: 消息列表 [{"role": "user", "content": "..."}]
            stream: 是否使用流式输出
            agent_name: Agent 名称（用于流式消息标识）
            tools: 工具定义（OpenAI function calling 格式）
            **kwargs: 额外参数（会覆盖默认配置）
        
        Returns:
            {
                "content": "回复内容",
                "reasoning_content": "推理过程（如果有）",
                "tool_calls": [...],
                "usage": {...}
            }
        """
        if stream:
            return await self._generate_stream(messages, agent_name, tools=tools, **kwargs)
        
        # 合并配置
        config = {**self.default_config, **kwargs}
        
        # 调用 Provider
        result = await self.provider.generate(messages, tools=tools, **config)
        
        # 统计 Token
        self._update_token_stats(result.get("usage", {}))
        
        return result
    
    async def _generate_stream(
        self,
        messages: List[Dict[str, str]],
        agent_name: str = "",
        tools: Optional[List[Dict]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        流式生成 + 发送到 UI
        
        Args:
            messages: 消息列表
            agent_name: Agent 名称
            tools: 工具定义
            **kwargs: 额外参数
        
        Returns:
            完整的生成结果
        """
        config = {**self.default_config, **kwargs}
        
        content = ""
        reasoning_content = ""
        tool_calls = []
        
        async for chunk in self.provider.generate_stream(messages, tools=tools, **config):
            chunk_type = chunk.get("type")
            
            if chunk_type == "content":
                chunk_text = chunk.get("chunk", "")
                content += chunk_text
                self._safe_send_stream(agent_name, chunk_text, is_reasoning=False)
            
            elif chunk_type == "reasoning":
                chunk_text = chunk.get("chunk", "")
                reasoning_content += chunk_text
                self._safe_send_stream(agent_name, chunk_text, is_reasoning=True)
            
            elif chunk_type == "tool_call":
                tool_calls.append(chunk.get("tool_call"))
            
            elif chunk_type == "final":
                # 最终结果，更新 Token 统计
                self._update_token_stats(chunk.get("usage", {}))
        
        # 发送流式结束标记
        self._safe_send_display("")
        
        return {
            "content": content,
            "reasoning_content": reasoning_content,
            "tool_calls": tool_calls,
            "usage": {}  # 流式模式下 OpenAI 不返回 usage
        }
    
    def _safe_send_stream(self, agent_name: str, chunk: str, is_reasoning: bool = False) -> None:
        """
        安全发送流式消息到 UI
        
        Args:
            agent_name: Agent 名称
            chunk: 消息片段
            is_reasoning: 是否是推理内容
        """
        if not chunk:
            return
        
        if not self.session_id:
            # 如果没有 session_id，降级为控制台打印
            print(chunk, end="", flush=True)
            return
        
        try:
            from akg_agents.cli.messages import LLMStreamMessage
            from akg_agents.cli.runtime.message_sender import send_message
            
            send_message(
                self.session_id,
                LLMStreamMessage(
                    agent=agent_name,
                    chunk=chunk,
                    is_reasoning=is_reasoning,
                )
            )
        except ImportError:
            logger.debug("Message sending not available (cli module not installed)")
        except Exception as e:
            logger.warning(f"Failed to send stream message: {e}")
    
    def _safe_send_display(self, text: str) -> None:
        """
        发送显示消息（用于流式结束）
        
        Args:
            text: 显示文本
        """
        if not self.session_id:
            return
        
        try:
            from akg_agents.cli.messages import DisplayMessage
            from akg_agents.cli.runtime.message_sender import send_message
            
            send_message(self.session_id, DisplayMessage(text=text))
        except ImportError:
            logger.debug("Message sending not available (cli module not installed)")
        except Exception as e:
            logger.warning(f"Failed to send display message: {e}")
    
    def _update_token_stats(self, usage: Dict[str, Any]) -> None:
        """更新 Token 统计"""
        if not usage:
            return
        
        self._prompt_tokens += usage.get("prompt_tokens", 0)
        self._completion_tokens += usage.get("completion_tokens", 0)
        self._total_tokens += usage.get("total_tokens", 0)
        
        logger.debug(
            f"LLM generate: prompt={usage.get('prompt_tokens', 0)}, "
            f"completion={usage.get('completion_tokens', 0)}, "
            f"total={usage.get('total_tokens', 0)}"
        )
    
    def get_total_tokens(self) -> int:
        """获取总 Token 使用量"""
        return self._total_tokens
    
    def get_prompt_tokens(self) -> int:
        """获取 Prompt Token 使用量"""
        return self._prompt_tokens
    
    def get_completion_tokens(self) -> int:
        """获取 Completion Token 使用量"""
        return self._completion_tokens
    
    def reset_token_stats(self) -> None:
        """重置 Token 统计"""
        self._total_tokens = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0
        logger.debug("Token statistics reset")
    
    def __repr__(self) -> str:
        return f"LLMClient(model={self.provider.model_name}, session={bool(self.session_id)})"
