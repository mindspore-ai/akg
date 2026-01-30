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
LLMProvider - 统一 LLM 提供者

支持所有 OpenAI 兼容的 API：
- OpenAI (GPT-4, o1)
- DeepSeek
- Claude (通过 OpenAI 兼容层)
- 智谱 GLM
- Moonshot
- vLLM (本地部署)
- Ollama (本地)
"""

import logging
from typing import AsyncIterator, Dict, Any, List, Optional
from abc import ABC, abstractmethod

import httpx

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None

logger = logging.getLogger(__name__)


class LLMProvider:
    """
    统一 LLM 提供者（基于 OpenAI 兼容接口）
    
    支持所有 OpenAI 兼容的 API：
    - OpenAI (GPT-4, o1)
    - DeepSeek
    - Claude (通过 Anthropic 的 OpenAI 兼容层)
    - 智谱 GLM
    - Moonshot
    - vLLM (本地部署)
    - Ollama (本地)
    
    注：Claude 虽然有原生 API，但也支持 OpenAI 格式，足够覆盖 90%+ 使用场景
    """
    
    def __init__(
        self,
        model_name: str,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        timeout: int = 300,
        thinking_enabled: bool = False,
        **kwargs
    ):
        """
        初始化 LLM Provider
        
        Args:
            model_name: 模型名称（如 gpt-4, deepseek-chat, claude-3-5-sonnet-20241022）
            api_key: API 密钥
            base_url: API 地址（默认 OpenAI，可改为 DeepSeek、Claude 等）
            timeout: 超时时间（秒）
            thinking_enabled: 是否启用 thinking 模式（DeepSeek-R1 等思考模型）
            **kwargs: 其他配置
        """
        self.model_name = model_name
        self.thinking_enabled = thinking_enabled
        self.config = kwargs
        
        if AsyncOpenAI is None:
            raise ImportError("请安装 openai: pip install openai")
        
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            http_client=httpx.AsyncClient(verify=False, timeout=timeout)
        )
        
        logger.info(f"Initialized LLMProvider: model={model_name}, base_url={base_url}, thinking={thinking_enabled}")

    async def generate(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """生成文本（非流式）"""
        # 构建请求参数
        request_kwargs = {
            "model": self.model_name,
            "messages": messages,
            **kwargs
        }
        
        # 添加 tools（如果有）
        if tools:
            request_kwargs["tools"] = tools
        
        # 添加 thinking 模式
        if self.thinking_enabled:
            request_kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
        
        # 调用 OpenAI API
        response = await self.client.chat.completions.create(**request_kwargs)
        
        # 转换为统一格式
        choice = response.choices[0]
        result = {
            "content": choice.message.content or "",
            "tool_calls": [],
            "reasoning_content": getattr(choice.message, "reasoning_content", "") or "",
            "usage": {}
        }
        
        # 处理 tool_calls
        if choice.message.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                }
                for tc in choice.message.tool_calls
            ]
        
        # 处理 usage
        if response.usage:
            result["usage"] = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        
        return result
    
    async def generate_stream(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict]] = None,
        **kwargs
    ) -> AsyncIterator[Dict[str, Any]]:
        """流式生成文本"""
        request_kwargs = {
            "model": self.model_name,
            "messages": messages,
            "stream": True,
            **kwargs
        }
        
        if tools:
            request_kwargs["tools"] = tools
        
        # 添加 thinking 模式
        if self.thinking_enabled:
            request_kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
        
        # 累积结果
        content = ""
        reasoning = ""
        tool_calls = []
        
        stream_response = await self.client.chat.completions.create(**request_kwargs)
        
        async for chunk in stream_response:
            delta = chunk.choices[0].delta
            
            # 处理内容
            if delta.content:
                content += delta.content
                yield {"type": "content", "chunk": delta.content}
            
            # 处理推理（o1 等模型）
            if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                reasoning += delta.reasoning_content
                yield {"type": "reasoning", "chunk": delta.reasoning_content}
            
            # 处理工具调用
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    yield {"type": "tool_call", "tool_call": tc}
                    tool_calls.append(tc)
        
        # 最终结果
        yield {
            "type": "final",
            "content": content,
            "reasoning_content": reasoning,
            "tool_calls": tool_calls,
            "usage": {}  # 流式模式下 OpenAI 不返回 usage
        }
    
    def count_tokens(self, text: str) -> int:
        """计算 Token 数量（使用 tiktoken）"""
        try:
            import tiktoken
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except ImportError:
            # 简单估算：1 token ≈ 4 字符
            logger.warning("tiktoken not installed, using simple estimation")
            return len(text) // 4
        except Exception as e:
            logger.warning(f"Token counting failed: {e}, using simple estimation")
            return len(text) // 4
