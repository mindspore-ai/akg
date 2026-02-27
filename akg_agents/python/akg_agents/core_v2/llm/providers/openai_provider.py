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
- OpenAI (GPT-4, o1/o3)
- DeepSeek
- Claude (通过 OpenAI 兼容层)
- 智谱 GLM
- Moonshot / Kimi
- 通义千问 / DashScope
- 豆包 / 火山引擎
- vLLM (本地部署)
- Ollama (本地)

不同 provider 的 thinking/reasoning 参数通过 settings.json 中的 extra_body 字段
直接配置并透传到 API 请求，无需框架做 provider 检测。
详见 settings.example.more.json 中各 provider 的配置示例。
"""

import logging
from typing import AsyncIterator, Dict, Any, List, Optional

import httpx

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None

logger = logging.getLogger(__name__)


class LLMProvider:
    """
    统一 LLM 提供者（基于 OpenAI 兼容接口）
    
    通过 extra_body 机制支持各 provider 的差异化参数（如 thinking/reasoning），
    用户在 settings.json 中直接配置 extra_body，框架原样透传到 API 请求。
    """
    
    def __init__(
        self,
        model_name: str,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        timeout: int = 300,
        extra_body: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        初始化 LLM Provider
        
        Args:
            model_name: 模型名称
            api_key: API 密钥
            base_url: API 地址
            timeout: 超时时间（秒）
            extra_body: 额外请求体参数，直接透传到 API 请求。
                        用于配置各 provider 的 thinking/reasoning 等特殊参数。
                        例如 DeepSeek: {"thinking": {"type": "enabled"}}
                        例如 OpenAI o3: {"reasoning_effort": "high"}
            **kwargs: 其他配置
        """
        self.model_name = model_name
        self.extra_body = extra_body or {}
        self.config = kwargs
        
        if AsyncOpenAI is None:
            raise ImportError("请安装 openai: pip install openai")
        
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            http_client=httpx.AsyncClient(verify=False, timeout=timeout)
        )
        
        logger.info(
            f"Initialized LLMProvider: model={model_name}, base_url={base_url}, "
            f"extra_body={bool(self.extra_body)}"
        )

    async def generate(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """生成文本（非流式）"""
        request_kwargs = {
            "model": self.model_name,
            "messages": messages,
            **kwargs
        }
        
        if tools:
            request_kwargs["tools"] = tools
        
        # 透传 extra_body（合并调用方可能传入的 extra_body）
        if self.extra_body:
            existing = request_kwargs.get("extra_body", {})
            request_kwargs["extra_body"] = {**self.extra_body, **existing}
        
        response = await self.client.chat.completions.create(**request_kwargs)
        
        # 转换为统一格式
        choice = response.choices[0]
        result = {
            "content": choice.message.content or "",
            "tool_calls": [],
            "reasoning_content": getattr(choice.message, "reasoning_content", "") or "",
            "usage": {}
        }
        
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
        
        # 透传 extra_body
        if self.extra_body:
            existing = request_kwargs.get("extra_body", {})
            request_kwargs["extra_body"] = {**self.extra_body, **existing}
        
        content = ""
        reasoning = ""
        # tool_call deltas 按 index 累积，流结束后合并为完整 tool_calls
        pending_tool_calls: Dict[int, Dict[str, Any]] = {}
        
        stream_response = await self.client.chat.completions.create(**request_kwargs)
        
        async for chunk in stream_response:
            delta = chunk.choices[0].delta
            
            if delta.content:
                content += delta.content
                yield {"type": "content", "chunk": delta.content}
            
            # 统一抽象：兼容 reasoning_content / reasoning 字段
            reasoning_chunk = (
                getattr(delta, "reasoning_content", None)
                or getattr(delta, "reasoning", None)
            )
            if reasoning_chunk:
                reasoning += reasoning_chunk
                yield {"type": "reasoning", "chunk": reasoning_chunk}
            
            # 累积 tool_call deltas（OpenAI 流式分多次发送 name 和 arguments 片段）
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in pending_tool_calls:
                        pending_tool_calls[idx] = {
                            "id": "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        }
                    tc = pending_tool_calls[idx]
                    if tc_delta.id:
                        tc["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            tc["function"]["name"] += tc_delta.function.name
                        if tc_delta.function.arguments:
                            tc["function"]["arguments"] += tc_delta.function.arguments
        
        tool_calls = [pending_tool_calls[i] for i in sorted(pending_tool_calls)]
        
        yield {
            "type": "final",
            "content": content,
            "reasoning_content": reasoning,
            "tool_calls": tool_calls,
            "usage": {}
        }
    
    def count_tokens(self, text: str) -> int:
        """计算 Token 数量（使用 tiktoken）"""
        try:
            import tiktoken
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except ImportError:
            logger.warning("tiktoken not installed, using simple estimation")
            return len(text) // 4
        except Exception as e:
            logger.warning(f"Token counting failed: {e}, using simple estimation")
            return len(text) // 4
