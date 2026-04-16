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
Anthropic LLM Provider - 支持 Anthropic 协议的 LLM 提供者

适用于：
- Kimi Coding Plan (api.kimi.com/coding)
- Claude API (api.anthropic.com)
- 其他使用 Anthropic 协议的 API

关键配置：
- base_url: 不带 /v1 后缀（Anthropic SDK 会自动追加 /v1/messages）
- model: kimi-for-coding / claude-3-5-sonnet-20241022 等

示例配置（settings.json）：
{
    "models": {
        "standard": {
            "base_url": "https://api.kimi.com/coding",
            "api_key": "sk-kimi-xxx",
            "model_name": "kimi-for-coding",
            "provider_type": "anthropic"
        }
    }
}
"""

import logging
import httpx
from typing import AsyncIterator, Dict, Any, List, Optional

try:
    from anthropic import AsyncAnthropic
except ImportError:
    AsyncAnthropic = None

logger = logging.getLogger(__name__)


class AnthropicProvider:
    """
    Anthropic 协议的 LLM 提供者

    支持 Kimi Coding Plan、Claude API 等 Anthropic 协议的服务。
    """

    def __init__(
        self,
        model_name: str,
        api_key: str,
        base_url: str = "https://api.anthropic.com",
        timeout: int = 300,
        extra_body: Optional[Dict[str, Any]] = None,
        verify_ssl: bool = False,
        **kwargs
    ):
        """
        初始化 Anthropic Provider

        Args:
            model_name: 模型名称（如 kimi-for-coding, claude-3-5-sonnet-20241022）
            api_key: API 密钥
            base_url: API 地址（不带 /v1，Anthropic SDK 会自动追加）
                     - Kimi: https://api.kimi.com/coding
                     - Claude: https://api.anthropic.com
            timeout: 超时时间（秒）
            extra_body: 额外请求参数（Anthropic 协议中较少使用）
            verify_ssl: 是否验证 SSL 证书（默认 False，用于企业代理环境）
            **kwargs: 其他配置
        """
        self.model_name = model_name
        self.extra_body = extra_body or {}
        self.verify_ssl = verify_ssl
        self.config = kwargs

        if AsyncAnthropic is None:
            raise ImportError("请安装 anthropic: pip install anthropic")

        # 创建自定义 http client 以支持 SSL 验证配置
        # httpx 默认会使用系统代理环境变量 (HTTP_PROXY, HTTPS_PROXY 等)
        http_client = httpx.AsyncClient(
            verify=verify_ssl,
            timeout=httpx.Timeout(timeout, connect=60.0),
        )

        # Anthropic SDK 会自动追加 /v1/messages
        # 所以 base_url 不应该带 /v1 后缀
        self.client = AsyncAnthropic(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            http_client=http_client,
        )

        logger.info(
            f"Initialized AnthropicProvider: model={model_name}, base_url={base_url}, verify_ssl={verify_ssl}"
        )

    async def generate(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """生成文本（非流式）"""

        # Anthropic API 需要分离 system message
        system_prompt = ""
        anthropic_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            else:
                anthropic_messages.append(msg)

        request_kwargs = {
            "model": self.model_name,
            "messages": anthropic_messages,
            **kwargs
        }

        if system_prompt:
            request_kwargs["system"] = system_prompt

        if tools:
            request_kwargs["tools"] = self._convert_tools(tools)

        response = await self.client.messages.create(**request_kwargs)

        # 转换为统一格式
        content = ""
        reasoning = ""
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                content += block.text
            elif block.type == "thinking":
                reasoning += getattr(block, "thinking", "")
            elif block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "type": "function",
                    "function": {
                        "name": block.name,
                        "arguments": block.input
                    }
                })

        result = {
            "content": content,
            "tool_calls": tool_calls,
            "reasoning_content": reasoning,
            "finish_reason": response.stop_reason or "",
            "usage": {}
        }

        if response.usage:
            result["usage"] = {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens
            }

        return result

    async def generate_stream(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict]] = None,
        **kwargs
    ) -> AsyncIterator[Dict[str, Any]]:
        """流式生成文本"""

        # 分离 system message
        system_prompt = ""
        anthropic_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            else:
                anthropic_messages.append(msg)

        request_kwargs = {
            "model": self.model_name,
            "messages": anthropic_messages,
            **kwargs
        }

        if system_prompt:
            request_kwargs["system"] = system_prompt

        if tools:
            request_kwargs["tools"] = self._convert_tools(tools)

        content = ""
        reasoning = ""
        tool_calls = []
        finish_reason = ""

        # 使用 async with 和 async for 处理异步流
        async with self.client.messages.stream(**request_kwargs) as stream:
            async for event in stream:
                if event.type == "content_block_delta":
                    delta = event.delta

                    if delta.type == "text_delta":
                        text_chunk = delta.text
                        content += text_chunk
                        yield {"type": "content", "chunk": text_chunk}

                    elif delta.type == "thinking_delta":
                        thinking_chunk = delta.thinking
                        reasoning += thinking_chunk
                        yield {"type": "reasoning", "chunk": thinking_chunk}

                elif event.type == "content_block_start":
                    block = event.content_block
                    if block.type == "tool_use":
                        tool_calls.append({
                            "id": block.id,
                            "type": "function",
                            "function": {
                                "name": block.name,
                                "arguments": {}
                            }
                        })

                elif event.type == "message_stop":
                    finish_reason = "end_turn"

            # 获取最终消息
            final_message = await stream.get_final_message()
            usage = final_message.usage

        yield {
            "type": "final",
            "content": content,
            "reasoning_content": reasoning,
            "finish_reason": finish_reason,
            "tool_calls": tool_calls,
            "usage": {
                "prompt_tokens": usage.input_tokens,
                "completion_tokens": usage.output_tokens,
                "total_tokens": usage.input_tokens + usage.output_tokens
            }
        }

    def _convert_tools(self, tools: List[Dict]) -> List[Dict]:
        """
        将 OpenAI 格式的 tools 转换为 Anthropic 格式

        OpenAI 格式：
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "...",
                "parameters": {...}
            }
        }

        Anthropic 格式：
        {
            "name": "get_weather",
            "description": "...",
            "input_schema": {...}
        }
        """
        anthropic_tools = []
        for tool in tools:
            if tool["type"] == "function":
                func = tool["function"]
                anthropic_tools.append({
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {})
                })
        return anthropic_tools

    def count_tokens(self, text: str) -> int:
        """计算 Token 数量（简单估算）"""
        # Anthropic 模型使用不同的 tokenization
        # 这里用简单估算，实际可以接入官方 tokenizer
        return len(text) // 4