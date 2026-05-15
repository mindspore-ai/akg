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
from __future__ import annotations

import logging
from typing import Any, Optional

import httpx
from langchain_core.language_models import LanguageModelInput
from langchain_deepseek import ChatDeepSeek

logger = logging.getLogger(__name__)


class ThinkingAwareChatDeepSeek(ChatDeepSeek):
    def __init__(
        self,
        *,
        timeout: Optional[httpx.Timeout] = None,
        **kwargs: Any,
    ):
        if timeout is None:
            timeout = httpx.Timeout(60, read=60 * 10)
        
        http_client = kwargs.pop("http_client", None)
        http_async_client = kwargs.pop("http_async_client", None)
        
        if http_client is None:
            http_client = httpx.Client(verify=False, timeout=timeout)
        if http_async_client is None:
            http_async_client = httpx.AsyncClient(verify=False, timeout=timeout)
        
        super().__init__(
            http_client=http_client,
            http_async_client=http_async_client,
            **kwargs,
        )
        
        logger.info("[ThinkingAwareChatDeepSeek] 初始化完成，支持 thinking mode（空 reasoning_content 模式）")
    
    def _get_request_payload(
        self,
        input_: LanguageModelInput,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict:
        """
        构建 API 请求 payload，为带有 tool_calls 的 assistant 消息注入空的 reasoning_content
        """
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        
        messages = payload.get("messages", [])
        injected_count = 0
        
        for msg in messages:
            if msg.get("role") != "assistant":
                continue
            
            tool_calls = msg.get("tool_calls", [])
            if not tool_calls:
                continue
            
            # 如果没有 reasoning_content，注入空字符串
            if not msg.get("reasoning_content"):
                msg["reasoning_content"] = ""
                injected_count += 1
        
        if injected_count > 0:
            logger.debug(
                f"[ThinkingAwareChatDeepSeek] 为 {injected_count} 个 assistant 消息注入空 reasoning_content"
            )
        
        return payload
