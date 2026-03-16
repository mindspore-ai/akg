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
测试 LLM Provider 和 Client
"""

import os
import pytest

os.environ['AKG_AGENTS_STREAM_OUTPUT'] = 'on'

from akg_agents.core_v2.llm import (
    create_llm_client,
    LLMProvider,
    LLMClient
)


class TestLLMProvider:
    """测试 LLM Provider"""
    
    def test_provider_init_openai(self, monkeypatch):
        """测试 LLM Provider 初始化（OpenAI）"""
        monkeypatch.setenv("AKG_AGENTS_API_KEY", "sk-test")
        
        provider = LLMProvider(
            model_name="gpt-4",
            api_key="sk-test",
            base_url="https://api.openai.com/v1"
        )
        
        assert provider.model_name == "gpt-4"
        assert provider.client is not None
    
    def test_provider_init_claude(self):
        """测试 LLM Provider 初始化（Claude 通过 OpenAI 兼容层）"""
        provider = LLMProvider(
            model_name="claude-3-5-sonnet-20241022",
            api_key="sk-test",
            base_url="https://api.anthropic.com/v1"  # OpenAI 兼容层
        )
        assert provider.model_name == "claude-3-5-sonnet-20241022"
        assert "anthropic" in str(provider.client.base_url)


class TestLLMClient:
    """测试 LLM Client"""
    
    def test_llm_client_init(self, monkeypatch):
        """测试 LLM Client 初始化"""
        monkeypatch.setenv("AKG_AGENTS_API_KEY", "sk-test")
        
        provider = LLMProvider(
            model_name="gpt-4",
            api_key="sk-test"
        )
        
        client = LLMClient(
            provider=provider,
            temperature=0.7,
            max_tokens=1000
        )
        
        assert client.provider == provider
        assert client.default_config["temperature"] == 0.7
        assert client.default_config["max_tokens"] == 1000
    
    def test_token_stats(self, monkeypatch):
        """测试 Token 统计功能"""
        monkeypatch.setenv("AKG_AGENTS_API_KEY", "sk-test")
        
        provider = LLMProvider(
            model_name="gpt-4",
            api_key="sk-test"
        )
        client = LLMClient(provider=provider)
        
        assert client.get_total_tokens() == 0
        assert client.get_prompt_tokens() == 0
        assert client.get_completion_tokens() == 0
        
        # 重置
        client.reset_token_stats()
        assert client.get_total_tokens() == 0


class TestLLMFactory:
    """测试 LLM 工厂函数"""
    
    def test_create_llm_client_default(self, monkeypatch):
        """测试使用默认配置创建 Client"""
        monkeypatch.setenv("AKG_AGENTS_API_KEY", "sk-test")
        monkeypatch.setenv("AKG_AGENTS_MODEL_NAME", "gpt-4")
        
        client = create_llm_client()
        
        assert isinstance(client, LLMClient)
        assert isinstance(client.provider, LLMProvider)
    
    def test_create_llm_client_custom(self, monkeypatch):
        """测试使用自定义参数创建 Client"""
        monkeypatch.setenv("AKG_AGENTS_API_KEY", "sk-test")
        
        client = create_llm_client(
            model_name="gpt-3.5-turbo",
            temperature=0.5
        )
        
        assert isinstance(client, LLMClient)
        assert client.provider.model_name == "gpt-3.5-turbo"
        assert client.default_config["temperature"] == 0.5
    
    def test_create_llm_client_claude(self):
        """测试创建 Claude Client（通过 OpenAI 兼容层）"""
        client = create_llm_client(
            model_name="claude-3-5-sonnet-20241022",
            base_url="https://api.anthropic.com/v1",
            api_key="sk-test"
        )
        assert isinstance(client, LLMClient)
        assert isinstance(client.provider, LLMProvider)
        assert client.provider.model_name == "claude-3-5-sonnet-20241022"


@pytest.mark.use_model
class TestLLMIntegration:
    """集成测试（需要真实 API）"""
    
    @pytest.mark.asyncio
    async def test_openai_generate(self):
        """测试 OpenAI 生成"""
        api_key = os.getenv("AKG_AGENTS_API_KEY")
        if not api_key:
            pytest.skip("需要设置 AKG_AGENTS_API_KEY")
        
        # 使用配置中的 model（避免 gpt-4 在 DeepSeek 上不存在）
        client = create_llm_client(temperature=0.0)
        
        messages = [
            {"role": "user", "content": "1+1等于多少？"}
        ]
        
        result = await client.generate(messages)
        
        assert "content" in result
        assert isinstance(result["content"], str)
        assert len(result["content"]) > 0
        
        # Token 统计
        assert client.get_total_tokens() > 0
        
        print(f"\n✅ 回复: {result['content']}")
        print(f"📊 Token 使用: {client.get_total_tokens()}")
    
    @pytest.mark.asyncio
    async def test_openai_with_tools(self):
        """测试 OpenAI 工具调用"""
        api_key = os.getenv("AKG_AGENTS_API_KEY")
        if not api_key:
            pytest.skip("需要设置 AKG_AGENTS_API_KEY")
        
        client = create_llm_client()
        
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "获取天气信息",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"}
                        },
                        "required": ["city"]
                    }
                }
            }
        ]
        
        messages = [
            {"role": "user", "content": "北京今天天气怎么样？"}
        ]
        
        result = await client.generate(messages, tools=tools)
        
        # 打印结果
        print(f"\n📊 结果:")
        print(f"  content: {result.get('content')}")
        print(f"  tool_calls: {len(result.get('tool_calls', []))} 个")
        
        # 验证（可能有 content 或 tool_calls）
        assert result.get("content") or result.get("tool_calls")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
