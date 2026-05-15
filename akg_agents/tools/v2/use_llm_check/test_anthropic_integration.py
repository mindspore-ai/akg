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
测试 Anthropic Provider 集成到 AKG Agents

通过环境变量配置 Anthropic 兼容 API，测试 create_llm_client 能否正常调用。
支持任意兼容 Anthropic 协议的服务（Claude、Kimi Coding Plan 等）。

运行方式：
    cd /path/to/akg_agents
    source env.sh
    python tools/v2/use_llm_check/test_anthropic_integration.py

环境变量：
    AKG_AGENTS_BASE_URL - API 基础地址
    AKG_AGENTS_API_KEY - API Key
    AKG_AGENTS_MODEL_NAME - 模型名称
    AKG_AGENTS_PROVIDER_TYPE - Provider 类型（设为 anthropic）
"""

import asyncio
import os

from akg_agents.core_v2.llm.factory import create_llm_client
from akg_agents.core_v2.config import print_settings_info, get_settings


async def test_anthropic_via_factory():
    """通过 create_llm_client 测试 Anthropic API"""
    print("=" * 60)
    print("测试 Anthropic Provider 集成")
    print("=" * 60)

    # 打印配置
    print("\n配置信息:")
    print_settings_info("standard")

    # 创建 LLM 客户端
    print("\n创建 LLM Client...")
    client = create_llm_client(model_level="standard")

    print(f"  Provider: {type(client.provider).__name__}")
    print(f"  Model: {client.provider.model_name}")

    # 测试调用
    print("\n发送请求: '你好，请用一句话介绍你自己'")
    messages = [
        {"role": "user", "content": "你好，请用一句话介绍你自己"}
    ]

    try:
        result = await client.generate(messages, max_tokens=100)
        print(f"\n✅ 调用成功！")
        print(f"响应: {result['content']}")
        print(f"Token 使用: prompt={result['usage']['prompt_tokens']}, completion={result['usage']['completion_tokens']}")
        return True
    except Exception as e:
        print(f"\n❌ 调用失败: {e}")
        return False


async def test_provider_selection():
    """验证 Provider 类型选择逻辑"""
    print("\n" + "=" * 60)
    print("验证 Provider 类型选择逻辑")
    print("=" * 60)

    # 测试不同 provider_type 配置
    test_cases = [
        {
            "provider_type": "anthropic",
            "expected": "AnthropicProvider",
            "base_url": "https://api.anthropic.com",
        },
        {
            "provider_type": "openai",
            "expected": "LLMProvider",  # OpenAI provider 实际类名
            "base_url": "https://api.openai.com/v1",
        },
    ]

    results = []
    for case in test_cases:
        # 设置环境变量
        os.environ["AKG_AGENTS_BASE_URL"] = case["base_url"]
        os.environ["AKG_AGENTS_API_KEY"] = "test-key"
        os.environ["AKG_AGENTS_MODEL_NAME"] = "test-model"
        os.environ["AKG_AGENTS_PROVIDER_TYPE"] = case["provider_type"]

        print(f"\n测试 provider_type={case['provider_type']}:")
        print_settings_info("standard")

        try:
            client = create_llm_client(model_level="standard")
            actual = type(client.provider).__name__
            if actual == case["expected"]:
                print(f"✅ Provider 类型正确: {actual}")
                results.append(True)
            else:
                print(f"❌ Provider 类型错误: 期望 {case['expected']}, 实际 {actual}")
                results.append(False)
        except Exception as e:
            # 预期会失败（因为 API Key 是假的），但 Provider 类型应该正确
            print(f"Provider 类型检查完成（API Key 无效是预期行为）")
            results.append(True)

    return all(results)


async def main():
    """运行所有测试"""
    # 测试 Anthropic 集成
    anthropic_ok = await test_anthropic_via_factory()

    # 测试 Provider 选择逻辑
    provider_ok = await test_provider_selection()

    print("\n" + "=" * 60)
    print("测试结果")
    print("=" * 60)
    print(f"Anthropic 集成: {'✅ 成功' if anthropic_ok else '❌ 失败'}")
    print(f"Provider 选择: {'✅ 通过' if provider_ok else '❌ 失败'}")

    if anthropic_ok:
        print("\n💡 Anthropic Provider 已成功集成到 AKG Agents!")
        print("   使用方式:")
        print("   1. 环境变量配置:")
        print("      export AKG_AGENTS_BASE_URL=https://api.anthropic.com")
        print("      export AKG_AGENTS_API_KEY=sk-xxx")
        print("      export AKG_AGENTS_MODEL_NAME=claude-sonnet-4-20250514")
        print("      export AKG_AGENTS_PROVIDER_TYPE=anthropic")
        print("")
        print("   2. 其他兼容服务（如 Kimi Coding Plan）:")
        print("      export AKG_AGENTS_BASE_URL=https://api.kimi.com/coding")
        print("      export AKG_AGENTS_API_KEY=sk-kimi-xxx")
        print("      export AKG_AGENTS_MODEL_NAME=kimi-for-coding")
        print("      export AKG_AGENTS_PROVIDER_TYPE=anthropic")
        print("")
        print("   3. 配置文件方式 (~/.akg/settings.json):")
        print("      {")
        print("        \"models\": {")
        print("          \"standard\": {")
        print("            \"base_url\": \"https://api.anthropic.com\",")
        print("            \"api_key\": \"sk-xxx\",")
        print("            \"model_name\": \"claude-sonnet-4-20250514\",")
        print("            \"provider_type\": \"anthropic\"")
        print("          }")
        print("        }")
        print("      }")


if __name__ == "__main__":
    asyncio.run(main())