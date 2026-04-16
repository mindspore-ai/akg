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
测试 Anthropic 兼容 API 调用

测试任意兼容 Anthropic 协议的 API（如 Claude、Kimi Coding Plan 等）。
使用 AKG Agents 统一配置系统，支持 ~/.akg/settings.json、.akg/settings.json、环境变量。

运行方式：
    cd /path/to/akg_agents
    source env.sh
    python tools/v2/use_llm_check/test_anthropic_api.py
"""

import asyncio
import os

from akg_agents.core_v2.config import get_settings, print_settings_info


async def test_anthropic_sdk():
    """使用 Anthropic SDK 测试 API"""
    print("\n" + "=" * 60)
    print("测试方法 1: Anthropic SDK (推荐)")
    print("=" * 60)

    try:
        from anthropic import AsyncAnthropic

        # 从 AKG Agents 配置获取参数
        settings = get_settings()
        model_config = settings.models.get("standard", {})

        api_key = model_config.api_key
        base_url = model_config.base_url
        model_name = model_config.model_name

        if api_key == "YOUR_API_KEY" or not api_key:
            print("❌ 请先配置 API Key（环境变量或 settings.json）")
            return False

        client = AsyncAnthropic(
            api_key=api_key,
            base_url=base_url,  # Anthropic SDK 会自动追加 /v1/messages
        )

        print(f"配置:")
        print(f"  base_url: {base_url}")
        print(f"  model: {model_name}")
        print(f"  实际请求路径: {base_url}/v1/messages")
        print()

        response = await client.messages.create(
            model=model_name,
            max_tokens=1024,
            messages=[
                {"role": "user", "content": "你好，请用一句话介绍你自己"}
            ]
        )

        print(f"✅ 请求成功！")
        print(f"响应:")
        for block in response.content:
            if hasattr(block, 'text'):
                print(f"  {block.text}")

        return True

    except ImportError:
        print("❌ 需要安装 anthropic SDK: pip install anthropic")
        return False
    except Exception as e:
        print(f"❌ Anthropic SDK 调用失败: {e}")
        return False


async def test_httpx_direct():
    """使用 httpx 直接发送 Anthropic 格式请求"""
    print("\n" + "=" * 60)
    print("测试方法 2: httpx 直接请求 (Anthropic 协议)")
    print("=" * 60)

    import httpx
    import json

    # 从 AKG Agents 配置获取参数
    settings = get_settings()
    model_config = settings.models.get("standard", {})

    api_key = model_config.api_key
    base_url = model_config.base_url
    model_name = model_config.model_name

    if api_key == "YOUR_API_KEY" or not api_key:
        print("❌ 请先配置 API Key（环境变量或 settings.json）")
        return False

    url = f"{base_url}/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_name,
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "你好，请用一句话介绍你自己"}]
    }

    print(f"请求详情:")
    print(f"  URL: {url}")
    print(f"  Headers: x-api-key={api_key[:20]}...")
    print(f"  Payload: {json.dumps(payload, ensure_ascii=False)}")

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(url, headers=headers, json=payload)

            print(f"\n响应状态码: {response.status_code}")

            if response.status_code == 200:
                data = response.json()
                print(f"✅ 请求成功！")
                print(f"响应: {json.dumps(data, ensure_ascii=False, indent=2)[:500]}...")
                return True
            else:
                print(f"❌ 请求失败")
                print(f"响应内容: {response.text}")
                return False

    except Exception as e:
        print(f"❌ httpx 请求失败: {e}")
        return False


async def main():
    """运行所有测试"""
    print("=" * 60)
    print("Anthropic 兼容 API 测试")
    print("=" * 60)

    # 打印配置信息
    print_settings_info("standard")

    # 从配置获取参数用于显示
    settings = get_settings()
    model_config = settings.models.get("standard", {})
    api_key = model_config.api_key or "YOUR_API_KEY"

    print(f"\n配置摘要:")
    print(f"  API Key: {api_key[:20]}...")

    # 测试 Anthropic SDK
    anthropic_ok = await test_anthropic_sdk()

    # 测试 httpx 直接请求
    httpx_ok = await test_httpx_direct()

    print("\n" + "=" * 60)
    print("测试结果总结")
    print("=" * 60)
    print(f"Anthropic SDK: {'✅ 成功' if anthropic_ok else '❌ 失败'}")
    print(f"httpx 直接请求: {'✅ 成功' if httpx_ok else '❌ 失败'}")

    if anthropic_ok or httpx_ok:
        print("\n💡 结论: API 使用 Anthropic 协议，调用成功")
        print("   配置方式:")
        print("   1. 环境变量:")
        print("      export AKG_AGENTS_BASE_URL=https://api.anthropic.com")
        print("      export AKG_AGENTS_API_KEY=sk-xxx")
        print("      export AKG_AGENTS_MODEL_NAME=claude-sonnet-4-20250514")
        print("      export AKG_AGENTS_PROVIDER_TYPE=anthropic")
        print("")
        print("   2. settings.json:")
        print("      ~/.akg/settings.json 或 .akg/settings.json")
    else:
        print("\n⚠️  所有测试都失败了，请检查:")
        print("   1. API Key 是否正确")
        print("   2. 网络是否能访问 API 服务")
        print("   3. provider_type 是否为 anthropic")


if __name__ == "__main__":
    asyncio.run(main())