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
测试 Kimi Coding Plan API 调用

Kimi Coding Plan 使用 Anthropic 协议，不是 OpenAI 协议。
关键配置：
- baseUrl: https://api.kimi.com/coding (不带 /v1，因为 Anthropic SDK 会自动追加 /v1/messages)
- model: kimi-for-coding
- protocol: anthropic-messages

运行方式：
    cd /path/to/akg_agents
    python tools/v2/use_llm_check/test_kimi_api.py
"""

import asyncio
import os

# Kimi Coding Plan 配置
KIMI_API_KEY = os.getenv("KIMI_API_KEY", "YOUR_API_KEY")
KIMI_BASE_URL = os.getenv("KIMI_BASE_URL", "https://api.kimi.com/coding")
KIMI_MODEL = os.getenv("KIMI_MODEL", "kimi-for-coding")


async def test_anthropic_sdk():
    """使用 Anthropic SDK 测试 Kimi API"""
    print("\n" + "=" * 60)
    print("测试方法 1: Anthropic SDK (推荐)")
    print("=" * 60)

    try:
        from anthropic import AsyncAnthropic

        client = AsyncAnthropic(
            api_key=KIMI_API_KEY,
            base_url=KIMI_BASE_URL,  # Anthropic SDK 会自动追加 /v1/messages
        )

        print(f"配置:")
        print(f"  base_url: {KIMI_BASE_URL}")
        print(f"  model: {KIMI_MODEL}")
        print(f"  实际请求路径: {KIMI_BASE_URL}/v1/messages")
        print()

        response = await client.messages.create(
            model=KIMI_MODEL,
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


async def test_openai_sdk():
    """使用 OpenAI SDK 测试 Kimi API（不推荐，可能失败）"""
    print("\n" + "=" * 60)
    print("测试方法 2: OpenAI SDK (不推荐 - Kimi 不支持 OpenAI 协议)")
    print("=" * 60)

    try:
        from openai import AsyncOpenAI

        # 尝试不同的 base_url 配置
        configs = [
            # 配置 1: 带 /v1 后缀
            {
                "base_url": KIMI_BASE_URL + "/v1",
                "endpoint": "/chat/completions",
                "desc": "带 /v1 后缀 (可能拼成 /v1/v1/chat/completions 导致 404)"
            },
            # 配置 2: 不带 /v1 后缀
            {
                "base_url": KIMI_BASE_URL,
                "endpoint": "/v1/chat/completions",
                "desc": "不带 /v1 后缀 (标准 OpenAI 路径)"
            },
        ]

        for config in configs:
            print(f"\n尝试配置: {config['desc']}")
            print(f"  base_url: {config['base_url']}")
            print(f"  期望 endpoint: {config['endpoint']}")

            try:
                client = AsyncOpenAI(
                    api_key=KIMI_API_KEY,
                    base_url=config["base_url"],
                )

                response = await client.chat.completions.create(
                    model=KIMI_MODEL,
                    messages=[{"role": "user", "content": "你好"}],
                    max_tokens=100,
                )

                print(f"✅ 成功！响应: {response.choices[0].message.content}")
                return True

            except Exception as e:
                print(f"❌ 失败: {e}")

        return False

    except ImportError:
        print("❌ 需要安装 openai SDK: pip install openai")
        return False


async def test_httpx_direct():
    """使用 httpx 直接发送 Anthropic 格式请求"""
    print("\n" + "=" * 60)
    print("测试方法 3: httpx 直接请求 (Anthropic 协议)")
    print("=" * 60)

    import httpx
    import json

    url = f"{KIMI_BASE_URL}/v1/messages"
    headers = {
        "x-api-key": KIMI_API_KEY,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }
    payload = {
        "model": KIMI_MODEL,
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "你好，请用一句话介绍你自己"}]
    }

    print(f"请求详情:")
    print(f"  URL: {url}")
    print(f"  Headers: x-api-key={KIMI_API_KEY[:20]}...")
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
    print("Kimi Coding Plan API 测试")
    print("=" * 60)
    print(f"API Key: {KIMI_API_KEY[:20]}...")
    print(f"Base URL: {KIMI_BASE_URL}")
    print(f"Model: {KIMI_MODEL}")

    # 测试 Anthropic SDK
    anthropic_ok = await test_anthropic_sdk()

    # 测试 httpx 直接请求
    httpx_ok = await test_httpx_direct()

    # 测试 OpenAI SDK (预期会失败)
    openai_ok = await test_openai_sdk()

    print("\n" + "=" * 60)
    print("测试结果总结")
    print("=" * 60)
    print(f"Anthropic SDK: {'✅ 成功' if anthropic_ok else '❌ 失败'}")
    print(f"httpx 直接请求: {'✅ 成功' if httpx_ok else '❌ 失败'}")
    print(f"OpenAI SDK: {'✅ 成功' if openai_ok else '❌ 失败 (预期)'}")

    if anthropic_ok or httpx_ok:
        print("\n💡 结论: Kimi Coding Plan 使用 Anthropic 协议")
        print("   项目需要添加 Anthropic Provider 支持才能使用 Kimi API")
    else:
        print("\n⚠️  所有测试都失败了，请检查:")
        print("   1. API Key 是否正确")
        print("   2. 网络是否能访问 api.kimi.com")
        print("   3. Kimi Coding Plan 是否已激活")


if __name__ == "__main__":
    asyncio.run(main())