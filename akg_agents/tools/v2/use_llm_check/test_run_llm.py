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
使用 core_v2 AgentBase 测试 LLM 调用

配置方式：
1. ~/.akg/settings.json
2. .akg/settings.json
3. 环境变量 AKG_AGENTS_*

运行：
    cd /path/to/akg_agents
    conda activate akg_agents
    source env.sh
    python tools/v2/use_llm_check/test_run_llm.py
"""

import asyncio

from akg_agents.core_v2.agents import AgentBase, Jinja2TemplateWrapper, register_agent
from akg_agents.core_v2.config import get_settings, print_settings_info


@register_agent
class TestAgent(AgentBase):
    """测试用的 Agent 类"""

    def __init__(self):
        context = {"agent_name": "TestAgent"}
        super().__init__(context=context)

    async def run(self, prompt: str, model_level: str = "standard"):
        """运行 LLM 调用"""
        template = Jinja2TemplateWrapper("{{ prompt }}")
        return await self.run_llm(template, {"prompt": prompt}, model_level)


async def test_simple_run_llm():
    """简单的 LLM 测试"""
    # 打印配置信息
    print_settings_info()
    
    # 创建测试 Agent
    agent = TestAgent()

    # 自定义 prompt
    prompt = "你好，请用一句话介绍一下自己"

    # 指定 model level（从 settings.json 中选择）
    model_level = "standard"

    print(f"\n🚀 开始测试 run_llm")
    print(f"   Prompt: {prompt}")
    print(f"   Model Level: {model_level}")

    # 调用 run_llm
    content, formatted_prompt, reasoning_content = await agent.run(prompt, model_level)

    # 输出结果
    print(f"\n=== 测试结果 ===")
    print(f"📝 Formatted Prompt: {formatted_prompt}")
    if reasoning_content:
        print(f"🧠 Reasoning: {reasoning_content[:200]}...")
    print(f"💬 Output: {content}")
    print("================")


if __name__ == "__main__":
    asyncio.run(test_simple_run_llm())
