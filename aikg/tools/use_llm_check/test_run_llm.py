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

import asyncio
from langchain.prompts import PromptTemplate
from ai_kernel_generator.core.agent.agent_base import AgentBase


class TestAgent(AgentBase):
    """测试用的Agent类，继承自AgentBase"""

    def __init__(self):
        agent_details = {"agent_name": "test_agent"}
        super().__init__(agent_details=agent_details)


async def test_simple_run_llm():
    """简单的VLLM测试"""
    # 创建测试Agent
    agent = TestAgent()

    # 自定义prompt string
    prompt_string = "你好，请简单介绍一下自己"

    # 创建提示模板
    prompt = PromptTemplate.from_template(prompt_string)

    # 准备输入
    input_data = {}

    # 指定model name
    model_name = "vllm_deepseek_r1_default"

    # 跑run_llm
    content, formatted_prompt, reasoning_content = await agent.run_llm(
        prompt, input_data, model_name
    )

    # 输出结果
    print(f"\n=== run_llm测试 ===")
    print(f"Prompt: {formatted_prompt}")
    if reasoning_content:
        print(f"Reasoning: {reasoning_content}")
    print(f"Model: {model_name}")
    print(f"Output: {content}")
    print("===================")


if __name__ == "__main__":
    asyncio.run(test_simple_run_llm())
