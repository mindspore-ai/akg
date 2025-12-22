#!/usr/bin/env python3
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

"""
MainOpAgent 示例 - 对话式算子生成（支持多轮对话和用户确认）
"""

import asyncio
import logging
import os
import sys
from typing import List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ai_kernel_generator.core.agent.main_op_agent import MainOpAgent
from ai_kernel_generator.utils.common_utils import load_yaml
from ai_kernel_generator import get_project_root

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def display_agent_response(state: dict):
    """显示 Agent 的响应"""
    task_code = state.get('task_code', '')
    description = state.get('op_description', '')
    op_name = state.get('op_name', '')
    
    print("\n" + "=" * 80)
    print("🤖 AI Kernel Assistant")
    print("=" * 80)
    
    # 显示描述信息
    if description:
        print(f"\n{description}")
    
    # 如果有生成的代码，显示代码
    if task_code:
        print(f"\n📦 算子名称: {op_name}")
        print(f"\n📝 生成的 Task 代码:")
        print("-" * 80)
        print(task_code)
        print("-" * 80)


def display_generation_result(state: dict):
    """显示代码生成结果"""
    print("\n" + "=" * 80)
    print("🤖 AI Kernel Assistant - 代码生成结果")
    print("=" * 80)
    
    generation_success = state.get("generation_success")
    verification_result = state.get("verification_result")
    
    if generation_success:
        print("\n✅ 代码生成成功!")
        generated_code = state.get("generated_code", "")
        if generated_code:
            print(f"\n📝 生成的 Triton 代码 (前 500 字符):")
            print("-" * 80)
            print(generated_code[:500])
            if len(generated_code) > 500:
                print("...")
            print("-" * 80)
        
        if verification_result:
            print("\n✅ 验证通过!")
            profile = state.get("profile_result")
            if profile:
                print(f"📊 性能数据: {profile}")
        else:
            verification_error = state.get('verification_error', '未知错误')
            print(f"\n⚠️ 验证未通过: {verification_error}")
    else:
        generation_error = state.get('generation_error', '未知错误')
        print(f"\n❌ 代码生成失败: {generation_error}")


def is_simple_command(user_input: str) -> Tuple[bool, str]:
    """
    识别极简单的控制命令，这些命令不需要 LLM 分析，直接执行
    
    使用模糊匹配，提高识别率
    
    Returns:
        (is_command, command_type): 
            - is_command: 是否是简单命令
            - command_type: 'save' / 'exit' 之一
    """
    user_input_lower = user_input.strip().lower()
    
    # 如果输入太长（超过20个字符），不太可能是简单命令
    if len(user_input_lower) > 20:
        return False, ''
    
    # 退出相关的关键词（模糊匹配）
    exit_keywords = ['退出', 'quit', 'exit', '再见', 'bye', '结束', 'end', '拜拜']
    if any(kw in user_input_lower for kw in exit_keywords):
        # 排除一些不应该退出的情况
        # 例如："不退出"、"还不想退出"
        if not any(neg in user_input_lower for neg in ['不', '别', '不要', '不想', 'not', "don't"]):
            return True, 'exit'
    
    # 保存相关的关键词（模糊匹配）
    save_keywords = ['保存', 'save']
    if any(kw in user_input_lower for kw in save_keywords):
        # 排除一些不应该保存的情况
        if not any(neg in user_input_lower for neg in ['不', '别', '不要', 'not', "don't"]):
            return True, 'save'
    
    return False, ''


async def interactive_demo(
    backend: str = "cuda",
    arch: str = "a100",
    device_ids: List[int] = None,
    dsl: str = "triton",
    framework: str = "torch"
):
    """
    交互式演示 - 自然对话式交互
    
    Args:
        backend: 后端类型 (cuda, ascend, cpu)
        arch: 硬件架构 (a100, ascend910b4, etc.)
        device_ids: 设备ID列表 (默认: [0])
        dsl: 目标DSL (triton, ascendc, etc.)
        framework: 框架类型 (torch)
    """
    if device_ids is None:
        device_ids = [0]
    
    print("=" * 80)
    print("🚀 AI Kernel Assistant - 对话式算子生成")
    print("=" * 80)
    print(f"\n⚙️  硬件配置:")
    print(f"   • Backend: {backend}")
    print(f"   • Arch: {arch}")
    print(f"   • Device IDs: {device_ids}")
    print(f"   • DSL: {dsl}")
    print(f"   • Framework: {framework}")
    print("\n💡 提示: 您可以用自然语言描述您的算子需求")
    print("   • 输入 '退出' 结束对话")
    print("   • 输入 '保存' 保存对话历史")
    print("   • 其他输入将由 AI 智能理解您的意图")
    print()
    
    # 1. 加载配置
    config_path = os.path.join(get_project_root(), "config", "default_triton_cuda_config.yaml")
    config = load_yaml(config_path)
    logger.info(f"✓ Loaded config from: {config_path}")
    
    # 2. 注册 Worker
    logger.info(f"Registering local worker: backend={backend}, arch={arch}, devices={device_ids}")
    from ai_kernel_generator.core.worker.manager import register_local_worker
    try:
        await register_local_worker(
            device_ids=device_ids,
            backend=backend,
            arch=arch
        )
        logger.info("✓ Worker registered successfully")
    except Exception as e:
        logger.warning(f"Failed to register worker: {e}")
        logger.warning("Continuing without worker registration.")
    
    # 3. 创建 MainOpAgent
    agent = MainOpAgent(
        config=config,
        framework=framework,
        backend=backend,
        arch=arch,
        dsl=dsl
    )
    logger.info("✓ MainOpAgent initialized")
    
    # 4. 开始对话
    print("\n" + "-" * 80)
    user_request = input("👤 请输入您的需求: ")
    
    # 第一轮：生成 task 代码
    state = await agent.start_conversation(user_request)
    display_agent_response(state)
    
    # 判断当前状态
    has_task_code = bool(state.get('task_code'))
    if has_task_code:
        print("\n💡 请确认是否按照上述的任务描述开始生成，或者增加您想要的修改")
    
    # 5. 对话循环 - 使用智能的意图分析
    while True:
        print()
        user_input = input("👤 请输入您的需求: ").strip()
        
        if not user_input:
            print("💡 请输入您的回复")
            continue
        
        # 检查是否是简单的控制命令（不需要 LLM 分析）
        is_command, command_type = is_simple_command(user_input)
        
        # 处理退出命令（直接退出，不调用 LLM）
        if is_command and command_type == 'exit':
            print("\n🤖 好的，再见！")
            break
        
        # 处理保存命令
        if is_command and command_type == 'save':
            log_dir = os.path.expanduser(config.get("log_dir", "~/aikg_logs"))
            os.makedirs(log_dir, exist_ok=True)
            save_path = os.path.join(log_dir, f"conversation_{state.get('task_id')}.json")
            agent.save_conversation(state, save_path)
            print(f"\n🤖 ✅ 对话历史已保存到: {save_path}")
            print("   再见！")
            break
        
        # 其他复杂输入：使用 MainOpAgent 的智能意图分析（action="auto"）
        # LLM 会根据对话历史和当前状态自动判断用户意图
        print("\n🤖 正在处理您的需求...")
        state = await agent.continue_conversation(
            current_state=state,
            user_input=user_input,
            action="auto"  # 让 LLM 自动分析用户意图
        )
        
        # 根据返回的状态显示相应的结果
        has_task_code = bool(state.get('task_code'))
        has_generated_code = bool(state.get('generated_code'))
        current_step = state.get('current_step', '')
        
        # 如果用户选择了取消（正常退出）
        if current_step == 'cancelled' or not state.get('should_continue', True):
            print("\n🤖 好的，再见！")
            break
        
        # 如果用户输入无关问题（显示提示但继续对话）
        if current_step == 'irrelevant_input' or current_step == 'rejected_by_intent':
            display_agent_response(state)
            print("\n💡 请输入您的算子开发需求，或输入 '退出' 结束对话")
            continue
        
        # 显示结果
        if has_generated_code:
            # 已生成 Triton 代码
            display_generation_result(state)
            print("\n💡 您可以继续提问、输入 '保存' 保存结果，或 '退出' 结束对话")
        elif has_task_code:
            # 已生成或修改了 Task 代码
            display_agent_response(state)
            print("\n💡 请确认是否按照上述的任务描述开始生成，或者增加您想要的修改")
        else:
            # 需要更多信息
            display_agent_response(state)
    
    print("\n" + "=" * 80)
    print("🎉 感谢使用 AI Kernel Assistant!")
    print("=" * 80)


def main():
    """命令行入口"""
    asyncio.run(interactive_demo())


if __name__ == "__main__":
    main()

