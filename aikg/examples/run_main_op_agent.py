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
MainOpAgent 对话式算子生成
"""

import asyncio
import logging
import os
import sys
from typing import List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ai_kernel_generator.core.agent.main_op_agent import MainOpAgent
from ai_kernel_generator.utils.common_utils import load_yaml
from ai_kernel_generator.utils.main_op_agent_display import is_simple_command
from ai_kernel_generator import get_project_root

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def interactive_demo(
    backend: str = "cuda",
    arch: str = "a100",
    device_ids: List[int] = None,
    dsl: str = "triton",
    framework: str = "torch"
):
    """
    交互式演示 - 极简版本
    
    所有显示逻辑都由 MainOpAgent 提供，前端只负责输入输出
    """
    # 1. 加载配置
    config_filename = f"default_triton_{backend}_config.yaml"
    config_path = os.path.join(get_project_root(), "config", config_filename)
    
    if not os.path.exists(config_path):
        logger.warning(f"Config file not found: {config_path}")
        config_filename = "default_triton_cuda_config.yaml"
        config_path = os.path.join(get_project_root(), "config", config_filename)
        logger.info(f"Fallback to default config: {config_path}")
    
    config = load_yaml(config_path)
    logger.info(f"✓ Loaded config from: {config_path}")
    
    # 2. 处理设备列表
    if device_ids is None:
        device_ids = [0]
    
    # 3. 显示欢迎信息
    print("=" * 80)
    print("🚀 AI Kernel Assistant - 对话式算子生成")
    print("=" * 80)
    print(f"\n⚙️  配置信息:")
    print(f"   • 配置文件: {config_filename}")
    print(f"   • Backend: {backend}")
    print(f"   • Arch: {arch}")
    print(f"   • Device IDs: {device_ids}")
    print(f"   • DSL: {dsl}")
    print(f"   • Framework: {framework}")
    print("\n💡 提示: 您可以用自然语言描述您的算子需求")
    print("   • 输入 '保存' 保存验证目录和对话历史（保存后继续对话）")
    print("   • 按 Ctrl+C 退出对话")
    print("   • 所有其他输入将由 AI 智能理解您的意图")
    print()
    
    # 4. 注册 Worker
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
    
    # 5. 创建 MainOpAgent
    agent = MainOpAgent(
        config=config,
        framework=framework,
        backend=backend,
        arch=arch,
        dsl=dsl
    )
    logger.info("✓ MainOpAgent initialized")
    
    # 6. 开始对话
    print("\n" + "-" * 80)
    try:
        user_request = input("👤 请输入您的需求: ")
    except KeyboardInterrupt:
        print("\n\n⚠️ 检测到 Ctrl+C，退出程序")
        print("\n" + "=" * 80)
        print("🎉 感谢使用 AI Kernel Assistant!")
        print("=" * 80)
        return
    
    # 第一轮：生成 task 代码
    state = await agent.start_conversation(user_request)
    
    # 显示返回的消息
    if state.get("display_message"):
        print(state["display_message"])
    if state.get("hint_message"):
        print(state["hint_message"])
    
    # 7. 对话循环
    while True:
        try:
            print()
            user_input = input("👤 请输入您的需求: ").strip()
        except KeyboardInterrupt:
            # Ctrl+C 强制退出
            print("\n\n⚠️ 检测到 Ctrl+C，正在取消操作...")
            agent.request_cancellation()
            print("🤖 再见！")
            break
        
        if not user_input:
            print("💡 请输入您的回复")
            continue
        print("\n🤖 正在处理您的需求...")
        state = await agent.continue_conversation(
            current_state=state,
            user_input=user_input,
            action="auto"
        )
        
        current_step = state.get("current_step", "")
        if current_step == "cancelled_by_user":
            print("\n🤖 操作已取消，再见！")
            break
        
        # 显示返回的消息
        if state.get("display_message"):
            print(state["display_message"])
        if state.get("hint_message"):
            print(state["hint_message"])
    
    print("\n" + "=" * 80)
    print("🎉 感谢使用 AI Kernel Assistant!")
    print("=" * 80)


def main():
    """命令行入口"""
    asyncio.run(interactive_demo())


if __name__ == "__main__":
    main()
