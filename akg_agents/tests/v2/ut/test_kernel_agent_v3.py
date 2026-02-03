#!/usr/bin/env python3
"""
KernelAgentV3 测试（迭代式执行）

运行方式：
python tests/v2/ut/test_kernel_agent_v3.py
"""

import asyncio
import os
import sys
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

from akg_agents.core_v2.agents.kernel_agent_v3 import KernelAgentV3


async def test_kernel_agent_v3():
    """测试 KernelAgentV3（迭代式执行，支持多轮交互）"""
    
    print("\n" + "="*80)
    print("KernelAgentV3 交互式测试")
    print("="*80 + "\n")
    
    # 检查 API Key
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("AIKG_API_KEY")
    if not api_key:
        print("[ERROR] 未找到 API Key")
        print("请设置环境变量:")
        print("  export OPENAI_API_KEY='your-api-key'")
        print("  或")
        print("  export AIKG_API_KEY='your-api-key'")
        return False
    
    print(f"[OK] 检测到 API Key: {api_key[:10]}...\n")
    
    try:
        # 初始化 Agent
        agent = KernelAgentV3(
            task_id="test_v3",
            model_level="standard"
        )
        
        round_num = 0
        while True:
            round_num += 1
            
            print("\n" + "="*80)
            print(f"第 {round_num} 轮交互")
            print("="*80)
            
            # 获取用户输入
            print("\n请输入您的需求（输入 'quit' 或 'exit' 退出）:")
            try:
                user_input = input("👤 您的需求: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\n[INFO] 用户中断，退出测试")
                break
            
            if user_input.lower() in ['quit', 'exit', 'q', '退出']:
                print("\n[INFO] 用户选择退出")
                break
            
            if not user_input:
                print("[WARNING] 输入为空，请重新输入")
                round_num -= 1
                continue
            
            # 执行任务
            print("\n" + "-"*80)
            print("开始处理需求（迭代式执行）...")
            print("-"*80 + "\n")
            
            result = await agent.run(user_input)
            
            # 显示结果
            print("\n" + "-"*80)
            print("本轮执行结果:")
            print("-"*80)
            print(f"状态: {result.get('status')}")
            print(f"输出: {result.get('output')}")
            
            if result.get('error_information'):
                print(f"错误: {result.get('error_information')}")
            
            # 显示当前计划
            if result.get('plan_list'):
                print(f"\n当前执行计划: {len(result['plan_list'])} 个步骤")
                for step in result['plan_list']:
                    status_emoji = {
                        "pending": "⏳",
                        "running": "🔄",
                        "success": "✅",
                        "failed": "❌"
                    }.get(step.get("status", "pending"), "❓")
                    
                    print(f"  {status_emoji} Step {step.get('step_id')}: {step.get('tool')} - {step.get('description', '')} [{step.get('status', 'pending')}]")
                    if step.get("retry_count", 0) > 0:
                        print(f"     重试次数: {step.get('retry_count')}/{step.get('max_retries', 3)}")
            
            # 显示执行历史摘要
            if result.get('history'):
                print(f"\n执行历史: 共 {len(result['history'])} 个动作")
                for i, record in enumerate(result['history'][-5:], 1):  # 只显示最后 5 个
                    print(f"  [{i}] {record.get('tool_name')} - {record.get('result', {}).get('status', 'N/A')}")
            
            print("-"*80)
            
            # 如果需要用户响应，继续循环
            if result.get('status') == 'waiting_for_user':
                print(f"\n💬 Agent 询问: {result.get('message')}")
                continue
            
            # 如果完成或出错，询问是否继续
            if result.get('status') in ['success', 'error', 'timeout']:
                print(f"\n任务{result.get('status')}！是否继续新任务？(y/n)")
                choice = input(">>> ").strip().lower()
                if choice != 'y':
                    break
        
        print("\n" + "="*80)
        print("测试结束")
        print("="*80)
        print(f"总轮次: {round_num}")
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_simple_case():
    """简单测试案例：生成一个 ReLU 算子"""
    
    print("\n" + "="*80)
    print("KernelAgentV3 简单测试案例")
    print("="*80 + "\n")
    
    agent = KernelAgentV3(
        task_id="test_simple_relu",
        model_level="standard"
    )
    
    # 第一轮：提出需求
    print("[Round 1] 用户需求: 生成一个 ReLU 算子\n")
    result1 = await agent.run("生成一个 ReLU 算子")
    
    print(f"状态: {result1.get('status')}")
    print(f"输出: {result1.get('output')}\n")
    
    if result1.get('status') == 'waiting_for_user':
        print(f"Agent 询问: {result1.get('message')}\n")
        
        # 第二轮：回答问题
        print("[Round 2] 用户回答: 使用 triton_cuda，默认配置\n")
        result2 = await agent.run("使用 triton_cuda，默认配置")
        
        print(f"状态: {result2.get('status')}")
        print(f"输出: {result2.get('output')}\n")
        
        # 显示最终计划
        if result2.get('plan_list'):
            print("最终执行计划:")
            for step in result2['plan_list']:
                print(f"  - {step.get('tool')}: {step.get('status')}")
    
    print("\n测试完成！")


if __name__ == "__main__":
    print("\n💡 提示：")
    print("   - 本测试演示 KernelAgentV3 的迭代式执行流程")
    print("   - Agent 会自动生成计划、执行工具、处理失败重试")
    print("   - 支持多轮用户交互\n")
    
    print("请选择测试模式：")
    print("  1. 交互式测试（推荐）")
    print("  2. 简单测试案例")
    print("  3. 退出")
    
    choice = input("\n请输入选项 (1/2/3): ").strip()
    
    if choice == "1":
        success = asyncio.run(test_kernel_agent_v3())
    elif choice == "2":
        success = asyncio.run(test_simple_case())
    else:
        print("退出测试")
        sys.exit(0)
    
    sys.exit(0 if success else 1)
