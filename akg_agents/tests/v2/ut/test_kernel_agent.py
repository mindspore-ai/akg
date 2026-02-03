#!/usr/bin/env python3
"""KernelAgent 测试 - 运行: python tests/v2/ut/test_kernel_agent.py"""

import asyncio
import os
import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

import akg_agents
from akg_agents.core_v2.agents.kernel_agent import KernelAgent


async def test_kernel_agent():
    """KernelAgent 交互式测试"""
    
    print("\n" + "="*80)
    print("KernelAgent 测试")
    print("="*80 + "\n")
    
    # 检查 API Key
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("AIKG_API_KEY")
    if not api_key:
        print("[ERROR] 未找到 API Key，请设置环境变量: OPENAI_API_KEY 或 AIKG_API_KEY")
        return False
    
    print(f"[OK] API Key: {api_key[:10]}...\n")
    
    try:
        agent = KernelAgent(task_id="test_kernel_agent", model_level="standard")
        round_num = 0
        
        while True:
            round_num += 1
            print("\n" + "="*80)
            print(f"第 {round_num} 轮")
            print("="*80)
            
            # 获取用户输入
            try:
                user_input = input("\n请输入需求（quit 退出）: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\n用户中断")
                break
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if not user_input:
                round_num -= 1
                continue
            
            print("\n" + "-"*80)
            print("处理中...")
            print("-"*80 + "\n")
            
            result = await agent.run(user_input)
            
            # 显示结果
            print("\n" + "-"*80)
            print(f"状态: {result.get('status')}")
            print(f"输出: {result.get('output')}")
            
            if result.get('error_information'):
                print(f"错误: {result.get('error_information')}")
            
            # 显示计划
            if result.get('plan_list'):
                print(f"\n计划: {len(result['plan_list'])} 步")
                for step in result['plan_list']:
                    emoji = {"pending": "⏳", "success": "✅", "failed": "❌"}.get(step.get("status"), "❓")
                    desc = step.get('desc') or step.get('tool', 'Unknown')
                    print(f"  {emoji} {step.get('step_id')}: {desc}")
            
            # 显示历史
            if result.get('history'):
                print(f"\n历史: {len(result['history'])} 个动作")
                for i, r in enumerate(result['history'][-3:], 1):
                    print(f"  [{i}] {r.get('tool_name')} - {r.get('result', {}).get('status')}")
            
            print("-"*80)
            
            # 如果需要用户响应
            if result.get('status') == 'waiting_for_user':
                print(f"\n💬 {result.get('message')}")
                continue
            
            # 如果完成或出错
            if result.get('status') in ['success', 'error']:
                print(f"\n任务{result.get('status')}！")
                continue
        
        print("\n" + "="*80)
        print(f"测试结束，共 {round_num} 轮")
        print("="*80)
        return True
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_kernel_agent())
    sys.exit(0 if success else 1)
