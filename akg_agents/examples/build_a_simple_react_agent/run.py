#!/usr/bin/env python3
"""
SimpleReActAgent 运行示例

使用方法:
    cd akg/aikg && source env.sh
    python examples/build_a_simple_react_agent/run.py

功能:
    - 交互式对话
    - 调用工具（计算、时间、天气、知识搜索）
    - 无 plan 功能，直接响应

示例对话:
    > 现在几点了？
    > 计算 123 * 456 + 789
    > 北京今天天气怎么样？
    > 什么是机器学习？
    > 你好，介绍一下你自己
"""

import asyncio
import sys
import logging
import time
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

# 减少不必要的日志
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("akg_agents.core_v2").setLevel(logging.WARNING)


async def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("  SimpleReActAgent - 简单对话 Agent 演示")
    print("=" * 60)
    
    # 检查环境
    try:
        from akg_agents.utils.environment_check import _check_llm_api
        print("\n🔍 检查 LLM API 配置...")
        if not _check_llm_api():
            print("❌ LLM API 配置有问题，请检查 settings.json")
            return 1
        print("✅ LLM API 配置正常\n")
    except Exception as e:
        print(f"⚠️ 无法检查 LLM API: {e}")
    
    # 导入 Agent
    from simple_react_agent import SimpleReActAgent
    
    # 创建 Agent
    task_id = f"simple_{int(time.time())}"
    print(f"🤖 创建 SimpleReActAgent (task_id: {task_id})")
    
    agent = SimpleReActAgent(
        task_id=task_id,
        model_level="standard"
    )
    
    # 显示可用工具
    print("\n📦 可用工具:")
    for tool in agent.available_tools:
        func = tool.get("function", {})
        print(f"   - {func.get('name')}: {func.get('description', '')[:50]}...")
    
    print("\n" + "-" * 60)
    print("开始对话！输入 'quit' 或 'exit' 退出")
    print("-" * 60 + "\n")
    
    # 对话循环
    round_num = 0
    
    while True:
        try:
            # 获取用户输入
            user_input = input("👤 你: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'q', '退出']:
                print("\n👋 再见！")
                break
            
            round_num += 1
            print()
            
            # 调用 Agent
            result = await agent.run(user_input)
            
            # 显示结果
            status = result.get("status")
            
            if status == "success":
                message = result.get("message") or result.get("output", "")
                print(f"🤖 助手: {message}")
            elif status == "waiting_for_user":
                message = result.get("message", "")
                print(f"🤖 助手: {message}")
            elif status == "error":
                error = result.get("error_information", "未知错误")
                print(f"❌ 错误: {error}")
            
            # 显示工具调用历史（如果有）
            history = result.get("history", [])
            tool_calls = [h for h in history if h.get("tool_name") not in ["ask_user", "history_summary"]]
            
            if tool_calls:
                print(f"\n   📋 本轮使用的工具: ", end="")
                tool_names = [h.get("tool_name") for h in tool_calls]
                print(", ".join(tool_names))
            
            print()
            
        except KeyboardInterrupt:
            print("\n\n👋 用户中断，再见！")
            break
        except EOFError:
            print("\n\n👋 再见！")
            break
        except Exception as e:
            print(f"\n❌ 发生错误: {e}")
            import traceback
            traceback.print_exc()
    
    # 统计
    print("\n" + "=" * 60)
    print(f"📊 对话统计: {round_num} 轮对话")
    
    if hasattr(agent, 'get_trace_summary'):
        summary = agent.get_trace_summary()
        print(f"   总动作数: {summary.get('total_actions', 0)}")
    
    print("=" * 60 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
