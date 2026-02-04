#!/usr/bin/env python3
"""KernelAgent 测试 - 运行: python tests/v2/ut/test_kernel_agent.py"""

import asyncio
import os
import sys
import logging
import json
from datetime import datetime
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

# 创建日志目录
LOG_DIR = Path.home() / ".akg" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# 创建会话日志文件
SESSION_LOG_FILE = LOG_DIR / f"kernel_agent_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"

def log_interaction(round_num: int, interaction_type: str, data: dict):
    """
    记录交互日志到文件
    
    Args:
        round_num: 轮次编号
        interaction_type: 交互类型 (user_input, llm_request, llm_response, tool_call, result)
        data: 数据字典
    """
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "round": round_num,
        "type": interaction_type,
        "data": data
    }
    
    try:
        with open(SESSION_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"⚠️  日志写入失败: {e}")


def print_log_summary():
    """打印日志摘要"""
    try:
        with open(SESSION_LOG_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        print(f"\n📊 [日志摘要]")
        print(f"   总记录数: {len(lines)}")
        
        # 统计各类型数量
        types = {}
        for line in lines:
            try:
                entry = json.loads(line)
                entry_type = entry.get("type", "unknown")
                types[entry_type] = types.get(entry_type, 0) + 1
            except:
                pass
        
        print(f"   记录类型分布:")
        for t, count in sorted(types.items()):
            print(f"      {t}: {count}")
        
    except Exception as e:
        print(f"⚠️  无法读取日志: {e}")

import akg_agents

# 诊断导入问题
def diagnose_imports():
    """诊断可能的导入问题"""
    issues = []
    available = []
    
    # 检查 KernelAgent (必需)
    try:
        from akg_agents.op.agents.kernel_agent import KernelAgent
        available.append("✅ KernelAgent")
    except ImportError as e:
        issues.append(f"❌ KernelAgent 导入失败: {e}")
        return issues, []
    
    # 检查 PlanAgent (core_v2/agents)
    try:
        from akg_agents.core_v2.agents.plan import PlanAgent
        available.append("✅ PlanAgent")
    except ImportError as e:
        issues.append(f"⚠️  PlanAgent 导入失败: {e}")
    
    op_agents = {
        "kernel_gen": "KernelGen",
        "kernel_designer": "KernelDesigner",
        "op_task_builder": "OpTaskBuilder"
    }
    
    for module_name, class_name in op_agents.items():
        try:
            __import__(f"akg_agents.op.agents.{module_name}")
            available.append(f"✅ {class_name}")
        except ImportError as e:
            issues.append(f"⚠️  {class_name} 导入失败: {e}")
    
    return issues, available

from akg_agents.op.agents.kernel_agent import KernelAgent


_current_round = 0

def wrap_agent_with_logging(agent):
    """包装 agent.run_llm 以记录 LLM 交互到日志文件（包含详细上下文）"""
    original_run_llm = agent.run_llm
    
    async def logged_run_llm(template, input_data, model_level="standard"):
        global _current_round
        
        # 生成 prompt
        prompt_text = template.format(**input_data) if hasattr(template, 'format') else str(template)
        
        # 获取 trace 摘要
        trace_summary = agent.get_trace_summary() if hasattr(agent, 'get_trace_summary') else {}
        
        # 记录请求的基本信息
        log_data = {
            "model_level": model_level,
            "prompt_length": len(prompt_text),
            "prompt_full": prompt_text,
            "current_node": getattr(agent, 'current_node_id', 'unknown'),
            "trace_summary": trace_summary
        }
        
        # 尝试获取并记录详细的上下文信息
        try:
            if hasattr(agent, 'trace') and hasattr(agent, 'current_node_id'):
                from akg_agents.core_v2.llm import create_llm_client
                llm_client = create_llm_client(model_level=model_level)
                
                # 获取完整历史（未压缩）
                full_history = agent.trace.get_full_action_history(agent.current_node_id)
                
                # 获取压缩历史
                history_records = await agent.trace.get_compressed_history_for_llm(
                    llm_client, agent.current_node_id, max_tokens=2000
                )
                
                # 记录上下文详情
                log_data["context_details"] = {
                    "full_history_count": len(full_history),
                    "compressed_history_count": len(history_records),
                    "compression_ratio": f"{len(history_records)}/{len(full_history)}" if full_history else "0/0",
                    "is_compressed": len(history_records) < len(full_history),
                    "current_plan_steps": len(getattr(agent, 'plan_list', [])),
                    "original_user_input": getattr(agent, '_original_user_input', 'N/A')[:100]
                }
                
                # 记录压缩历史的详细信息
                log_data["compressed_history"] = []
                for r in history_records:
                    entry = {
                        "action_id": r.action_id,
                        "tool_name": r.tool_name,
                    }
                    
                    # 特殊处理 history_summary
                    if r.tool_name == "history_summary":
                        entry["summary"] = r.result.get("summary", "")[:200] + "..."
                        entry["original_actions"] = r.arguments.get("original_actions", 0)
                        entry["compressed"] = True
                    else:
                        entry["arguments"] = str(r.arguments)[:100]
                        entry["result_status"] = r.result.get("status", "N/A")
                        entry["compressed"] = False
                    
                    log_data["compressed_history"].append(entry)
                
        except Exception as e:
            log_data["context_error"] = str(e)
        
        log_interaction(_current_round, "llm_request", log_data)
        
        # 打印上下文摘要
        print(f"   📤 [LLM Request]")
        print(f"      Prompt: {len(prompt_text)} chars")
        print(f"      Node: {log_data['current_node']}")
        if "context_details" in log_data:
            ctx = log_data["context_details"]
            print(f"      History: {ctx['compression_ratio']} ({'压缩' if ctx['is_compressed'] else '完整'})")
            print(f"      Plan Steps: {ctx['current_plan_steps']}")
        
        # 调用 LLM
        result = await original_run_llm(template, input_data, model_level)
        
        # 记录响应
        if isinstance(result, tuple) and result:
            response = result[0] if len(result) > 0 else ""
            log_interaction(_current_round, "llm_response", {
                "response_length": len(response) if isinstance(response, str) else 0,
                "response_full": response
            })
            print(f"   📥 [LLM Response] {len(response) if isinstance(response, str) else 0} chars")
        
        return result
    
    agent.run_llm = logged_run_llm
    return agent


async def test_kernel_agent():
    """KernelAgent 交互式测试"""
    
    print("\n" + "="*80)
    print("KernelAgent 测试")
    print("="*80 + "\n")
    
    # 诊断导入问题
    print("🔍 [诊断] 检查模块导入...")
    import_issues, available_agents = diagnose_imports()
    
    if available_agents:
        print("\n✅ [可用模块]")
        for agent in available_agents:
            print(f"   {agent}")
    
    if import_issues:
        print("\n📝 [注意事项]")
        for issue in import_issues:
            print(f"   {issue}")
        print()
    else:
        print("   ✅ 所有模块都正常\n")
    
    # 检查 API Key
    from akg_agents.utils.environment_check import _check_llm_api
    if not _check_llm_api():
        raise ValueError("LLM API Key 配置或连接有问题，请检查。")
    
    try:
        # 使用唯一 task_id，避免加载旧历史
        import time
        task_id = f"test_{int(time.time())}"
        
        print("🔧 [初始化] 创建 KernelAgent...")
        print(f"📁 任务ID: {task_id}")
        print(f"📝 日志文件: {SESSION_LOG_FILE}\n")
        
        agent = KernelAgent(task_id=task_id, model_level="standard")
        
        # 包装 agent 以记录 LLM 交互
        agent = wrap_agent_with_logging(agent)
        
        # 打印注册的工具（更详细）
        all_tool_names = [t.get("function", {}).get("name", "Unknown") for t in agent.available_tools]
        agent_tool_names = list(agent.agent_registry.keys())
        basic_tool_names = [t for t in all_tool_names if t not in agent_tool_names]
        
        print(f"\n📦 [已注册工具]")
        print(f"   Total: {len(all_tool_names)} tools")
        print(f"   ├─ Basic Tools ({len(basic_tool_names)}): {basic_tool_names}")
        print(f"   └─ Agent Tools ({len(agent_tool_names)}): {agent_tool_names}")
        
        if len(agent_tool_names) == 0:
            print(f"\n   ⚠️  警告：没有加载任何 Agent Tools！")
            print(f"   这可能是因为 agents 模块没有被导入，请检查 _load_agent_registry()")
        
        # 记录初始化信息
        log_interaction(0, "initialization", {
            "total_tools": len(all_tool_names),
            "basic_tools": basic_tool_names,
            "agent_tools": agent_tool_names
        })
        print()
        
        round_num = 0
        
        while True:
            round_num += 1
            global _current_round
            _current_round = round_num
            
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
            print(f"🚀 [处理中] 用户输入: {user_input[:100]}")
            print("-"*80 + "\n")
            
            # 记录用户输入
            trace_before = agent.get_trace_summary() if hasattr(agent, 'get_trace_summary') else {}
            log_interaction(round_num, "user_input", {
                "input": user_input,
                "current_node": getattr(agent, 'current_node_id', 'unknown'),
                "total_actions_before": trace_before.get('total_actions', 0)
            })
            
            # 执行
            result = await agent.run(user_input)
            
            # 记录新增的工具调用
            trace_after = agent.get_trace_summary() if hasattr(agent, 'get_trace_summary') else {}
            try:
                full_history = agent.trace.get_full_action_history(agent.current_node_id)
                new_actions = full_history[trace_before.get('total_actions', 0):]
                for action in new_actions:
                    log_interaction(round_num, "tool_call", {
                        "tool_name": action.tool_name,
                        "arguments": action.arguments,
                        "result": action.result
                    })
            except:
                pass  # 忽略记录失败
            
            # 记录执行结果
            log_interaction(round_num, "result", {
                "status": result.get('status'),
                "output": result.get('output', ''),
                "error": result.get('error_information', ''),
                "current_node": result.get('current_node', 'unknown'),
                "total_actions_after": trace_after.get('total_actions', 0)
            })
            
            # 显示结果（更详细）
            print("\n" + "="*80)
            print(f"📊 [执行结果]")
            print("-"*80)
            print(f"   状态: {result.get('status')}")
            print(f"   输出: {result.get('output')[:200] if result.get('output') else 'N/A'}")
            if len(result.get('output', '')) > 200:
                print(f"        (输出过长，已截断，共 {len(result.get('output'))} 字符)")
            
            if result.get('error_information'):
                print(f"   ❌ 错误: {result.get('error_information')}")
            
            # 显示计划（更详细）
            if result.get('plan_list'):
                print(f"\n📋 [执行计划] {len(result['plan_list'])} 个步骤:")
                for step in result['plan_list']:
                    emoji = {"pending": "⏳", "success": "✅", "failed": "❌"}.get(step.get("status"), "❓")
                    desc = step.get('desc') or step.get('description') or step.get('tool', 'Unknown')
                    step_id = step.get('step_id', '?')
                    print(f"   {emoji} Step {step_id}: {desc[:80]}")
                    if step.get('retry_count', 0) > 0:
                        print(f"       └─ 重试次数: {step.get('retry_count')}/{step.get('max_retries', 3)}")
            
            # 显示历史（最近5个）
            if result.get('history'):
                print(f"\n📜 [执行历史] 共 {len(result['history'])} 个动作:")
                recent = result['history'][-5:]
                for i, r in enumerate(recent, len(result['history'])-len(recent)+1):
                    tool = r.get('tool_name', 'Unknown')
                    # 特殊处理 compressed summary
                    if r.get('compressed'):
                        print(f"   [{i}] {tool} [压缩: {r.get('original_actions', 0)} → 1]")
                    else:
                        status = r.get('result', {}).get('status', 'N/A')
                        print(f"   [{i}] {tool} → {status}")
            
            print("="*80)
            
            # 如果需要用户响应
            if result.get('status') == 'waiting_for_user':
                print(f"\n💬 [Agent 询问]")
                print("-"*80)
                print(result.get('message'))
                print("-"*80)
                continue
            
            # 如果完成或出错
            if result.get('status') in ['success', 'error']:
                status_emoji = "✅" if result.get('status') == 'success' else "❌"
                print(f"\n{status_emoji} 任务 {result.get('status').upper()}!")
                if result.get('status') == 'success':
                    print(f"   总共执行了 {result.get('total_actions', 0)} 个动作")
                continue
        
        # 打印统计信息
        print("\n" + "="*80)
        print(f"🎉 测试结束")
        print("-"*80)
        print(f"   总轮次: {round_num}")
        
        if hasattr(agent, 'get_trace_summary'):
            summary = agent.get_trace_summary()
            print(f"   当前节点: {summary.get('current_node', 'unknown')}")
            print(f"   总动作数: {summary.get('total_actions', 0)}")
            print(f"   路径长度: {summary.get('path_length', 0)}")
            
            # 打印历史压缩统计
            try:
                from akg_agents.core_v2.llm import create_llm_client
                llm_client = create_llm_client(model_level=agent.model_level or "standard")
                
                full_history = agent.trace.get_full_action_history(agent.current_node_id)
                compressed_history = await agent.trace.get_compressed_history_for_llm(
                    llm_client, agent.current_node_id, max_tokens=2000
                )
                
                print(f"\n   📊 历史压缩统计:")
                print(f"      完整历史: {len(full_history)} 个动作")
                print(f"      压缩后: {len(compressed_history)} 个记录")
                
                if len(compressed_history) < len(full_history):
                    compression_ratio = (1 - len(compressed_history) / len(full_history)) * 100
                    print(f"      压缩率: {compression_ratio:.1f}%")
                    
                    # 检查是否有 summary
                    has_summary = any(r.tool_name == "history_summary" for r in compressed_history)
                    if has_summary:
                        summary_action = next(r for r in compressed_history if r.tool_name == "history_summary")
                        original_count = summary_action.arguments.get("original_actions", 0)
                        print(f"      摘要: {original_count} 个动作 → 1 个摘要")
                else:
                    print(f"      未压缩 (历史较短)")
            except Exception as e:
                print(f"   ⚠️  无法获取压缩统计: {e}")
        
        print("="*80)
        
        # 打印日志信息
        print_log_summary()
        print(f"\n📝 日志文件: {SESSION_LOG_FILE}")
        print(f"   查看命令: python tests/v2/ut/view_test_logs.py --latest")
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
