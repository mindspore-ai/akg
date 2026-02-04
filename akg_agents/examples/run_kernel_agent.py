#!/usr/bin/env python3
"""
KernelAgent 示例 - 展示如何使用 KernelAgent 进行 kernel 开发

运行方式：
  1. 交互模式: python examples/run_kernel_agent.py --interactive
  2. 单次执行: python examples/run_kernel_agent.py --query "你的需求描述"
  3. 自定义配置: python examples/run_kernel_agent.py --query "需求" --framework torch --backend cpu

示例：
  # 基础使用
  python examples/run_kernel_agent.py --query "实现一个 ReLU kernel"
  
  # 指定框架和后端
  python examples/run_kernel_agent.py --query "优化矩阵乘法" --framework torch --backend cpu --arch x86_64 --dsl cpp
  
  # 交互模式
  python examples/run_kernel_agent.py --interactive --backend cuda --arch a100
"""

import asyncio
import argparse
import sys
import logging
import json
import time
from datetime import datetime
from pathlib import Path


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)


class KernelAgentExample:
    """KernelAgent 示例封装类"""
    
    def __init__(self, task_id: str = None, model_level: str = "standard", 
                 framework: str = "torch", backend: str = "cuda", 
                 arch: str = "a100", dsl: str = "triton",
                 devices: str = "0", enable_logging: bool = True):
        """
        初始化示例
        
        Args:
            task_id: 任务ID，默认自动生成
            model_level: 模型级别 (fast/standard/advanced)
            framework: 框架类型 (torch/mindspore/numpy等)
            backend: 后端类型 (cuda/cpu/npu等)
            arch: 架构类型 (a100/x86_64/ascend910等)
            dsl: DSL 语言类型 (triton/cpp/python等)
            devices: 设备ID列表，逗号分隔 (如 "0" 或 "0,1,2")
            enable_logging: 是否启用详细日志
        """
        self.task_id = task_id or f"kernel_task_{int(time.time())}"
        self.model_level = model_level
        self.framework = framework
        self.backend = backend
        self.arch = arch
        self.dsl = dsl
        self.devices = devices
        self.enable_logging = enable_logging
        self.agent = None
        self.round_num = 0
        
        # 创建日志目录
        if enable_logging:
            self.log_dir = Path.home() / ".akg" / "logs"
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.log_file = self.log_dir / f"kernel_agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        else:
            self.log_file = None
    
    def _log_interaction(self, interaction_type: str, data: dict):
        """记录交互日志"""
        if not self.enable_logging or not self.log_file:
            return
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "round": self.round_num,
            "type": interaction_type,
            "data": data
        }
        
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"⚠️  日志写入失败: {e}")
    
    def _wrap_agent_with_logging(self, agent):
        """包装 agent 以记录 LLM 交互"""
        if not self.enable_logging:
            return agent
        
        original_run_llm = agent.run_llm
        
        async def logged_run_llm(template, input_data, model_level="standard"):
            # 生成 prompt
            prompt_text = template.format(**input_data) if hasattr(template, 'format') else str(template)
            
            # 记录请求
            log_data = {
                "model_level": model_level,
                "prompt_length": len(prompt_text),
                "current_node": getattr(agent, 'current_node_id', 'unknown'),
            }
            self._log_interaction("llm_request", log_data)
            
            print(f"   📤 [LLM Request] Prompt: {len(prompt_text)} chars")
            
            # 调用 LLM
            result = await original_run_llm(template, input_data, model_level)
            
            # 记录响应
            if isinstance(result, tuple) and result:
                response = result[0] if len(result) > 0 else ""
                self._log_interaction("llm_response", {
                    "response_length": len(response) if isinstance(response, str) else 0
                })
                print(f"   📥 [LLM Response] {len(response) if isinstance(response, str) else 0} chars")
            
            return result
        
        agent.run_llm = logged_run_llm
        return agent
    
    async def initialize(self):
        """初始化 KernelAgent"""
        print("\n" + "="*80)
        print("KernelAgent 示例")
        print("="*80 + "\n")
        
        # 检查环境
        print("🔍 [检查] 验证环境配置...")
        from akg_agents.utils.environment_check import _check_llm_api
        if not _check_llm_api():
            raise ValueError("LLM API Key 配置或连接有问题，请检查。")
        
        # 创建 agent
        print(f"🔧 [初始化] 创建 KernelAgent...")
        print(f"   任务ID: {self.task_id}")
        print(f"   模型级别: {self.model_level}")
        print(f"   框架: {self.framework}")
        print(f"   后端: {self.backend}")
        print(f"   架构: {self.arch}")
        print(f"   DSL: {self.dsl}")
        print(f"   设备: {self.devices}")
        if self.log_file:
            print(f"   日志文件: {self.log_file}")
        
        from akg_agents.core_v2.agents.kernel_agent import KernelAgent
        self.agent = KernelAgent(
            task_id=self.task_id, 
            model_level=self.model_level,
            framework=self.framework,
            backend=self.backend,
            arch=self.arch,
            dsl=self.dsl
        )
        
        # 包装 agent
        self.agent = self._wrap_agent_with_logging(self.agent)
        
        # 显示可用工具
        all_tool_names = [t.get("function", {}).get("name", "Unknown") for t in self.agent.available_tools]
        agent_tool_names = list(self.agent.agent_registry.keys())
        basic_tool_names = [t for t in all_tool_names if t not in agent_tool_names]
        
        print(f"\n📦 [已注册工具]")
        print(f"   Total: {len(all_tool_names)} tools")
        print(f"   ├─ Basic Tools ({len(basic_tool_names)}): {basic_tool_names[:5]}")
        if len(basic_tool_names) > 5:
            print(f"      └─ ...and {len(basic_tool_names) - 5} more")
        print(f"   └─ Agent Tools ({len(agent_tool_names)}): {agent_tool_names}")
        
        self._log_interaction("initialization", {
            "total_tools": len(all_tool_names),
            "basic_tools": basic_tool_names,
            "agent_tools": agent_tool_names
        })
        
        print("\n✅ 初始化完成\n")
    
    async def run_single(self, query: str):
        """
        单次运行模式
        
        Args:
            query: 用户需求描述
        """
        if not self.agent:
            await self.initialize()
        
        self.round_num = 1
        
        print("="*80)
        print(f"第 {self.round_num} 轮")
        print("="*80)
        print(f"\n🚀 [处理中] 用户输入: {query}\n")
        
        # 记录用户输入
        self._log_interaction("user_input", {
            "input": query,
            "current_node": getattr(self.agent, 'current_node_id', 'unknown')
        })
        
        # 执行
        result = await self.agent.run(query)
        
        # 显示结果
        self._display_result(result)
        
        return result
    
    async def run_interactive(self):
        """交互式运行模式"""
        if not self.agent:
            await self.initialize()
        
        while True:
            self.round_num += 1
            
            print("\n" + "="*80)
            print(f"第 {self.round_num} 轮")
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
                self.round_num -= 1
                continue
            
            print("\n" + "-"*80)
            print(f"🚀 [处理中] 用户输入: {user_input[:100]}")
            print("-"*80 + "\n")
            
            # 记录用户输入
            self._log_interaction("user_input", {
                "input": user_input,
                "current_node": getattr(self.agent, 'current_node_id', 'unknown')
            })
            
            # 执行
            result = await self.agent.run(user_input)
            
            # 显示结果
            self._display_result(result)
            
            # 处理等待用户响应的情况
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
        
        # 显示统计信息
        self._display_summary()
    
    def _display_result(self, result: dict):
        """显示执行结果"""
        print("\n" + "="*80)
        print(f"📊 [执行结果]")
        print("-"*80)
        print(f"   状态: {result.get('status')}")
        
        output = result.get('output', '')
        if output:
            print(f"   输出: {output[:200]}")
            if len(output) > 200:
                print(f"        (输出过长，已截断，共 {len(output)} 字符)")
        else:
            print(f"   输出: N/A")
        
        if result.get('error_information'):
            print(f"   ❌ 错误: {result.get('error_information')}")
        
        # 显示计划
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
                if r.get('compressed'):
                    print(f"   [{i}] {tool} [压缩: {r.get('original_actions', 0)} → 1]")
                else:
                    status = r.get('result', {}).get('status', 'N/A')
                    print(f"   [{i}] {tool} → {status}")
        
        print("="*80)
        
        # 记录结果
        self._log_interaction("result", {
            "status": result.get('status'),
            "output": output,
            "error": result.get('error_information', ''),
            "current_node": result.get('current_node', 'unknown')
        })
    
    def _display_summary(self):
        """显示统计摘要"""
        print("\n" + "="*80)
        print(f"🎉 运行结束")
        print("-"*80)
        print(f"   总轮次: {self.round_num}")
        
        if hasattr(self.agent, 'get_trace_summary'):
            summary = self.agent.get_trace_summary()
            print(f"   当前节点: {summary.get('current_node', 'unknown')}")
            print(f"   总动作数: {summary.get('total_actions', 0)}")
            print(f"   路径长度: {summary.get('path_length', 0)}")
        
        if self.log_file:
            print(f"\n📝 日志文件: {self.log_file}")
        
        print("="*80)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="KernelAgent 示例 - 展示如何使用 KernelAgent 进行 kernel 开发",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 交互模式
  python examples/run_kernel_agent.py --interactive
  
  # 单次执行
  python examples/run_kernel_agent.py --query "实现一个 ReLU kernel"
  
  # 指定框架和后端
  python examples/run_kernel_agent.py --query "优化矩阵乘法" --framework torch --backend cpu --arch x86_64 --dsl cpp
  
  # 完整配置示例
  python examples/run_kernel_agent.py --query "实现 softmax" --framework torch --backend cpu --arch x86_64 --dsl cpp --devices 0 --model-level fast
  
  # 禁用日志
  python examples/run_kernel_agent.py --query "实现卷积" --no-logging
        """
    )
    
    # 运行模式
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='交互式模式，可以连续输入多个需求'
    )
    mode_group.add_argument(
        '--query', '-q',
        type=str,
        help='用户需求描述（单次执行模式）'
    )
    
    # Kernel 配置参数
    parser.add_argument(
        '--framework',
        type=str,
        default='torch',
        help='框架类型：torch, mindspore, numpy 等 (默认: torch)'
    )
    parser.add_argument(
        '--backend',
        type=str,
        default='cuda',
        help='后端类型：cuda, cpu, npu 等 (默认: cuda)'
    )
    parser.add_argument(
        '--arch',
        type=str,
        default='a100',
        help='架构类型：a100, x86_64, ascend910 等 (默认: a100)'
    )
    parser.add_argument(
        '--dsl',
        type=str,
        default='triton',
        help='DSL 语言类型：triton, cpp, python 等 (默认: triton)'
    )
    parser.add_argument(
        '--devices',
        type=str,
        default='0',
        help='设备 ID 列表，逗号分隔，如 0 或 0,1,2 (默认: 0)'
    )
    
    # Agent 配置参数
    parser.add_argument(
        '--task-id',
        type=str,
        default=None,
        help='任务 ID，默认自动生成时间戳 ID'
    )
    parser.add_argument(
        '--model-level',
        type=str,
        default='standard',
        choices=['fast', 'standard', 'advanced'],
        help='LLM 模型级别 (默认: standard)'
    )
    parser.add_argument(
        '--no-logging',
        action='store_true',
        help='禁用详细日志记录'
    )
    
    return parser.parse_args()


async def main():
    """主函数"""
    args = parse_args()
    
    try:
        # 创建示例实例
        example = KernelAgentExample(
            task_id=args.task_id,
            model_level=args.model_level,
            framework=args.framework,
            backend=args.backend,
            arch=args.arch,
            dsl=args.dsl,
            devices=args.devices,
            enable_logging=not args.no_logging
        )
        
        # 根据模式运行
        if args.interactive:
            await example.run_interactive()
        else:
            result = await example.run_single(args.query)
            
            # 返回退出码
            if result.get('status') == 'success':
                return 0
            elif result.get('status') == 'error':
                return 1
            else:
                return 2
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断")
        return 130
    except Exception as e:
        print(f"\n❌ [错误] {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
