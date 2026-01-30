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

"""OpTaskBuilder CLI Tool: 将用户想法转换为KernelBench格式的交互式工具

此工具允许用户输入一个想法（自然语言描述），然后通过多轮交互
逐步构建为正确的KernelBench格式输入。
"""

import asyncio
import os
import sys
import argparse
from typing import Dict, Any, Optional
import json


def create_mock_config():
    """创建模拟配置，实际使用时应从配置文件加载"""
    return {
        "agent_model_config": {
            "op_task_builder": "default",
            "coder": "default",
        },
        "log_dir": "/tmp/akg_agents_test",
        "op_task_builder_max_iterations": 5,
    }


def print_welcome():
    """打印欢迎信息"""
    print("=" * 60)
    print("AKG Agents - OpTaskBuilder Tool")
    print("=" * 60)
    print("这是一个交互式工具，帮助您将自然语言描述转换为KernelBench格式。")
    print("\n使用指南：")
    print("1. 输入您的需求想法（例如：'实现ReLU算子，输入16x16384张量'）")
    print("2. 系统将根据您的需求生成KernelBench格式的代码")
    print("3. 如果需求不明确，会提示您补充信息")
    print("4. 支持多轮交互，直到生成满足要求的代码")
    print("=" * 60)


def print_status(result: Dict[str, Any]):
    """打印当前状态"""
    status = result.get("status", "unknown")
    agent_message = result.get("agent_message", "")
    
    print(f"\n状态: {status}")
    print(f"系统消息: {agent_message}")


def get_user_input(prompt: str = "请输入您的需求: ") -> str:
    """获取用户输入"""
    try:
        user_input = input(prompt).strip()
        return user_input
    except (EOFError, KeyboardInterrupt):
        print("\n\n程序已退出。")
        sys.exit(0)


async def run_op_task_builder_simulation(user_input: str,
                                         user_feedback: Optional[str] = None,
                                         previous_state: Optional[Dict[str, Any]] = None,
                                         framework: str = "torch",
                                         backend: str = "cuda",
                                         arch: str = "a100") -> Dict[str, Any]:
    """运行OpTaskBuilder流程

    Note: 当前实现为模拟版本，当LLM API可用时，可以替换为真实的OpTaskBuilderWorkflow调用
    """
    print(f"处理需求: {user_input}")

    # 如果有可用的LLM API，这里应该调用真实的OpTaskBuilderWorkflow
    # 示例（当前注释）:
    # from akg_agents.op.workflows.op_task_builder_workflow import run_op_task_builder
    # result = await run_op_task_builder(
    #     user_input=user_input,
    #     config=config,  # 从上下文获取配置
    #     user_feedback=user_feedback,
    #     framework=framework,
    #     backend=backend,
    #     arch=arch
    # )
    # return result

    if previous_state is None:
        iteration = 0
        conversation_history = []
    else:
        iteration = previous_state.get("iteration", 0) + 1
        conversation_history = previous_state.get("conversation_history", [])

    # 记录用户输入
    user_content = user_input
    if user_feedback:
        user_content += f"\n用户补充: {user_feedback}"

    conversation_history.append({
        "role": "user",
        "content": user_content
    })

    # 尝试根据用户输入判断需求类型
    user_lower = user_input.lower()

    # 确定算子名称
    op_name = "unknown"
    if "relu" in user_lower:
        op_name = "relu"
    elif "matmul" in user_lower or "matrix" in user_lower and "mul" in user_lower:
        op_name = "matmul"
    elif "add" in user_lower or "plus" in user_lower:
        op_name = "add"
    elif "softmax" in user_lower:
        op_name = "softmax"
    elif "layernorm" in user_lower or "layer norm" in user_lower:
        op_name = "layernorm"
    elif "gelu" in user_lower:
        op_name = "gelu"

    # 检查是否包含形状信息
    has_shape_info = any(s in user_input for s in ["x", "*", "shape", "[", "]", "batch", "dim"])

    if op_name != "unknown":
        if has_shape_info or iteration > 0:
            # 如果已有形状信息或不是第一轮，生成KernelBench格式代码
            task_desc = generate_kernelbench_template(op_name, user_input)
            return {
                "status": "ready",
                "op_name": op_name,
                "agent_message": f"已生成{op_name}算子的KernelBench格式代码",
                "generated_task_desc": task_desc,
                "iteration": iteration,
                "conversation_history": conversation_history
            }
        else:
            # 需要形状信息
            return {
                "status": "need_clarification",
                "op_name": op_name,
                "agent_message": f"检测到{op_name}算子需求，但缺少输入形状信息。",
                "clarification_question": f"请提供{op_name}算子的输入张量形状，例如：'输入是[16, 16384]的float32张量'",
                "iteration": iteration,
                "conversation_history": conversation_history
            }
    else:
        # 无法识别算子类型
        if iteration < 2:
            return {
                "status": "need_clarification",
                "op_name": "unknown",
                "agent_message": "请明确您需要实现什么算子（如：ReLU、MatMul、Add等），并提供输入输出形状。",
                "clarification_question": "请明确需要实现的算子类型和输入形状",
                "iteration": iteration,
                "conversation_history": conversation_history
            }
        else:
            return {
                "status": "unsupported",
                "op_name": "unknown",
                "agent_message": "当前系统无法处理此类型的需求。AIKG主要用于生成高性能算子kernel代码。请提供具体的算子类型（如：ReLU, MatMul, Add等）。",
                "iteration": iteration,
                "conversation_history": conversation_history
            }


def generate_kernelbench_template(op_name: str, user_input: str) -> str:
    """根据算子名称生成KernelBench模板"""
    # 这里是模拟生成的代码，实际实现会连接LLM
    if op_name == "relu":
        return '''import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        return torch.relu(x)

batch_size = 16
dim = 16384

def get_inputs():
    return [torch.randn(batch_size, dim)]

def get_init_inputs():
    return []
'''
    elif op_name == "matmul":
        return '''import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y):
        return torch.matmul(x, y)

batch_size = 1
dim_m = 1024
dim_n = 1024
dim_k = 1024

def get_inputs():
    return [torch.randn(batch_size, dim_m, dim_k), torch.randn(batch_size, dim_k, dim_n)]

def get_init_inputs():
    return []
'''
    elif op_name == "add":
        return '''import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y):
        return x + y

batch_size = 16
dim = 16384

def get_inputs():
    return [torch.randn(batch_size, dim), torch.randn(batch_size, dim)]

def get_init_inputs():
    return []
'''
    elif op_name == "softmax":
        return '''import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        return torch.softmax(x, dim=-1)

batch_size = 16
classes = 1000

def get_inputs():
    return [torch.randn(batch_size, classes)]

def get_init_inputs():
    return []
'''
    else:
        # 默认返回一个通用模板
        return f'''import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        # TODO: 实现 {op_name} 算子逻辑
        return x

# TODO: 根据需求设置合适的参数
batch_size = 16
dim = 1024

def get_inputs():
    return [torch.randn(batch_size, dim)]

def get_init_inputs():
    return []
'''


def save_task_desc_to_file(task_desc: str, op_name: str = "task"):
    """保存生成的task_desc到文件"""
    filename = f"{op_name}_task_desc.py"
    counter = 1
    
    # 避免覆盖已存在的文件
    while os.path.exists(filename):
        filename = f"{op_name}_task_desc_{counter}.py"
        counter += 1
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(task_desc)
    
    print(f"\n生成的KernelBench格式代码已保存到: {filename}")
    return filename


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='AIKG OpTaskBuilder Tool')
    parser.add_argument('--framework', type=str, default='torch', help='目标框架 (default: torch)')
    parser.add_argument('--backend', type=str, default='cuda', help='目标后端 (default: cuda)')
    parser.add_argument('--arch', type=str, default='a100', help='目标架构 (default: a100)')
    
    args = parser.parse_args()
    
    print_welcome()
    
    # 创建配置
    config = create_mock_config()
    
    # 初始化状态
    state = None
    current_input = ""
    feedback = None
    max_iterations = config.get("op_task_builder_max_iterations", 5)
    
    print(f"\n开始交互，最多 {max_iterations} 轮对话...")
    
    for i in range(max_iterations):
        print(f"\n--- 第 {i+1} 轮 ---")
        
        if i == 0:
            # 第一轮，获取初始用户需求
            current_input = get_user_input("请输入您的需求想法: ")
        else:
            # 从上一轮结果获取系统消息后，再要求用户输入
            if state and state.get("status") in ["need_clarification", "need_modification"]:
                clarification_question = state.get("clarification_question", "请提供更多信息：")
                print(f"系统: {clarification_question}")
                feedback = get_user_input("请补充信息: ")
        
        # 运行一轮op_task_builder
        result = await run_op_task_builder_simulation(
            user_input=current_input,
            user_feedback=feedback,
            previous_state=state,
            framework=args.framework,
            backend=args.backend,
            arch=args.arch
        )
        
        # 更新状态
        state = result
        print_status(result)
        
        # 检查结果状态
        status = result.get("status", "")
        if status == "ready":
            print("\n" + "="*60)
            print("成功生成KernelBench格式代码！")
            print("="*60)
            
            generated_task_desc = result.get("generated_task_desc", "")
            op_name = result.get("op_name", "unnamed")
            
            if generated_task_desc:
                print("\n生成的代码预览:")
                print("-" * 40)
                print(generated_task_desc[:500] + ("..." if len(generated_task_desc) > 500 else ""))
                print("-" * 40)
                
                # 询问是否保存
                save_choice = get_user_input("\n是否要保存生成的代码到文件？(y/n): ")
                if save_choice.lower() in ['y', 'yes', '是', '']:
                    save_task_desc_to_file(generated_task_desc, op_name)
                
                print("\n生成的KernelBench格式代码如下：")
                print("="*60)
                print(generated_task_desc)
                print("="*60)
                
                break
            else:
                print("错误：生成的代码为空")
        
        elif status == "unsupported":
            print(f"\n系统不支持此类型的需求。原因: {result.get('agent_message', '')}")
            break
        
        elif status in ["need_clarification", "need_modification"]:
            if i == max_iterations - 1:
                print(f"\n已达到最大交互次数({max_iterations})，结束交互。")
            else:
                print("请继续补充信息...")
        else:
            print(f"未知状态: {status}")
            
    else:
        # 循环结束仍未生成ready状态
        print(f"\n已达到最大交互次数({max_iterations})，未能生成满足要求的代码。")
        print("请尝试提供更明确的需求描述。")
    
    print("\n感谢使用AIKG OpTaskBuilder Tool！")


if __name__ == "__main__":
    asyncio.run(main())