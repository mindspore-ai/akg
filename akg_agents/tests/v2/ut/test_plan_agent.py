#!/usr/bin/env python3
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
PlanAgent 测试

运行方式：
    # 交互式测试
    python tests/v2/ut/test_plan_agent.py
    
    # 使用预设场景
    python tests/v2/ut/test_plan_agent.py --preset simple
    python tests/v2/ut/test_plan_agent.py --preset incomplete
    python tests/v2/ut/test_plan_agent.py --preset all
"""

import asyncio
import argparse
import sys
import os
import json
import logging
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "python"))

from akg_agents.core_v2.agents.plan import PlanAgent

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

# 模拟可用工具列表
AVAILABLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "call_task_builder",
            "description": "生成算子的 task_desc 定义（PyTorch 格式）",
            "parameters": {
                "type": "object",
                "properties": {
                    "op_name": {"type": "string", "description": "算子名称"},
                    "user_request": {"type": "string", "description": "用户需求描述"}
                },
                "required": ["op_name", "user_request"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "call_kernel_gen",
            "description": "生成 kernel 代码",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_desc": {"type": "string", "description": "task_desc 代码"},
                    "op_name": {"type": "string", "description": "算子名称"}
                },
                "required": ["task_desc", "op_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "call_verifier",
            "description": "验证 kernel 精度和性能",
            "parameters": {
                "type": "object",
                "properties": {
                    "kernel_code": {"type": "string", "description": "kernel 代码"},
                    "task_desc": {"type": "string", "description": "task_desc 代码"}
                },
                "required": ["kernel_code", "task_desc"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取文件内容",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "文件路径"}
                },
                "required": ["file_path"]
            }
        }
    }
]

# 预设测试场景
PRESETS = {
    "simple": {
        "user_input": "使用 triton-cuda 为 torch 生成一个 ReLU 算子，backend 是 cuda",
        "description": "简单算子，信息完整",
        "history_compress": []
    },
    "complex": {
        "user_input": "使用 triton-cuda 生成一个高性能的 MatMul 算子，要求使用 tensor core 优化，framework torch，backend cuda",
        "description": "复杂算子，有性能要求",
        "history_compress": []
    },
    "incomplete_op": {
        "user_input": "帮我生成一个算子",
        "description": "算子生成，缺少多项信息",
        "history_compress": []
    },
    "incomplete_file": {
        "user_input": "帮我读一个文件",
        "description": "文件操作，缺少路径",
        "history_compress": []
    },
    "with_code": {
        "user_input": """帮我用 triton-cuda 生成这个算子的 kernel，framework torch，backend cuda:

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return torch.softmax(x, dim=-1)

def get_inputs():
    return [torch.randn(16, 1024, 1024)]

def get_init_inputs():
    return []
""",
        "description": "用户提供了 task_desc 代码",
        "history_compress": []
    },
    "file_complete": {
        "user_input": "读取 /path/to/config.yaml 文件",
        "description": "文件操作，信息完整",
        "history_compress": []
    },
    "with_history": {
        "user_input": "继续完成 ReLU 算子的生成",
        "description": "有执行历史的场景",
        "history_compress": [
            {
                "tool_name": "call_task_builder",
                "status": "success",
                "summary": "task_desc 生成成功"
            }
        ]
    }
}


def print_separator(title: str = "", char: str = "=", width: int = 80):
    """打印分隔线"""
    if title:
        padding = (width - len(title) - 2) // 2
        print(f"\n{char * padding} {title} {char * padding}")
    else:
        print(char * width)


def print_result(result: dict):
    """打印规划结果"""
    print_separator("规划结果")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print_separator()
    
    # 简要总结
    status = result.get("result", {}).get("status", "unknown")
    desc = result.get("result", {}).get("desc", "")
    print(f"\n状态: {status}")
    print(f"描述: {desc}")
    
    if status == "success":
        steps = result.get("arguments", {}).get("steps", [])
        print(f"\n规划步骤 ({len(steps)} 步):")
        for step in steps:
            print(f"  {step['step_id']}. {step['desc']}")


async def test_single(preset_name: str, preset: dict):
    """测试单个场景"""
    print_separator(f"测试场景: {preset_name}")
    print(f"描述: {preset['description']}")
    print(f"用户输入: {preset['user_input'][:100]}{'...' if len(preset['user_input']) > 100 else ''}")
    
    if preset.get("history_compress"):
        print(f"执行历史: {len(preset['history_compress'])} 条记录")
    
    agent = PlanAgent()
    
    result, prompt, reasoning = await agent.run(
        user_input=preset["user_input"],
        available_tools=AVAILABLE_TOOLS,
        history_compress=preset.get("history_compress", []),
        task_id=f"test_{preset_name}",
        model_level="standard"
    )
    
    print_result(result)
    return result


async def run_preset_test(preset_name: str):
    """运行预设测试"""
    if preset_name == "all":
        for name, preset in PRESETS.items():
            await test_single(name, preset)
            print("\n")
        return
    
    if preset_name not in PRESETS:
        print(f"未知预设: {preset_name}")
        print(f"可用预设: {', '.join(PRESETS.keys())}, all")
        return
    
    await test_single(preset_name, PRESETS[preset_name])


async def run_interactive_test():
    """交互式测试"""
    print_separator("PlanAgent 交互式测试")
    print("\n可用命令:")
    print("  输入需求描述开始规划")
    print("  输入 'preset <name>' 使用预设场景")
    print("  输入 'q' 或 'quit' 退出")
    print(f"\n可用预设: {', '.join(PRESETS.keys())}")
    print_separator()
    
    # 检查 API Key
    from akg_agents.utils.environment_check import _check_llm_api
    if not _check_llm_api():
        raise ValueError("LLM API Key 配置或连接有问题，请检查。")
    
    agent = PlanAgent()
    
    while True:
        print("\n请输入您的需求（或输入 'q' 退出）:")
        try:
            user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n退出测试")
            break
        
        if user_input.lower() in ['q', 'quit', 'exit']:
            print("退出测试")
            break
        
        if not user_input:
            continue
        
        # 处理预设命令
        if user_input.startswith("preset "):
            preset_name = user_input.split(" ", 1)[1].strip()
            if preset_name in PRESETS:
                await test_single(preset_name, PRESETS[preset_name])
            else:
                print(f"未知预设: {preset_name}")
                print(f"可用预设: {', '.join(PRESETS.keys())}")
            continue
        
        # 正常规划
        try:
            result, prompt, reasoning = await agent.run(
                user_input=user_input,
                available_tools=AVAILABLE_TOOLS,
                history_compress=[],
                task_id="interactive_test",
                model_level="standard"
            )
            print_result(result)
                
        except Exception as e:
            print(f"\n错误: {e}")
            import traceback
            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="PlanAgent 测试")
    parser.add_argument(
        "--preset", "-p",
        choices=list(PRESETS.keys()) + ["all"],
        help="使用预设场景进行测试"
    )
    
    args = parser.parse_args()
    
    if args.preset:
        asyncio.run(run_preset_test(args.preset))
    else:
        asyncio.run(run_interactive_test())


if __name__ == "__main__":
    main()

