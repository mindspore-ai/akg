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
PlanningAgent 交互式测试

使用方法：
    # 默认使用 standard 模型
    python -m tests.op.ut.test_planning_agent_interactive
    
    # 使用 fast 模型（更快但质量可能略低）
    python -m tests.op.ut.test_planning_agent_interactive --model fast
    
    # 使用 complex 模型（更慢但质量更高）
    python -m tests.op.ut.test_planning_agent_interactive --model complex
    
    # 预设场景测试
    python -m tests.op.ut.test_planning_agent_interactive --preset simple
    python -m tests.op.ut.test_planning_agent_interactive --preset complex
    python -m tests.op.ut.test_planning_agent_interactive --preset with_task_desc
"""

import asyncio
import argparse
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from akg_agents.core_v2.agents.plan import PlanningAgent


# 模拟可用工具列表（OpenAI function calling 格式）
AVAILABLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "call_op_task_builder",
            "description": "生成或修改 task_desc 代码。task_desc 定义了算子的输入输出规格。",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_request": {
                        "type": "string",
                        "description": "用户的算子生成请求或修改要求"
                    }
                },
                "required": ["user_request"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "call_designer",
            "description": "设计 kernel 架构。适用于复杂算子或有特殊性能要求的场景。",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_code": {
                        "type": "string",
                        "description": "task_desc 代码"
                    },
                    "op_name": {
                        "type": "string",
                        "description": "算子名称"
                    },
                    "user_requirements": {
                        "type": "string",
                        "description": "用户的优化需求（如核内二次切分）"
                    }
                },
                "required": ["task_code", "op_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "call_coder_only",
            "description": "快速生成 kernel 代码。默认的代码生成方式。",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_code": {
                        "type": "string",
                        "description": "task_desc 代码"
                    },
                    "op_name": {
                        "type": "string",
                        "description": "算子名称"
                    },
                    "user_requirements": {
                        "type": "string",
                        "description": "kernel 优化需求（可选）"
                    }
                },
                "required": ["task_code", "op_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "call_evolve",
            "description": "使用进化搜索生成高性能 kernel。适用于需要高性能的场景。",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_code": {
                        "type": "string",
                        "description": "task_desc 代码"
                    },
                    "op_name": {
                        "type": "string",
                        "description": "算子名称"
                    },
                    "user_requirements": {
                        "type": "string",
                        "description": "kernel 优化需求（可选）"
                    }
                },
                "required": ["task_code", "op_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "call_adaptive_search",
            "description": "使用树搜索生成 kernel。",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_code": {
                        "type": "string",
                        "description": "task_desc 代码"
                    },
                    "op_name": {
                        "type": "string",
                        "description": "算子名称"
                    }
                },
                "required": ["task_code", "op_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "call_kernel_verifier",
            "description": "验证 kernel 的精度和性能。",
            "parameters": {
                "type": "object",
                "properties": {
                    "kernel_code": {
                        "type": "string",
                        "description": "kernel 代码"
                    },
                    "task_code": {
                        "type": "string",
                        "description": "task_desc 代码"
                    },
                    "op_name": {
                        "type": "string",
                        "description": "算子名称"
                    }
                },
                "required": ["kernel_code", "task_code", "op_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "ask_user",
            "description": "向用户询问或展示信息。",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "要询问或展示的内容"
                    }
                },
                "required": ["message"]
            }
        }
    }
]

# 预设测试场景
PRESETS = {
    "simple": {
        "user_input": "生成一个 ReLU 算子",
        "has_task_desc": False,
        "description": "简单算子，无特殊要求"
    },
    "complex": {
        "user_input": "生成一个高性能的 MatMul 算子，要求使用 tensor core 优化",
        "has_task_desc": False,
        "description": "复杂算子，有性能要求"
    },
    "with_task_desc": {
        "user_input": """帮我生成这个算子的 kernel:

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
        "has_task_desc": True,
        "description": "用户已提供 task_desc"
    },
    "optimization": {
        "user_input": "生成 ReLU 算子，核内进行二次切分",
        "has_task_desc": False,
        "description": "有特殊优化要求"
    }
}


def print_separator(title: str = "", char: str = "=", width: int = 80):
    """打印分隔线"""
    if title:
        padding = (width - len(title) - 2) // 2
        print(f"\n{char * padding} {title} {char * padding}")
    else:
        print(char * width)


def print_result(title: str, content: str):
    """打印结果"""
    print_separator(title)
    print(content)
    print_separator()


async def run_interactive_test(model_level: str = "standard"):
    """交互式测试"""
    print_separator("PlanningAgent 交互式测试")
    print(f"模型级别: {model_level}")
    print("\n可用命令:")
    print("  输入需求描述开始规划")
    print("  输入 'q' 或 'quit' 退出")
    print("  输入 'preset <name>' 使用预设场景 (simple/complex/with_task_desc/optimization)")
    print_separator()
    
    agent = PlanningAgent(task_id="interactive_test", model_level=model_level)
    
    while True:
        print("\n请输入您的需求（或输入 'q' 退出）:")
        user_input = input("> ").strip()
        
        if user_input.lower() in ['q', 'quit', 'exit']:
            print("退出测试")
            break
        
        if not user_input:
            continue
        
        # 处理预设命令
        has_task_desc = False
        if user_input.startswith("preset "):
            preset_name = user_input.split(" ", 1)[1].strip()
            if preset_name in PRESETS:
                preset = PRESETS[preset_name]
                user_input = preset["user_input"]
                has_task_desc = preset["has_task_desc"]
                print(f"\n使用预设场景: {preset_name}")
                print(f"描述: {preset['description']}")
                print(f"用户输入: {user_input[:100]}...")
            else:
                print(f"未知预设: {preset_name}")
                print(f"可用预设: {', '.join(PRESETS.keys())}")
                continue
        
        # 检测是否包含 task_desc
        if "class Model" in user_input and "nn.Module" in user_input:
            has_task_desc = True
            print("\n检测到用户提供了 task_desc 代码")
        
        try:
            # 第一步：生成全局 todolist
            print("\n正在生成全局 todolist...")
            global_todolist = await agent.generate_global_todolist(
                user_input=user_input,
                available_tools=AVAILABLE_TOOLS,
                has_task_desc=has_task_desc
            )
            print_result("全局 TodoList", global_todolist)
            
            # 询问是否继续生成详细 todolist
            print("\n是否继续生成详细 todolist? (y/n, 默认 y)")
            continue_input = input("> ").strip().lower()
            
            if continue_input in ['n', 'no']:
                continue
            
            # 第二步：生成详细 todolist
            print("\n正在生成详细 todolist...")
            detailed_todolist = await agent.generate_detailed_todolist(
                user_input=user_input,
                global_todo_list=global_todolist,
                available_tools=AVAILABLE_TOOLS,
                action_history=[]
            )
            print_result("详细 TodoList", detailed_todolist)
            
            # 模拟动作历史，测试进度更新
            print("\n是否模拟执行第一个动作并更新 todolist? (y/n, 默认 n)")
            simulate_input = input("> ").strip().lower()
            
            if simulate_input in ['y', 'yes']:
                # 模拟执行第一个动作
                action_history = [
                    {
                        "tool_name": "call_op_task_builder",
                        "result": {
                            "status": "success",
                            "output_path": "op_task_builder/task_desc.py"
                        }
                    }
                ]
                
                print("\n模拟动作历史:")
                print(f"  1. call_op_task_builder → success")
                
                print("\n正在更新详细 todolist...")
                updated_todolist = await agent.generate_detailed_todolist(
                    user_input=user_input,
                    global_todo_list=global_todolist,
                    available_tools=AVAILABLE_TOOLS,
                    action_history=action_history
                )
                print_result("更新后的 TodoList", updated_todolist)
        
        except Exception as e:
            print(f"\n错误: {e}")
            import traceback
            traceback.print_exc()


async def run_preset_test(preset_name: str, model_level: str = "standard"):
    """运行预设测试"""
    if preset_name not in PRESETS:
        print(f"未知预设: {preset_name}")
        print(f"可用预设: {', '.join(PRESETS.keys())}")
        return
    
    preset = PRESETS[preset_name]
    
    print_separator(f"预设测试: {preset_name}")
    print(f"描述: {preset['description']}")
    print(f"模型级别: {model_level}")
    print(f"has_task_desc: {preset['has_task_desc']}")
    print_separator()
    
    print("\n用户输入:")
    print(preset["user_input"])
    
    agent = PlanningAgent(task_id=f"preset_test_{preset_name}", model_level=model_level)
    
    try:
        # 生成全局 todolist
        print("\n正在生成全局 todolist...")
        global_todolist = await agent.generate_global_todolist(
            user_input=preset["user_input"],
            available_tools=AVAILABLE_TOOLS,
            has_task_desc=preset["has_task_desc"]
        )
        print_result("全局 TodoList", global_todolist)
        
        # 生成详细 todolist
        print("\n正在生成详细 todolist...")
        detailed_todolist = await agent.generate_detailed_todolist(
            user_input=preset["user_input"],
            global_todo_list=global_todolist,
            available_tools=AVAILABLE_TOOLS,
            action_history=[]
        )
        print_result("详细 TodoList", detailed_todolist)
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


async def run_all_presets(model_level: str = "standard"):
    """运行所有预设测试"""
    print_separator("运行所有预设测试")
    
    for preset_name in PRESETS:
        await run_preset_test(preset_name, model_level)
        print("\n" + "=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="PlanningAgent 交互式测试")
    parser.add_argument(
        "--model", "-m",
        choices=["fast", "standard", "complex"],
        default="standard",
        help="模型级别 (default: standard)"
    )
    parser.add_argument(
        "--preset", "-p",
        choices=list(PRESETS.keys()) + ["all"],
        help="使用预设场景进行测试"
    )
    
    args = parser.parse_args()
    
    if args.preset:
        if args.preset == "all":
            asyncio.run(run_all_presets(args.model))
        else:
            asyncio.run(run_preset_test(args.preset, args.model))
    else:
        asyncio.run(run_interactive_test(args.model))


if __name__ == "__main__":
    main()

