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
MainOpAgent 显示格式化模块

提供用户界面相关的格式化函数：
- 格式化显示消息
- 生成提示消息
- 识别简单命令
"""

from typing import Dict, Any, Tuple


def format_display_message(state: Dict[str, Any]) -> str:
    """
    格式化显示消息供前端使用
    
    Args:
        state: 当前状态
        
    Returns:
        格式化后的显示消息
    """
    current_step = state.get("current_step", "")
    
    # 1. 生成 Task 代码后的显示
    if current_step in ["user_confirm", "waiting_for_user_confirmation"]:
        task_code = state.get("task_code", "")
        op_name = state.get("op_name", "")
        description = state.get("op_description", "")
        
        lines = []
        lines.append("=" * 80)
        lines.append("🤖 AI Kernel Assistant")
        lines.append("=" * 80)
        
        if description:
            lines.append(f"\n{description}")
        
        if task_code:
            lines.append(f"\n📦 算子名称: {op_name}")
            lines.append(f"\n📝 生成的 Task 代码:")
            lines.append("-" * 80)
            lines.append(task_code)
            lines.append("-" * 80)
        
        return "\n".join(lines)
    
    # 2. 代码生成完成后的显示
    elif current_step == "completed":
        generation_success = state.get("generation_success")
        verification_result = state.get("verification_result")
        
        lines = []
        lines.append("=" * 80)
        lines.append("🤖 AI Kernel Assistant - 代码生成结果")
        lines.append("=" * 80)
        
        if generation_success:
            lines.append("\n✅ 代码生成成功!")
            generated_code = state.get("generated_code", "")
            if generated_code:
                lines.append(f"\n📝 生成的 {state.get('dsl', 'Triton')} 代码:")
                lines.append("-" * 80)
                lines.append(generated_code)
                lines.append("-" * 80)
            
            if verification_result:
                lines.append("\n✅ 验证通过!")
                profile = state.get("profile_result")
                if profile:
                    # 格式化显示性能数据
                    lines.append("\n📊 性能测试结果:")
                    base_time = profile.get("base_time", 0)
                    gen_time = profile.get("gen_time", 0)
                    speedup = profile.get("speedup", 0)
                    if base_time > 0:
                        lines.append(f"   • 原始性能: {base_time:.2f} us")
                    if gen_time > 0:
                        lines.append(f"   • 生成性能: {gen_time:.2f} us")
                    if speedup > 0:
                        lines.append(f"   • 加速比: {speedup:.2f}x")
                    
                    # 显示其他性能数据
                    run_times = profile.get("run_times")
                    warmup_times = profile.get("warmup_times")
                    if run_times:
                        lines.append(f"   • 测试轮数: {run_times} 次" + (f"（预热 {warmup_times} 次）" if warmup_times else ""))
            else:
                verification_error = state.get("verification_error", "未知错误")
                lines.append(f"\n⚠️ 验证未通过: {verification_error}")
        else:
            generation_error = state.get("generation_error", "未知错误")
            lines.append(f"\n❌ 代码生成失败: {generation_error}")
        
        return "\n".join(lines)
    
    # 3. 代码生成失败
    elif current_step == "failed":
        verification_error = state.get("verification_error", "未知错误")
        lines = []
        lines.append("=" * 80)
        lines.append("🤖 AI Kernel Assistant")
        lines.append("=" * 80)
        lines.append(f"\n❌ 代码生成失败: {verification_error}")
        return "\n".join(lines)
    
    # 4. 意图分类拒绝或无关输入
    elif current_step in ["rejected_by_intent", "irrelevant_input"]:
        description = state.get("op_description", "")
        lines = []
        lines.append("=" * 80)
        lines.append("🤖 AI Kernel Assistant")
        lines.append("=" * 80)
        lines.append(f"\n{description}")
        return "\n".join(lines)
    
    # 5. 不支持的需求
    elif current_step == "unsupported":
        description = state.get("op_description", "")
        lines = []
        lines.append("=" * 80)
        lines.append("🤖 AI Kernel Assistant")
        lines.append("=" * 80)
        lines.append(f"\n{description}")
        return "\n".join(lines)
    
    # 6. 其他情况
    else:
        description = state.get("op_description", "")
        if description:
            return f"\n🤖 {description}"
        return ""


def get_hint_message(state: Dict[str, Any]) -> str:
    """
    获取提示消息
    
    Args:
        state: 当前状态
        
    Returns:
        提示消息
    """
    current_step = state.get("current_step", "")
    has_task_code = bool(state.get("task_code"))
    has_generated_code = bool(state.get("generated_code"))
    
    # 根据状态返回不同的提示
    if current_step in ["rejected_by_intent", "irrelevant_input"]:
        return "\n💡 请输入您的算子开发需求，或输入 '退出' 结束对话"
    elif current_step == "completed":
        return "\n💡 您可以继续提问、输入 '保存' 保存结果，或 '退出' 结束对话"
    elif has_task_code and not has_generated_code:
        return "\n💡 请确认是否按照上述的任务描述开始生成，或者增加您想要的修改"
    else:
        return ""


def is_simple_command(user_input: str) -> Tuple[bool, str]:
    """
    识别简单的控制命令
    
    Args:
        user_input: 用户输入
        
    Returns:
        (is_command, command_type): 是否是简单命令，命令类型 ('exit' / 'save')
    """
    user_input_lower = user_input.strip().lower()
    
    # 如果输入太长，不太可能是简单命令
    if len(user_input_lower) > 20:
        return False, ''
    
    # 退出相关的关键词
    exit_keywords = ['退出', 'quit', 'exit', '再见', 'bye', '结束', 'end', '拜拜']
    if any(kw in user_input_lower for kw in exit_keywords):
        if not any(neg in user_input_lower for neg in ['不', '别', '不要', '不想', 'not', "don't"]):
            return True, 'exit'
    
    # 保存相关的关键词 - 严格匹配，避免复合指令被误判
    # 只有当输入几乎只包含"保存"或"save"时才认为是简单保存命令
    save_keywords = ['保存', 'save']
    
    # 移除保存关键词后，检查剩余内容
    temp_input = user_input_lower
    for kw in save_keywords:
        temp_input = temp_input.replace(kw, '')
    
    # 移除常见的无意义字符（标点、空格、助词等）
    temp_input = temp_input.strip()
    for char in ['一下', '吧', '呗', '。', '，', '！', '？', '.', ',', '!', '?', ' ']:
        temp_input = temp_input.replace(char, '')
    
    # 如果去掉保存关键词和无意义字符后，输入为空或很短，才认为是简单保存命令
    if any(kw in user_input_lower for kw in save_keywords):
        if len(temp_input) <= 2:  # 剩余内容很少（容忍一些助词残留）
            if not any(neg in user_input_lower for neg in ['不', '别', '不要', 'not', "don't"]):
                return True, 'save'
    
    return False, ''

