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
Tool 输入参数 Schema
建议：后续添加tool的时候在这里加上pydantic的定义
"""

from typing import Optional, List
from pydantic import BaseModel, Field

class AskUserInput(BaseModel):
    message: str = Field(description="向用户询问的问题或显示的消息")


class FinishInput(BaseModel):
    final_answer: str = Field(description="最终回答或总结")
    success: bool = Field(default=True, description="任务是否成功完成")


class ReadFileInput(BaseModel):
    """read_file tool 输入参数"""
    file_path: str = Field(description="要读取的文件路径")
    encoding: str = Field(default="utf-8", description="文件编码格式")

class SubAgentInput(BaseModel):
    """SubAgent 通用输入参数（用于代码生成类子 Agent）"""
    task_code: str = Field(description="OpTaskBuilder 生成的 task 代码")
    op_name: str = Field(description="算子名称，如 'relu', 'matmul' 等")
    task_id: str = Field(default="default_task", description="任务 ID")
    task_label: str = Field(default="", description="任务标签")
    task_type: Optional[str] = Field(
        default="precision_only",
        description="任务类型：'precision_only' 或 'profile'（用于 codeonly）"
    )
    generated_code: Optional[str] = Field(
        default="",
        description="已生成的代码（用于某些 SubAgent，如 kernel_verifier）"
    )
    device_id: Optional[int] = Field(default=0, description="device ID")
    user_requirements: Optional[str] = Field(
        default="",
        description="用户的额外需求说明。这些需求会被传递给子Agent的prompt中作为算子生成的指导。"
    )


class OpTaskBuilderInput(BaseModel):
    """OpTaskBuilder 子 Agent 输入参数"""
    user_request: str = Field(description="用户的算子生成请求")
    user_feedback: Optional[str] = Field(
        default=None, 
        description="用户的反馈或修改要求"
    )
    task_code: Optional[str] = Field(
        default="",
        description="之前生成的 task_desc 代码（用于修改场景）"
    )
    op_name: Optional[str] = Field(
        default="",
        description="算子名称"
    )
    task_id: str = Field(default="default_task", description="任务 ID")


class WriteFileInput(BaseModel):
    """write_file tool 输入参数
    
    默认保存路径规则：
    - 默认目录: ./akg_agents_outputs/{op_name}/
    - task_desc 代码: task_desc.py
    - triton kernel 代码: kernel.py
    //待完善
    """
    file_path: Optional[str] = Field(
        default=None, 
        description="要写入的文件路径。如果不指定，将使用默认路径: ./akg_agents_outputs/{op_name}/{filename}"
    )
    content: str = Field(description="要写入的文件内容")
    op_name: Optional[str] = Field(
        default=None,
        description="算子名称，用于构建默认目录。如: relu, matmul 等"
    )
    file_type: Optional[str] = Field(
        default="kernel",
        description="文件类型，用于自动命名: 'task_desc'(Torch代码) 或 'kernel'(Triton代码)"
    )
    encoding: str = Field(default="utf-8", description="文件编码格式")
    overwrite: bool = Field(default=False, description="如果文件已存在是否覆盖")


class ExecuteScriptInput(BaseModel):
    """
    用于执行 Skill 中的脚本文件（Python/Bash）
    """
    script_path: str = Field(
        description="脚本文件路径，例如: resources/skills/kernel-workflow/scripts/check_torch_code.py"
    )
    args: Optional[str] = Field(
        default="",
        description="传递给脚本的命令行参数字符串"
    )
    stdin_input: Optional[str] = Field(
        default=None,
        description="传递给脚本的标准输入内容（用于传递代码等长文本）"
    )
    timeout: int = Field(
        default=60,
        description="脚本执行超时时间（秒）"
    )
    working_dir: Optional[str] = Field(
        default=None,
        description="脚本工作目录，默认为项目根目录"
    )


# ============================================================================
# Domain Tools Schemas (verifyTool, profileTool)
# ============================================================================

class VerifyToolInput(BaseModel):
    """
    用于验证生成的 Kernel 代码的正确性，对比框架实现和生成实现的输出结果。
    """
    task_code: str = Field(
        description=(
            "PyTorch/MindSpore 框架实现代码。"
            "必须包含: class Model(nn.Module)、get_inputs()、get_init_inputs()。"
            "如果之前调用过 call_op_task_builder，从其返回的 task_code 字段获取。"
        )
    )
    generated_code: str = Field(
        description=(
            "待验证的 Triton/CUDA/AscendC 生成代码。"
            "必须包含 class ModelNew 实现。"
            "如果之前调用过代码生成类工具，从其返回结果中获取。"
        )
    )
    op_name: str = Field(
        description="算子名称，如 'relu', 'matmul', 'softmax', 'layernorm' 等"
    )
    task_id: str = Field(
        default="default_task",
        description="任务 ID，用于日志和目录命名"
    )
    device_id: int = Field(
        default=0,
        description="设备 ID，默认为 0"
    )
    timeout: int = Field(
        default=300,
        description="验证超时时间（秒），默认 300 秒"
    )


class ProfileToolInput(BaseModel):
    """用于对 Kernel 进行性能分析，返回执行时间和加速比。"""
    task_code: str = Field(
        description=(
            "PyTorch/MindSpore 框架实现代码。"
            "必须包含: class Model(nn.Module)、get_inputs()、get_init_inputs()。"
            "如果之前调用过 call_op_task_builder，从其返回的 task_code 字段获取。"
        )
    )
    generated_code: str = Field(
        description=(
            "待分析的 Triton/CUDA/AscendC 生成代码。"
            "必须包含 class ModelNew 实现。"
            "如果之前调用过代码生成类工具，从其返回结果中获取。"
        )
    )
    op_name: str = Field(
        description="算子名称，如 'relu', 'matmul', 'softmax', 'layernorm' 等"
    )
    task_id: str = Field(
        default="default_task",
        description="任务 ID，用于日志和目录命名"
    )
    device_id: int = Field(
        default=0,
        description="设备 ID，默认为 0"
    )
    run_times: int = Field(
        default=50,
        description="性能测试运行次数，默认 50 次。精确测试可设为 100"
    )
    warmup_times: int = Field(
        default=5,
        description="预热次数，默认 5 次。精确测试可设为 10"
    )

