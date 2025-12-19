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
数据模型定义

定义工作流服务的所有 Pydantic 数据模型。
"""

from typing import Optional, List
from pydantic import BaseModel, Field


class CliMainAgentRequest(BaseModel):
    """MainAgent 请求（多轮对话：TaskInit + 用户确认 + 触发 Job）。"""

    session_id: str = Field(
        default="", description="客户端会话 ID，用于隔离不同客户端的请求"
    )
    action: str = Field(
        description="动作：start/revise/confirm/cancel（start 会覆盖该 session 的历史状态）"
    )
    user_input: str = Field(
        default="",
        description="用户输入：start=初始需求；revise=补充/反馈；confirm/cancel 可为空",
    )

    framework: str = Field(default="torch", description="框架名称")
    backend: str = Field(default="cuda", description="后端名称")
    arch: str = Field(default="a100", description="架构名称")
    dsl: str = Field(default="triton_cuda", description="DSL 类型")

    # 影响 job 执行的参数（confirm 阶段生效）
    use_stream: bool = Field(default=False, description="是否启用 LLM 流式输出")

    # Evolve 参数（confirm 阶段生效）
    use_evolve: bool = Field(default=False, description="是否使用 evolve 模式")
    max_rounds: int = Field(default=1, description="Evolve 最大轮数")
    parallel_num: int = Field(default=1, description="Evolve 并行任务数")
    num_islands: int = Field(default=1, description="Evolve 岛屿数量")
    migration_interval: int = Field(default=0, description="Evolve 迁移间隔")
    elite_size: int = Field(default=0, description="Evolve 精英数量")
    parent_selection_prob: float = Field(default=0.5, description="Evolve 父代选择概率")


class StepTiming(BaseModel):
    """阶段耗时信息"""

    stage: str
    duration: float
    timestamp: str
    error: Optional[str] = None


class CliExecuteResponse(BaseModel):
    """工作流执行响应"""

    success: bool
    op_name: str
    task_init_status: str
    task_desc: str = Field(default="", description="KernelBench 格式代码")
    kernel_code: str = Field(default="", description="生成的 Kernel 代码")
    verification_result: bool
    step_timings: List[StepTiming]
    total_time: float
    error: Optional[str] = None
    metadata: Optional[dict] = Field(
        default_factory=dict, description="额外元数据，如性能分析结果"
    )


class ServerStatusResponse(BaseModel):
    """服务器状态响应"""

    status: str
    version: str
    backend: str
    arch: str
    devices: List[int]
