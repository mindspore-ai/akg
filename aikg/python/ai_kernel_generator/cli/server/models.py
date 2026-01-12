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

    # RAG 参数
    rag: bool = Field(default=False, description="是否使用 RAG 模式")

    # 输出路径（保存 saved_verifications）
    output_path: Optional[str] = Field(
        default=None,
        description="保存目录根路径（用于 saved_verifications）",
    )


class ServerStatusResponse(BaseModel):
    """服务器状态响应"""

    status: str
    version: str
    backend: str
    arch: str
    devices: List[int]
