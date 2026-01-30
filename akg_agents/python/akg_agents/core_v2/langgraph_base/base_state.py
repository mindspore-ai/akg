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

"""通用 LangGraph 状态定义

提供领域无关的基础状态类型，适用于任何 LangGraph 工作流：
- 文档处理
- RAG 应用
- 代码生成
- 等等
"""

from typing import TypedDict, Annotated, Optional, List
from operator import add


class BaseState(TypedDict, total=False):
    """通用 LangGraph 工作流状态基类
    
    仅包含框架级字段，完全不涉及任何特定领域概念。
    领域专用状态（如算子生成）应继承此类并添加专用字段。
    
    Example:
        class MyDomainState(BaseState, total=False):
            # 添加领域专用字段
            document_content: str
            analysis_result: dict
    """
    
    # === 任务标识 ===
    task_id: str                    # 任务唯一标识
    task_label: str                 # 任务显示标签（用于 UI）
    session_id: str                 # 会话 ID（用于流式输出和 session 隔离）
    
    # === 流程控制 ===
    iteration: int                  # 当前迭代次数
    step_count: int                 # 当前步数
    max_iterations: int             # 最大迭代次数
    
    # === 历史记录 ===
    # 使用 Annotated[List[str], add] 实现自动累积
    agent_history: Annotated[List[str], add]  # Agent 执行历史
    
    # === 通用结果字段 ===
    success: bool                   # 任务是否成功
    error_message: Optional[str]    # 错误信息（如果失败）

