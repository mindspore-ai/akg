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
core_v2/langgraph - 通用 LangGraph 框架

提供领域无关的 LangGraph 工作流基础设施：
- BaseState: 通用状态定义
- BaseWorkflow: 通用工作流基类
- BaseLangGraphTask: 通用任务基类
- 路由工具、可视化、节点追踪等
"""

from akg_agents.core_v2.langgraph_base.base_state import BaseState
from akg_agents.core_v2.langgraph_base.base_workflow import BaseWorkflow
from akg_agents.core_v2.langgraph_base.base_task import BaseLangGraphTask
from akg_agents.core_v2.langgraph_base.visualizer import WorkflowVisualizer
from akg_agents.core_v2.langgraph_base.node_tracker import track_node
from akg_agents.core_v2.langgraph_base.base_routers import (
    check_step_limit,
    check_agent_repeat_limit,
    get_illegal_agents,
)

__all__ = [
    "BaseState",
    "BaseWorkflow",
    "BaseLangGraphTask",
    "WorkflowVisualizer",
    "track_node",
    "check_step_limit",
    "check_agent_repeat_limit",
    "get_illegal_agents",
]

