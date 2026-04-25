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
op/langgraph - 算子专用 LangGraph 组件

提供算子生成场景的 LangGraph 组件：
- KernelGenState: 算子生成状态（继承 BaseState）
- NodeFactory: 算子节点工厂
- RouterFactory: 算子路由工厂
- LangGraphTask: 算子任务类（继承 BaseLangGraphTask）
- OpTaskBuilderState: OpTaskBuilder 状态
- ConversationalOpGenState: 对话式算子生成状态
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from akg_agents.op.langgraph_op.state import KernelGenState
    from akg_agents.op.langgraph_op.nodes import NodeFactory
    from akg_agents.op.langgraph_op.routers import RouterFactory
    from akg_agents.op.langgraph_op.task import LangGraphTask
    from akg_agents.op.langgraph_op.op_task_builder_state import (
        OpTaskBuilderState,
        OpTaskBuilderStatus,
    )
    from akg_agents.op.langgraph_op.conversational_state import (
        ConversationalOpGenState,
        SubAgentState,
        Message,
        MAX_CONVERSATION_HISTORY_LENGTH,
        limit_conversation_history,
    )

__all__ = [
    "KernelGenState",
    "NodeFactory",
    "RouterFactory",
    "LangGraphTask",
    "OpTaskBuilderState",
    "OpTaskBuilderStatus",
    "ConversationalOpGenState",
    "SubAgentState",
    "Message",
    "MAX_CONVERSATION_HISTORY_LENGTH",
    "limit_conversation_history",
]


def __getattr__(name):
    if name == "KernelGenState":
        from akg_agents.op.langgraph_op.state import KernelGenState

        return KernelGenState
    if name == "NodeFactory":
        from akg_agents.op.langgraph_op.nodes import NodeFactory

        return NodeFactory
    if name == "RouterFactory":
        from akg_agents.op.langgraph_op.routers import RouterFactory

        return RouterFactory
    if name == "LangGraphTask":
        from akg_agents.op.langgraph_op.task import LangGraphTask

        return LangGraphTask
    if name in ("OpTaskBuilderState", "OpTaskBuilderStatus"):
        from akg_agents.op.langgraph_op.op_task_builder_state import (
            OpTaskBuilderState,
            OpTaskBuilderStatus,
        )

        return OpTaskBuilderState if name == "OpTaskBuilderState" else OpTaskBuilderStatus
    if name in (
        "ConversationalOpGenState",
        "SubAgentState",
        "Message",
        "MAX_CONVERSATION_HISTORY_LENGTH",
        "limit_conversation_history",
    ):
        from akg_agents.op.langgraph_op.conversational_state import (
            ConversationalOpGenState,
            SubAgentState,
            Message,
            MAX_CONVERSATION_HISTORY_LENGTH,
            limit_conversation_history,
        )

        return {
            "ConversationalOpGenState": ConversationalOpGenState,
            "SubAgentState": SubAgentState,
            "Message": Message,
            "MAX_CONVERSATION_HISTORY_LENGTH": MAX_CONVERSATION_HISTORY_LENGTH,
            "limit_conversation_history": limit_conversation_history,
        }[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

