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
Tools for ReAct Agent

包含：
- Tool Schemas: 所有 Tool 的 Pydantic 输入参数定义
- SubAgent Tools: SubAgent → LangChain Tool 适配器（包括 call_op_task_builder）
- Basic Tools: ask_user, finish, read_file
- Domain Tools: verify_kernel, profile_kernel (领域专用工具)
"""

from akg_agents.core.tools.tool_schemas import (
    AskUserInput,
    FinishInput,
    ReadFileInput,
    SubAgentInput,
    OpTaskBuilderInput,
    # Domain Tools Schemas
    VerifyToolInput,
    ProfileToolInput,
)
from akg_agents.core.tools.sub_agent_tool import create_sub_agent_tools
from akg_agents.core.tools.basic_tools import (
    ask_user,
    finish,
    read_file,
    create_basic_tools,
)
from akg_agents.core.tools.domain_tools import (
    create_domain_tools,
    get_default_domain_tools,
)

__all__ = [
    # Schemas
    "AskUserInput",
    "FinishInput",
    "ReadFileInput",
    "SubAgentInput",
    "OpTaskBuilderInput",
    "VerifyToolInput",
    "ProfileToolInput",
    # SubAgent Tools
    "create_sub_agent_tools",
    # Basic Tools
    "ask_user",
    "finish",
    "read_file",
    "create_basic_tools",
    # Domain Tools
    "create_domain_tools",
    "get_default_domain_tools",
]
