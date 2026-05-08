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
算子生成领域特化的 ToolExecutor

继承 core_v2 的通用 ToolExecutor，添加:
- domain 类工具的硬件参数自动注入
- Agent 调用时的硬件参数合并
- Workflow 初始状态构建（含 op 领域参数）
"""

import logging
from typing import Dict, Any, List

from akg_agents.core_v2.tools.tool_executor import ToolExecutor
from akg_agents.core_v2.tools.tool_registry import ToolRegistry

logger = logging.getLogger(__name__)


class OpToolExecutor(ToolExecutor):
    """算子生成领域的 ToolExecutor

    在通用 ToolExecutor 基础上，为 domain 类工具注入硬件参数（framework/backend/arch/dsl），
    为 Agent/Workflow 调用注入算子领域上下文。
    """

    def _get_hardware_params(self, arguments: Dict[str, Any]) -> Dict[str, str]:
        """从 arguments 或 agent_context 获取硬件参数

        优先级: arguments > agent_context > 默认值
        """
        return {
            "framework": arguments.get("framework", self.agent_context.get("framework", "torch")),
            "backend": arguments.get("backend", self.agent_context.get("backend", "cuda")),
            "arch": arguments.get("arch", self.agent_context.get("arch", "a100")),
            "dsl": arguments.get("dsl", self.agent_context.get("dsl", "triton")),
        }

    async def _execute_via_registry(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """通过 ToolRegistry 执行工具，domain 类工具自动注入硬件参数。"""
        tool_info = ToolRegistry.get_tool(tool_name)
        if tool_info is None:
            display_name = tool_name if len(tool_name) <= 60 else tool_name[:60] + "..."
            return {
                "status": "error",
                "output": "",
                "error_information": f"未知工具: '{display_name}'，请检查工具名拼写。可用工具: {ToolRegistry.list_names()}"
            }

        if tool_info.category == "domain":
            hw_params = self._get_hardware_params(arguments)
            for k, v in hw_params.items():
                arguments.setdefault(k, v)
            logger.info(f"[OpToolExecutor] domain_tool: {tool_name}, hw_params: {hw_params}")

        if tool_name == "load_skill":
            skills_dir = self.agent_context.get("skills_dir")
            if skills_dir:
                arguments.setdefault("skills_dir", skills_dir)

        return await ToolRegistry.aexecute(tool_name, arguments)

    async def _execute_agent(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Agent 执行时注入硬件参数和领域上下文。"""
        hw_params = self._get_hardware_params(arguments)
        for k, v in hw_params.items():
            arguments.setdefault(k, v)
        return await super()._execute_agent(tool_name, arguments)

    def _build_workflow_state(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """构建 workflow 初始状态（含算子领域参数）"""
        state = {
            "op_name": arguments.get("op_name", ""),
            "task_desc": arguments.get("task_desc", ""),
            "dsl": arguments.get("dsl", self.agent_context.get("dsl", "")),
            "framework": arguments.get("framework", self.agent_context.get("framework", "")),
            "backend": arguments.get("backend", self.agent_context.get("backend", "")),
            "arch": arguments.get("arch", self.agent_context.get("arch", "")),
            "task_id": arguments.get("task_id", self.agent_context.get("task_id", "")),
            "user_requirements": arguments.get("user_requirements", ""),
            "previous_code": arguments.get("previous_code", ""),
            "verifier_error": arguments.get("verifier_error", ""),
            "conductor_suggestion": arguments.get("conductor_suggestion", ""),
            "cur_path": arguments.get("cur_path", ""),
            "result": {},
            "should_continue": True,
            "current_step": "",
            "iterations": 0,
            "max_iterations": arguments.get("max_iterations", 10),
        }
        session_id = self.agent_context.get("session_id", "")
        if session_id:
            state["session_id"] = session_id
        return state
