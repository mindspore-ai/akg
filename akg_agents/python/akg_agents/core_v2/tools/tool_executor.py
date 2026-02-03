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

import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)

class ToolExecutor:
    def __init__(self, agent_registry: Dict[str, Any] = None, 
                 agent_context: Dict[str, Any] = None,
                 history: List = None):
        self.agent_registry = agent_registry or {}
        self.agent_context = agent_context or {}
        self.history = history or []
        self.tool_types = self._load_tool_types()
    
    def _load_tool_types(self) -> Dict[str, str]:
        """加载工具类型映射"""
        try:
            from akg_agents import get_project_root
            tools_file = Path(get_project_root()) / "core_v2" / "config" / "domain_tools.yaml"
            with open(tools_file, "r", encoding="utf-8") as f:
                tools_config = yaml.safe_load(f)
            
            tool_types = {}
            for tool_name, tool_def in tools_config.get("tools", {}).items():
                tool_types[tool_name] = tool_def.get("type", "basic_tool")
            return tool_types
        except Exception as e:
            logger.warning(f"[ToolExecutor] 加载工具类型失败: {e}")
            return {}
    
    async def execute(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        if tool_name in self.agent_registry:
            return await self._execute_agent(tool_name, arguments)
        else:
            return await self._execute_basic_tool(tool_name, arguments)
    
    async def _execute_agent(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        try:
            agent_info = self.agent_registry[tool_name]
            agent_class = agent_info["agent_class"]
            
            # plan agent 特殊处理
            if tool_name == "plan":
                agent = agent_class()
                
                # 自动补充参数
                history_compress = [
                    {"tool_name": r.tool_name, "arguments": r.arguments, "result": r.result}
                    for r in self.history[-10:]
                ] if self.history else []
                
                result, full_prompt, reasoning = await agent.run(
                    user_input=arguments.get("user_input", self.agent_context.get("user_input", "")),
                    available_tools=arguments.get("available_tools", []),
                    history_compress=history_compress,
                    task_id=self.agent_context.get("task_id", ""),
                    model_level=self.agent_context.get("model_level", "standard")
                )
                
                # 转换 plan 格式
                if isinstance(result, dict) and "result" in result:
                    plan_result = result["result"]
                    if plan_result.get("status") == "success":
                        return {
                            "status": "success",
                            "output": result.get("arguments", {}),
                            "error_information": ""
                        }
                    else:
                        return {
                            "status": "fail",
                            "output": "",
                            "error_information": plan_result.get("desc", "规划失败")
                        }
                return {"status": "error", "output": "", "error_information": "plan 返回格式错误"}
            
            # 其他 agent
            agent = agent_class(config=self.agent_context.get("config", {}))
            
            run_params = {
                "op_name": arguments.get("op_name", ""),
                "task_desc": arguments.get("task_desc", ""),
                "dsl": arguments.get("dsl", self.agent_context.get("dsl", "triton")),
                "framework": arguments.get("framework", self.agent_context.get("framework", "torch")),
                "backend": arguments.get("backend", self.agent_context.get("backend", "cuda")),
                "arch": arguments.get("arch", self.agent_context.get("arch", "a100")),
                "user_requirements": arguments.get("user_requirements", ""),
                "task_id": arguments.get("task_id", self.agent_context.get("task_id", "")),
                "history_compress": [
                    {"tool_name": r.tool_name, "arguments": r.arguments, "result": r.result}
                    for r in self.history[-10:]
                ] if self.history else []
            }
            
            result = await agent.run(**run_params)
            
            if isinstance(result, tuple) and len(result) == 3:
                generated_code, full_prompt, reasoning = result
                return {
                    "status": "success",
                    "output": generated_code,
                    "error_information": "",
                    "generated_code": generated_code,
                    "full_prompt": full_prompt,
                    "reasoning": reasoning
                }
            elif isinstance(result, dict) and "status" in result:
                return result
            else:
                return {"status": "success", "output": str(result), "error_information": ""}
        
        except Exception as e:
            logger.error(f"[ToolExecutor] Agent 执行失败: {tool_name}, {e}", exc_info=True)
            return {"status": "error", "output": "", "error_information": f"Agent 执行失败: {str(e)}"}
    
    async def _execute_basic_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """执行 basic_tool"""
        try:
            from akg_agents.core_v2.tools import basic_tools
            
            tool_func = getattr(basic_tools, tool_name, None)
            if not tool_func:
                return {"status": "error", "output": "", "error_information": f"未知工具: {tool_name}"}
            
            result = tool_func(**arguments)
            
            if isinstance(result, dict) and "status" in result:
                return result
            
            return {"status": "success", "output": str(result), "error_information": ""}
        
        except Exception as e:
            logger.error(f"[ToolExecutor] 工具执行失败: {tool_name}, {e}")
            return {"status": "error", "output": "", "error_information": str(e)}