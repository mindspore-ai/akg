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
任务构造器内部工具注册中心

遵循 akg_agents v2 的工具描述格式（YAML schema），使用独立注册表，
避免污染全局 tools.yaml。工具只在 TaskConstructor agent 的 ReAct 循环中可用。
"""

import logging
import inspect
from typing import Dict, Any, Callable, List, Optional

logger = logging.getLogger(__name__)


class TaskToolRegistry:
    """
    任务构造器工具注册中心（单例）

    工具定义格式与 tools.yaml 兼容：
    {
        "type": "domain_tool",
        "function": {
            "name": "tool_name",
            "description": "...",
            "parameters": { JSON Schema }
        }
    }
    """

    _tools: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def register(
        cls,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        func: Callable,
        tool_type: str = "domain_tool",
    ) -> None:
        """注册工具"""
        cls._tools[name] = {
            "type": tool_type,
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters,
            },
            "execute": func,
        }
        logger.debug(f"[TaskToolRegistry] 注册工具: {name}")

    @classmethod
    def execute(cls, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """执行工具，返回标准结果 {status, output, error_information}"""
        if name not in cls._tools:
            return {
                "status": "error",
                "output": "",
                "error_information": f"未知工具: {name}，可用: {list(cls._tools.keys())}",
            }

        tool = cls._tools[name]
        func = tool["execute"]

        try:
            sig = inspect.signature(func)
            params = sig.parameters

            has_var_keyword = any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
            )
            if has_var_keyword:
                result = func(**arguments)
            else:
                accepted = {n for n in params.keys() if n != "self"}
                filtered = {k: v for k, v in arguments.items() if k in accepted}
                result = func(**filtered)

            if isinstance(result, dict):
                if "status" in result:
                    return result
                return {"status": "success", "output": str(result), "error_information": ""}
            return {"status": "success", "output": str(result), "error_information": ""}

        except Exception as e:
            logger.error(f"[TaskToolRegistry] 执行失败 {name}: {e}")
            return {"status": "error", "output": "", "error_information": str(e)}

    @classmethod
    def get_tools_for_prompt(cls) -> str:
        """生成 LLM system prompt 中的工具描述"""
        lines = []
        for name, tool in cls._tools.items():
            func_def = tool["function"]
            params = func_def["parameters"].get("properties", {})
            required = func_def["parameters"].get("required", [])

            param_strs = []
            for pname, pinfo in params.items():
                req = " (必填)" if pname in required else " (可选)"
                desc = pinfo.get("description", pinfo.get("type", ""))
                param_strs.append(f"    - {pname}: {desc}{req}")

            lines.append(
                f"### {name}\n{func_def['description']}\n参数:\n"
                + "\n".join(param_strs)
            )
        return "\n\n".join(lines)

    @classmethod
    def list_tools_yaml(cls) -> Dict[str, Dict[str, Any]]:
        """导出为 tools.yaml 兼容格式"""
        return {
            name: {
                "type": tool["type"],
                "function": tool["function"],
            }
            for name, tool in cls._tools.items()
        }

    @classmethod
    def list_names(cls) -> List[str]:
        return list(cls._tools.keys())

    @classmethod
    def get_tool_info(cls, name: str) -> Optional[Dict[str, Any]]:
        tool = cls._tools.get(name)
        if tool:
            return tool["function"]
        return None
