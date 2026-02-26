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
统一工具注册表 (Unified Tool Registry)

所有工具通过 ToolRegistry.register() 注册，通过 scope + category 过滤。
替代原有的 tools.yaml + TaskToolRegistry 双系统。

核心概念:
  - category: 工具的功能分类 (basic, domain, code_analysis, execution, interaction, agent, workflow)
  - scope: 哪些 agent 可以使用该工具 (kernel_agent, task_constructor, all, ...)
    "all" 表示所有 scope 都可使用

使用方式:
    from akg_agents.core_v2.tools.tool_registry import ToolRegistry

    # 注册工具
    ToolRegistry.register(
        name="read_file",
        description="读取文件内容",
        parameters={...},
        func=read_file,
        category="basic",
        scopes=["all"],
    )

    # 获取工具列表 (OpenAI function calling 格式)
    tools = ToolRegistry.get_tools_for_prompt(scope="kernel_agent", format="openai_json")

    # 获取工具列表 (Markdown 格式, 用于纯文本 prompt)
    desc = ToolRegistry.get_tools_for_prompt(scope="task_constructor", format="markdown")

    # 执行工具 (同步)
    result = ToolRegistry.execute("read_file", {"file_path": "/path/to/file"})

    # 执行工具 (异步)
    result = await ToolRegistry.aexecute("read_file", {"file_path": "/path/to/file"})
"""

import inspect
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable, ClassVar

logger = logging.getLogger(__name__)


@dataclass
class ToolInfo:
    """工具元信息

    Attributes:
        name: 工具名称（全局唯一）
        description: 工具描述（会注入到 LLM prompt 中，应尽可能详细）
        parameters: 参数的 JSON Schema 定义
        category: 功能分类
            - "basic": 基础文件/IO 操作
            - "domain": 领域专用工具（verify_kernel, profile_kernel 等）
            - "code_analysis": 代码分析工具（trace_dependencies, assemble_task 等）
            - "execution": 代码执行工具（execute_script 等）
            - "interaction": 交互工具（ask_user, finish 等）
            - "agent": Agent 调用工具（call_kernel_gen 等）
            - "workflow": Workflow 调用工具（use_xxx_workflow 等）
        scopes: 允许使用此工具的 scope 列表。包含 "all" 表示所有 scope 均可使用
        execute: 实际执行函数（支持同步和异步函数）
        requires_approval: 是否需要用户确认后才能执行
    """
    name: str
    description: str
    parameters: Dict[str, Any]
    category: str
    scopes: List[str]
    execute: Callable
    requires_approval: bool = False


class ToolRegistry:
    """统一工具注册表

    使用 classmethod 提供全局单例模式。所有工具通过 register() 注册，
    通过 get_tools() / get_tools_for_prompt() 按 scope + category 过滤获取。

    设计要点:
    - 纯代码注册，无 YAML 依赖
    - scope 过滤: 每个 agent 声明自己的 scope，只看到被允许的工具
    - category 分类: 便于按功能组批量启用/禁用
    - introspection 参数过滤: execute() 自动根据函数签名过滤多余参数
    - 同时支持同步 (execute) 和异步 (aexecute) 执行
    """

    _tools: ClassVar[Dict[str, ToolInfo]] = {}

    @classmethod
    def register(
        cls,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        func: Callable,
        category: str = "basic",
        scopes: Optional[List[str]] = None,
        requires_approval: bool = False,
    ) -> None:
        """注册工具到全局注册表

        Args:
            name: 工具名称（全局唯一，重复注册会覆盖）
            description: 工具描述
            parameters: JSON Schema 格式的参数定义
            func: 执行函数（同步或异步）
            category: 功能分类
            scopes: 允许使用的 scope 列表，默认 ["all"]
            requires_approval: 是否需要用户确认
        """
        if scopes is None:
            scopes = ["all"]

        tool_info = ToolInfo(
            name=name,
            description=description,
            parameters=parameters,
            category=category,
            scopes=scopes,
            execute=func,
            requires_approval=requires_approval,
        )
        cls._tools[name] = tool_info
        logger.debug(f"[ToolRegistry] 注册工具: {name} (category={category}, scopes={scopes})")

    @classmethod
    def has_tool(cls, name: str) -> bool:
        """检查工具是否已注册"""
        return name in cls._tools

    @classmethod
    def get_tool(cls, name: str) -> Optional[ToolInfo]:
        """获取单个工具信息"""
        return cls._tools.get(name)

    @classmethod
    def get_tools(
        cls,
        scope: Optional[str] = None,
        categories: Optional[List[str]] = None,
    ) -> List[ToolInfo]:
        """按 scope + category 过滤获取工具列表

        过滤规则:
        - scope: 工具的 scopes 包含 "all" 或包含指定 scope 时匹配
        - categories: 工具的 category 在列表中时匹配（为 None 表示不限）

        Args:
            scope: Agent 的 scope 标识（如 "kernel_agent", "task_constructor"）
            categories: 要过滤的 category 列表

        Returns:
            匹配的 ToolInfo 列表
        """
        tools = []
        for tool in cls._tools.values():
            # scope 过滤: "all" 匹配任何 scope
            if scope and "all" not in tool.scopes and scope not in tool.scopes:
                continue
            # category 过滤
            if categories and tool.category not in categories:
                continue
            tools.append(tool)
        return tools

    @classmethod
    def get_tools_for_prompt(
        cls,
        scope: Optional[str] = None,
        categories: Optional[List[str]] = None,
        format: str = "openai_json",
    ) -> Any:
        """生成 prompt 中的工具描述

        Args:
            scope: Agent scope
            categories: Category 过滤
            format: 输出格式
                - "openai_json": OpenAI function calling 格式 (List[Dict])
                - "markdown": Markdown 文本格式 (str)

        Returns:
            openai_json 格式返回 List[Dict]，markdown 格式返回 str
        """
        tools = cls.get_tools(scope=scope, categories=categories)

        if format == "openai_json":
            return cls._format_openai_json(tools)
        elif format == "markdown":
            return cls._format_markdown(tools)
        else:
            raise ValueError(f"未知格式: {format}，支持: openai_json, markdown")

    @classmethod
    def execute(cls, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """同步执行工具

        通过 introspection 自动过滤多余参数，确保只传入函数接受的参数。
        执行后对超长输出自动截断。

        Args:
            name: 工具名称
            arguments: 工具参数

        Returns:
            标准结果字典 {"status", "output", "error_information", ...}
        """
        if name not in cls._tools:
            return {
                "status": "error",
                "output": "",
                "error_information": f"未知工具: {name}，可用: {cls.list_names()}",
            }

        tool = cls._tools[name]
        func = tool.execute

        try:
            filtered_args = cls._filter_func_args(func, arguments)
            result = func(**filtered_args)

            if isinstance(result, dict) and "status" in result:
                return cls._truncate_result(result)
            return {"status": "success", "output": str(result), "error_information": ""}

        except Exception as e:
            logger.error(f"[ToolRegistry] 执行失败 {name}: {e}", exc_info=True)
            return {"status": "error", "output": "", "error_information": str(e)}

    @classmethod
    async def aexecute(cls, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """异步执行工具

        支持异步和同步函数。异步函数会被 await，同步函数直接调用。
        执行后对超长输出自动截断。

        Args:
            name: 工具名称
            arguments: 工具参数

        Returns:
            标准结果字典 {"status", "output", "error_information", ...}
        """
        if name not in cls._tools:
            return {
                "status": "error",
                "output": "",
                "error_information": f"未知工具: {name}，可用: {cls.list_names()}",
            }

        tool = cls._tools[name]
        func = tool.execute

        try:
            filtered_args = cls._filter_func_args(func, arguments)

            if inspect.iscoroutinefunction(func):
                result = await func(**filtered_args)
            else:
                result = func(**filtered_args)

            if isinstance(result, dict) and "status" in result:
                return cls._truncate_result(result)
            return {"status": "success", "output": str(result), "error_information": ""}

        except Exception as e:
            logger.error(f"[ToolRegistry] 异步执行失败 {name}: {e}", exc_info=True)
            return {"status": "error", "output": "", "error_information": str(e)}

    @classmethod
    def list_names(cls, scope: Optional[str] = None) -> List[str]:
        """列出工具名称

        Args:
            scope: 可选的 scope 过滤

        Returns:
            工具名称列表
        """
        if scope is None:
            return list(cls._tools.keys())
        return [t.name for t in cls.get_tools(scope=scope)]

    @classmethod
    def reset(cls) -> None:
        """清空所有注册的工具（仅用于测试）"""
        cls._tools.clear()

    # ==================== 内部方法 ====================

    @staticmethod
    def _truncate_result(result: Dict[str, Any]) -> Dict[str, Any]:
        """对工具返回结果中的超长字段进行截断

        使用 Truncate 基础设施，自动截断 output 和 error_information 字段。
        """
        try:
            from akg_agents.core_v2.tools.truncation import Truncate
            return Truncate.result(result)
        except ImportError:
            # truncation 模块不可用时跳过截断
            return result

    @staticmethod
    def _filter_func_args(func: Callable, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """根据函数签名过滤参数

        避免传入函数不接受的参数（如 workspace_dir 传给不需要它的函数）。
        如果函数接受 **kwargs，则传入所有参数。

        Args:
            func: 目标函数
            arguments: 原始参数字典

        Returns:
            过滤后的参数字典
        """
        sig = inspect.signature(func)
        params = sig.parameters

        # 如果函数接受 **kwargs，传入所有参数
        has_var_keyword = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
        )
        if has_var_keyword:
            return arguments

        # 否则只传入函数接受的参数
        accepted = {name for name in params.keys() if name != "self"}
        return {k: v for k, v in arguments.items() if k in accepted}

    @staticmethod
    def _format_openai_json(tools: List[ToolInfo]) -> List[Dict[str, Any]]:
        """格式化为 OpenAI function calling 格式

        返回格式:
        [
            {
                "type": "function",
                "function": {
                    "name": "tool_name",
                    "description": "...",
                    "parameters": { JSON Schema }
                }
            },
            ...
        ]
        """
        result = []
        for tool in tools:
            result.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                }
            })
        return result

    @staticmethod
    def _format_markdown(tools: List[ToolInfo]) -> str:
        """格式化为 Markdown 文本（用于纯文本 prompt 注入）

        返回格式:
        ### tool_name
        描述...
        参数:
            - param1: 描述 (必填)
            - param2: 描述 (可选)
        """
        lines = []
        for tool in tools:
            params = tool.parameters.get("properties", {})
            required = tool.parameters.get("required", [])

            param_strs = []
            for pname, pinfo in params.items():
                req = " (必填)" if pname in required else " (可选)"
                desc = pinfo.get("description", pinfo.get("type", ""))
                param_strs.append(f"    - {pname}: {desc}{req}")

            lines.append(
                f"### {tool.name}\n{tool.description}\n参数:\n"
                + "\n".join(param_strs)
            )
        return "\n\n".join(lines)
