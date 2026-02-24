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
任务构造器代码操作工具 - 入口模块

将工具实现拆分到子模块中，此文件负责工具注册。
子模块:
  - path_utils: 共享路径解析
  - ast_utils: AST 解析（函数提取、依赖追踪）
  - code_cleanup: 代码清理（import 去重、私有模块移除）
  - assembly: 任务装配、优化、验证
  - execution: 代码执行、对比测试（函数保留，不再注册为工具）

注意: apply_patch 已由 core_v2 的 edit_file 替代，
      run_code 已由 core_v2 的 execute_script（code 参数）替代。
"""

from akg_agents.core_v2.tools.tool_registry import ToolRegistry
from akg_agents.op.tools.task_constructor.assembly import (
    trace_dependencies,
    assemble_task,
    validate_task,
    optimize_task,
)
from akg_agents.op.tools.task_constructor.execution import (
    test_with_reference,
)


def _register_all():
    """注册 TaskConstructor 代码操作工具到统一 ToolRegistry"""

    ToolRegistry.register(
        name="trace_dependencies",
        description=(
            "【依赖追踪】给定入口函数名，自动找出同文件内所有被调用的函数。\n"
            "通过分析文件 import 自动识别需要内联的外部模块调用，附带来源模块路径。\n"
            "返回完整的依赖链、建议的 functions 列表、以及需要处理的外部依赖。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "workspace 中的文件路径"},
                "entry_functions": {
                    "type": "array", "items": {"type": "string"},
                    "description": "入口函数名列表"
                },
            },
            "required": ["file_path", "entry_functions"],
        },
        func=trace_dependencies,
        category="code_analysis",
        scopes=["task_constructor"],
    )

    ToolRegistry.register(
        name="assemble_task",
        description=(
            "【选择性拼装】从 workspace 源文件中提取指定函数 + Model/get_inputs → 自包含任务文件。\n"
            "三种用法:\n"
            '  完整嵌入: source_files=["workspace/file.py"]\n'
            '  选择函数: source_files=[{"path": "workspace/file.py", "functions": ["func1"]}]\n'
            '  排除函数: source_files=[{"path": "workspace/file.py", "exclude_functions": ["unused1"]}]\n'
            "工具用 AST 解析精确提取/排除函数，自动包含 import。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "source_files": {"type": "array", "description": "源文件列表"},
                "imports_code": {"type": "string", "description": "额外import（可选）"},
                "model_code": {"type": "string", "description": "Model 类代码"},
                "helper_code": {"type": "string", "description": "辅助代码"},
                "get_inputs_code": {"type": "string", "description": "get_inputs() 函数"},
                "get_init_inputs_code": {"type": "string", "description": "get_init_inputs() 函数"},
                "output_file": {"type": "string", "description": "输出文件名"},
            },
            "required": ["source_files", "model_code", "get_inputs_code", "get_init_inputs_code"],
        },
        func=assemble_task,
        category="code_analysis",
        scopes=["task_constructor"],
    )

    ToolRegistry.register(
        name="validate_task",
        description="预运行验证任务代码。检查: 实例化 -> forward -> NaN/Inf -> 一致性。",
        parameters={
            "type": "object",
            "properties": {
                "task_code": {"type": "string", "description": "任务代码"},
                "task_file": {"type": "string", "description": "任务文件路径"},
                "timeout": {"type": "integer", "description": "超时秒数"},
            },
            "required": [],
        },
        func=validate_task,
        category="code_analysis",
        scopes=["task_constructor"],
    )

    ToolRegistry.register(
        name="test_with_reference",
        description="【正确性验证】将 Model 输出与 reference 函数对比，支持多组输入。",
        parameters={
            "type": "object",
            "properties": {
                "task_file": {"type": "string", "description": "任务文件路径"},
                "task_code": {"type": "string", "description": "任务代码"},
                "reference_code": {"type": "string", "description": "Reference 代码"},
                "multi_inputs_code": {"type": "string", "description": "多组输入代码"},
                "timeout": {"type": "integer", "description": "超时秒数"},
            },
            "required": ["reference_code"],
        },
        func=test_with_reference,
        category="code_analysis",
        scopes=["task_constructor"],
    )

    ToolRegistry.register(
        name="optimize_task",
        description="优化清理任务代码：去重 import、移除无用代码、格式化。在 validate 通过后、finish 前调用。",
        parameters={
            "type": "object",
            "properties": {
                "task_file": {"type": "string", "description": "任务文件路径"},
                "task_code": {"type": "string", "description": "任务代码"},
            },
            "required": [],
        },
        func=optimize_task,
        category="code_analysis",
        scopes=["task_constructor"],
    )


_register_all()
