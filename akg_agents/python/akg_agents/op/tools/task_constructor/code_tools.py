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
            "返回完整的依赖链、建议的 functions 列表、以及需要处理的外部依赖。\n"
            "后续直接将返回的 functions 列表传给 assemble_task，不要逐个 read_function。"
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
            '  选择函数: source_files=[{"path": "workspace/file.py", "functions": ["func1","func2"]}]\n'
            '  排除函数: source_files=[{"path": "workspace/file.py", "exclude_functions": ["unused1"]}]\n'
            "注意: dict 格式时 functions 或 exclude_functions 必须非空，不能传空列表。\n"
            "工具用 AST 解析精确提取/排除函数，自动包含 import。\n"
            "外部依赖内联: 用 helper_code 传入内联的外部函数，"
            "用 imports_code 传入额外的 import 语句。这样生成的文件直接可用，无需手动 edit_file 修复。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "source_files": {
                    "type": "array",
                    "description": (
                        "源文件列表。每个元素可以是:\n"
                        '- 字符串: "workspace/file.py"（完整嵌入整个文件）\n'
                        '- 字典: {"path": "workspace/file.py", "functions": ["f1","f2"]}（选择性提取）\n'
                        '- 字典: {"path": "workspace/file.py", "exclude_functions": ["unused"]}（排除式嵌入）\n'
                        "dict 格式时 functions/exclude_functions 必须是非空列表，不能传 []。"
                    ),
                },
                "imports_code": {
                    "type": "string",
                    "description": (
                        "额外的 import 语句（可选）。会被放在源文件代码之前。"
                        "用于添加源文件中没有但 helper_code 或 Model 需要的 import。"
                    ),
                },
                "model_code": {
                    "type": "string",
                    "description": "Model 类代码。必须包含 class Model(nn.Module) 及其 __init__ 和 forward 方法。",
                },
                "helper_code": {
                    "type": "string",
                    "description": (
                        "辅助代码（可选）。会被放在源文件代码之前、Model 之前。"
                        "用于内联外部依赖函数（如替换 utils.canonicalize_dim 的自定义实现）。"
                        "这样生成的文件直接自包含，无需后续 edit_file 修补。"
                    ),
                },
                "get_inputs_code": {"type": "string", "description": "get_inputs() 函数代码"},
                "get_init_inputs_code": {"type": "string", "description": "get_init_inputs() 函数代码"},
                "output_file": {"type": "string", "description": "输出文件名，默认 task_output.py"},
            },
            "required": ["source_files", "model_code", "get_inputs_code", "get_init_inputs_code"],
        },
        func=assemble_task,
        category="code_analysis",
        scopes=["task_constructor"],
    )

    ToolRegistry.register(
        name="test_with_reference",
        description=(
            "【验证+正确性对比】\n"
            "始终执行: AST 格式检查 + 运行时验证 (实例化, forward, NaN/Inf)。\n"
            "当提供 reference_code 时，额外进行 Model 与 reference_forward 的对比测试，支持多组输入。\n"
            "仅验证时: 不传 reference_code 即可。\n"
            "全部测试通过后应立即调用 finish，不要再修改代码。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "task_file": {"type": "string", "description": "任务文件路径（与 task_code 二选一）"},
                "task_code": {"type": "string", "description": "任务代码字符串（与 task_file 二选一）"},
                "reference_code": {
                    "type": "string",
                    "description": (
                        "Reference 代码（可选）。必须定义 reference_forward(inputs, init_inputs) 函数。\n"
                        "inputs = get_inputs() 的返回值列表，init_inputs = get_init_inputs() 的返回值列表。\n"
                        "示例: def reference_forward(inputs, init_inputs):\\n"
                        "    x, y = inputs\\n    return torch.some_func(x, y)"
                    ),
                },
                "multi_inputs_code": {
                    "type": "string",
                    "description": (
                        "多组测试输入代码（可选）。必须定义 get_multi_test_inputs() 函数，返回列表。\n"
                        "每个元素: {'name': '场景名', 'inputs': [...], 'init_inputs': [...]}。\n"
                        "应覆盖: 不同 shape(小/中/大/超大)、不同 dim、不同 dtype、边界情况。"
                    ),
                },
                "timeout": {"type": "integer", "description": "超时秒数，默认 120"},
            },
            "required": [],
        },
        func=test_with_reference,
        category="code_analysis",
        scopes=["task_constructor"],
    )

    ToolRegistry.register(
        name="optimize_task",
        description="优化清理任务代码：去重 import、移除无用代码、格式化。在 assemble_task 后、test_with_reference 前调用。",
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
