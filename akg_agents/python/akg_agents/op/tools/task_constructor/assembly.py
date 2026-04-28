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

"""任务装配、优化、验证工具"""

import ast
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Any, List

from akg_agents.op.tools.task_constructor.path_utils import resolve_path
from akg_agents.op.tools.task_constructor.ast_utils import trace_function_deps
from akg_agents.op.tools.task_constructor.code_cleanup import (
    cleanup_task_code,
    merge_from_imports_text,
    extract_functions_from_file,
)


def trace_dependencies(
    file_path: str,
    entry_functions: List[str],
    workspace_dir: str = "",
    output_dir: str = "",
) -> Dict[str, Any]:
    """
    AST 依赖追踪：给定文件路径 + 入口函数名，自动找出所有被调用的同文件函数。
    通过分析文件 import 自动识别需要内联的外部模块调用，附带来源模块路径。
    """
    if not file_path:
        return {"status": "error", "output": "", "error_information": "需要 file_path"}
    if not entry_functions:
        return {"status": "error", "output": "", "error_information": "需要 entry_functions 列表"}

    ws = Path(workspace_dir) if workspace_dir else None
    od = Path(output_dir) if output_dir else None
    path = resolve_path(file_path, workspace_dir=ws, output_dir=od)
    if not path.exists():
        return {"status": "error", "output": "", "error_information": f"文件不存在: {path}"}

    try:
        content = path.read_text(encoding="utf-8")
        tree = ast.parse(content)
    except Exception as e:
        return {"status": "error", "output": "", "error_information": f"解析失败: {e}"}

    full_deps, external_calls = trace_function_deps(content, tree, entry_functions)
    added = [n for n in full_deps if n not in entry_functions]

    output = f"入口函数: {entry_functions}\n"
    output += f"完整依赖链 ({len(full_deps)} 个函数):\n"
    for name in full_deps:
        marker = " <- 入口" if name in entry_functions else " <- 自动发现"
        output += f"  - {name}{marker}\n"

    if added:
        output += f"\n自动发现的额外依赖: {added}\n"
        output += "建议在 assemble_task 的 functions 中包含这些函数。\n"
    else:
        output += "\n无额外同文件依赖。\n"

    if external_calls:
        output += f"\n[!] 需要内联的外部调用 ({len(external_calls)} 个, 来自内部模块):\n"
        for call_str, source_module in external_calls:
            output += f"  - {call_str}  (来源: {source_module})\n"
        output += (
            "\n处理方式:\n"
            "  1. 用 read_function 在来源模块中查看原始函数签名和实现\n"
            "  2. 在 helper_code 中内联等价实现，参数签名必须与原始完全一致\n"
            "  3. 对于模块别名（如 aten = torch._ops.ops.aten），在 helper_code 中定义"
        )

    return {"status": "success", "output": output, "error_information": ""}


def assemble_task(
    source_files: List = None,
    imports_code: str = "",
    model_code: str = "",
    helper_code: str = "",
    get_inputs_code: str = "",
    get_init_inputs_code: str = "",
    output_file: str = "task_output.py",
    workspace_dir: str = "",
    output_dir: str = "",
) -> Dict[str, Any]:
    """
    核心工具：拼装标准化任务文件。

    支持三种模式：
    1. 完整嵌入: source_files=["workspace/file.py"]
    2. 选择性提取: source_files=[{"path": "...", "functions": ["f1", "f2"]}]
    3. 排除模式: source_files=[{"path": "...", "exclude_functions": ["unused"]}]
    """
    if source_files is None:
        source_files = []

    ws = Path(workspace_dir) if workspace_dir else None
    od = Path(output_dir) if output_dir else None

    parts = []
    headers_seen = set()
    warnings = []

    if imports_code.strip():
        parts.append(f"# ===== Imports =====\n{imports_code.strip()}")

    if helper_code.strip():
        parts.append(f"# ===== Helper =====\n{helper_code.strip()}")

    for sf in source_files:
        if isinstance(sf, str):
            sf_path = resolve_path(sf, workspace_dir=ws, output_dir=od)
            if not sf_path.exists():
                return {"status": "error", "output": "",
                        "error_information": f"源文件不存在: {sf} (resolved: {sf_path})"}
            try:
                content = sf_path.read_text(encoding="utf-8")
                parts.append(f"# ===== Source: {sf_path.name} =====\n{content}")
            except Exception as e:
                return {"status": "error", "output": "", "error_information": f"读取 {sf} 失败: {e}"}

        elif isinstance(sf, dict):
            sf_path_str = sf.get("path", "")
            func_names = sf.get("functions", [])
            exclude_names = sf.get("exclude_functions", [])
            include_header = sf.get("include_header", True)

            if not sf_path_str:
                return {"status": "error", "output": "",
                        "error_information": "source_files dict 需要 'path' 字段"}

            sf_path = resolve_path(sf_path_str, workspace_dir=ws, output_dir=od)
            if not sf_path.exists():
                return {"status": "error", "output": "",
                        "error_information": f"源文件不存在: {sf_path_str} (resolved: {sf_path})"}

            if exclude_names and not func_names:
                try:
                    content = sf_path.read_text(encoding="utf-8")
                    tree = ast.parse(content)
                except Exception as e:
                    return {"status": "error", "output": "",
                            "error_information": f"解析 {sf_path_str} 失败: {e}"}

                lines = content.splitlines()
                exclude_set = set(exclude_names)
                remove_ranges = []
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        if node.name in exclude_set:
                            start = node.lineno - 1
                            if node.decorator_list:
                                start = min(start, min(d.lineno - 1 for d in node.decorator_list))
                            end = node.end_lineno
                            remove_ranges.append((start, end))

                if remove_ranges:
                    remove_ranges.sort()
                    keep_lines = []
                    skip_until = 0
                    for i, line in enumerate(lines):
                        if i < skip_until:
                            continue
                        removed = False
                        for rstart, rend in remove_ranges:
                            if rstart <= i < rend:
                                skip_until = rend
                                removed = True
                                break
                        if not removed:
                            keep_lines.append(line)
                    content = "\n".join(keep_lines)
                    warnings.append(
                        f"已从 {sf_path.name} 排除 {len(remove_ranges)} 个函数: {exclude_names}"
                    )

                parts.append(
                    f"# ===== Source: {sf_path.name} (excluded: {', '.join(exclude_names)}) =====\n{content}"
                )

            elif func_names:
                try:
                    header, funcs_code, not_found = extract_functions_from_file(sf_path, func_names)
                except Exception as e:
                    return {"status": "error", "output": "", "error_information": f"提取函数失败: {e}"}

                if not_found:
                    warnings.append(f"未找到函数: {not_found} (in {sf_path.name})")

                source_parts = []
                if include_header and header and sf_path.name not in headers_seen:
                    source_parts.append(header)
                    headers_seen.add(sf_path.name)
                if funcs_code:
                    source_parts.append(funcs_code)

                if source_parts:
                    label = ", ".join(func_names[:5])
                    if len(func_names) > 5:
                        label += f" ... ({len(func_names)} total)"
                    parts.append(
                        f"# ===== From {sf_path.name}: {label} =====\n"
                        + "\n\n".join(source_parts)
                    )
            else:
                return {"status": "error", "output": "",
                        "error_information": f"{sf_path_str}: dict 格式需要 'functions' 或 'exclude_functions' 字段"}
        else:
            return {"status": "error", "output": "",
                    "error_information": f"source_files 元素格式错误: {type(sf)}"}

    if model_code.strip():
        parts.append(f"\n# ===== Model =====\n{model_code}")

    if get_inputs_code.strip():
        parts.append(f"\n# ===== Test inputs =====\n{get_inputs_code}")

    if get_init_inputs_code.strip():
        parts.append(f"\n{get_init_inputs_code}")

    full_code = "\n\n".join(parts)

    # 自动清理：去重import、移除内部私有模块引用、清理元数据
    full_code = cleanup_task_code(full_code)

    out_path = Path(output_file).expanduser()
    if not out_path.is_absolute():
        if od:
            out_path = od / output_file
        else:
            out_path = Path.cwd() / output_file
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(full_code, encoding="utf-8")

    total_lines = full_code.count("\n") + 1
    output_msg = f"[任务文件已生成] {out_path}\n总行数: {total_lines}"
    if warnings:
        output_msg += "\n警告: " + "; ".join(warnings)

    return {"status": "success", "output": output_msg, "error_information": ""}


def validate_task(
    task_code: str = "",
    task_file: str = "",
    timeout: int = 60,
    workspace_dir: str = "",
    output_dir: str = "",
) -> Dict[str, Any]:
    """预运行验证标准化格式的任务代码"""
    ws = Path(workspace_dir) if workspace_dir else None
    od = Path(output_dir) if output_dir else None

    if task_file and not task_code:
        path = resolve_path(task_file, workspace_dir=ws, output_dir=od)
        if path.exists():
            task_code = path.read_text(encoding="utf-8")
        else:
            return {"status": "error", "output": "",
                    "error_information": f"任务文件不存在: {path}"}

    if not task_code.strip():
        return {"status": "error", "output": "", "error_information": "task_code 不能为空"}

    return _run_validation(task_code, timeout=timeout)


def _run_validation(task_code: str, timeout: int = 60) -> Dict[str, Any]:
    """内联验证：AST 检查 + 运行时检查"""
    issues = []

    try:
        tree = ast.parse(task_code)
    except SyntaxError as e:
        return {"status": "error", "output": "",
                "error_information": f"语法错误: {e}"}

    has_model = False
    has_forward = False
    has_get_inputs = False
    has_get_init_inputs = False

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Model":
            has_model = True
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if item.name == "forward":
                        has_forward = True
        elif isinstance(node, ast.FunctionDef):
            if node.name == "get_inputs":
                has_get_inputs = True
            elif node.name == "get_init_inputs":
                has_get_init_inputs = True

    if not has_model:
        issues.append("缺少 class Model(nn.Module)")
    if not has_forward:
        issues.append("缺少 Model.forward() 方法")
    if not has_get_inputs:
        issues.append("缺少 get_inputs() 函数")
    if not has_get_init_inputs:
        issues.append("缺少 get_init_inputs() 函数")

    if issues:
        return {"status": "error", "output": "",
                "error_information": "格式问题: " + "; ".join(issues)}

    test_script = task_code + "\n\n" + (
        "import torch\n"
        "try:\n"
        "    init_inputs = get_init_inputs()\n"
        "    model = Model(*init_inputs)\n"
        "    inputs = get_inputs()\n"
        "    output = model.forward(*inputs)\n"
        "    if isinstance(output, torch.Tensor):\n"
        "        has_nan = torch.isnan(output).any().item()\n"
        "        has_inf = torch.isinf(output).any().item()\n"
        "        print(f'shape={output.shape}, dtype={output.dtype}, nan={has_nan}, inf={has_inf}')\n"
        "        if has_nan: print('WARNING: output contains NaN')\n"
        "        if has_inf: print('WARNING: output contains Inf')\n"
        "    elif isinstance(output, (tuple, list)):\n"
        "        for i, t in enumerate(output):\n"
        "            if isinstance(t, torch.Tensor):\n"
        "                print(f'output[{i}]: shape={t.shape}, dtype={t.dtype}')\n"
        "    print('VALIDATION_OK')\n"
        "except Exception as e:\n"
        "    print(f'VALIDATION_ERROR: {type(e).__name__}: {e}')\n"
    )

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False, encoding="utf-8") as f:
            f.write(test_script)
            tmp_path = f.name

        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True, text=True, timeout=timeout,
        )
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()

        if "VALIDATION_OK" in stdout:
            warn_parts = []
            if "WARNING: output contains NaN" in stdout:
                warn_parts.append("输出包含 NaN")
            if "WARNING: output contains Inf" in stdout:
                warn_parts.append("输出包含 Inf")
            output_msg = f"验证通过\n{stdout}"
            if warn_parts:
                output_msg += f"\n警告: {'; '.join(warn_parts)}"
            return {"status": "success", "output": output_msg, "error_information": ""}
        elif "VALIDATION_ERROR" in stdout:
            error_line = [line for line in stdout.splitlines() if "VALIDATION_ERROR" in line]
            return {"status": "error", "output": stdout,
                    "error_information": error_line[0] if error_line else "运行时验证失败"}
        else:
            combined = stdout + ("\n" + stderr if stderr else "")
            return {"status": "error", "output": combined,
                    "error_information": f"运行时错误 (exit code {result.returncode})"}
    except subprocess.TimeoutExpired:
        return {"status": "error", "output": "",
                "error_information": f"验证超时 ({timeout}s)"}
    except Exception as e:
        return {"status": "error", "output": "", "error_information": str(e)}
    finally:
        if tmp_path:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass


def optimize_task(
    task_file: str = "",
    task_code: str = "",
    workspace_dir: str = "",
    output_dir: str = "",
) -> Dict[str, Any]:
    """
    优化清理任务代码：去重 import、移除无用代码、格式化。
    在 validate_task 通过后调用，优化后应再次 validate_task 确认。
    """
    ws = Path(workspace_dir) if workspace_dir else None
    od = Path(output_dir) if output_dir else None
    task_path = None

    if task_file and not task_code:
        task_path = resolve_path(task_file, workspace_dir=ws, output_dir=od)
        if task_path.exists():
            task_code = task_path.read_text(encoding="utf-8")
        else:
            return {"status": "error", "output": "",
                    "error_information": f"任务文件不存在: {task_path}"}

    if not task_code.strip():
        return {"status": "error", "output": "", "error_information": "task_code 不能为空"}

    original_lines = task_code.count('\n') + 1

    optimized = cleanup_task_code(task_code)
    optimized = merge_from_imports_text(optimized)

    try:
        ast.parse(optimized)
    except SyntaxError as e:
        return {"status": "error", "output": "",
                "error_information": f"优化后代码存在语法错误: {e}，未修改原文件"}

    new_lines = optimized.count('\n') + 1

    if task_path:
        task_path.write_text(optimized, encoding="utf-8")
        output_msg = f"[已优化] {task_path}\n行数: {original_lines} → {new_lines} (减少 {original_lines - new_lines} 行)"
    else:
        output_msg = f"[已优化] 行数: {original_lines} → {new_lines}"

    return {"status": "success", "output": output_msg, "error_information": "",
            "optimized_code": optimized if not task_path else ""}
