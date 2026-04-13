#!/usr/bin/env python3
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

"""静态代码检查脚本 — 在验证前对生成代码进行快速静态分析。

等价于 akg_agents.op.utils.code_checker.CodeChecker，但作为独立脚本运行，
不依赖 akg_agents 框架。

检查流程：
1. ast.parse 语法检查
2. py_compile 编译检查
3. import 可用性检查
4. 中文文本混入检测
5. DSL 合规性检测（triton 系列：@triton.jit kernel 定义和调用）

用法:
    python code_check.py --code_file <代码文件路径> --backend <backend> --dsl <dsl>

退出码:
    0 — 全部通过
    1 — 发现问题
"""

import argparse
import ast
import importlib.util
import io
import os
import py_compile
import re
import sys
import tempfile
import tokenize
from typing import List, Dict, Optional


# ---------------------------------------------------------------------------
# DSL 合规性检测常量
# ---------------------------------------------------------------------------

_TRITON_DECORATORS = frozenset({"jit", "autotune", "heuristics"})

_TORCH_COMPUTE_OPS_HARD = frozenset({
    "matmul", "mm", "bmm", "addmm", "addmv", "addbmm", "baddbmm",
    "einsum", "dot", "mv", "inner", "outer", "linear",
    "conv1d", "conv2d", "conv3d",
    "conv_transpose1d", "conv_transpose2d", "conv_transpose3d",
    "layer_norm", "batch_norm", "group_norm", "instance_norm",
    "softmax", "log_softmax", "logsumexp",
    "max_pool1d", "max_pool2d", "max_pool3d",
    "avg_pool1d", "avg_pool2d", "avg_pool3d",
    "adaptive_avg_pool2d",
    "embedding", "interpolate", "cumsum", "cumprod",
})

_TORCH_COMPUTE_OPS_SOFT = frozenset({
    "relu", "gelu", "silu", "sigmoid", "tanh",
    "leaky_relu", "elu", "hardswish", "mish",
    "exp", "log", "sqrt", "rsqrt", "pow",
    "sin", "cos", "abs",
    "clamp", "clamp_min", "clamp_max",
    "sum", "mean", "prod", "norm",
    "amax", "amin", "argmax", "argmin",
})

_TORCH_CALL_PREFIXES = frozenset({"torch", "F"})

_CHINESE_RUN_RE = re.compile(r"[\u4e00-\u9fff]{3,}")


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------

def _is_triton_decorator(node: ast.expr) -> bool:
    if isinstance(node, ast.Attribute):
        return (
            isinstance(node.value, ast.Name)
            and node.value.id == "triton"
            and node.attr in _TRITON_DECORATORS
        )
    if isinstance(node, ast.Call):
        return _is_triton_decorator(node.func)
    return False


def _find_model_new_class(tree: ast.Module) -> Optional[ast.ClassDef]:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "ModelNew":
            return node
    return None


def _find_forward(cls_node: ast.ClassDef) -> Optional[ast.FunctionDef]:
    for item in cls_node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == "forward":
            return item
    return None


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_syntax(code: str) -> List[Dict]:
    errors = []
    try:
        ast.parse(code)
    except SyntaxError as e:
        line_num = e.lineno or 0
        code_lines = code.split("\n")
        snippet = code_lines[line_num - 1].rstrip() if 0 < line_num <= len(code_lines) else ""
        error_msg = e.msg or "语法错误"
        if e.offset:
            error_msg += f"（第 {e.offset} 列）"
        errors.append({
            "line": line_num,
            "error_type": "syntax_error",
            "detail": f"Python 语法错误: {error_msg}",
            "suggestion": f"请检查第 {line_num} 行：括号/引号匹配、缩进、关键字拼写、冒号/逗号等",
            "code_snippet": snippet,
        })
    return errors


def check_compile(code: str) -> List[Dict]:
    errors = []
    tmp_src = tmp_pyc = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
            f.write(code)
            tmp_src = f.name
        fd, tmp_pyc = tempfile.mkstemp(suffix=".pyc")
        os.close(fd)
        py_compile.compile(tmp_src, cfile=tmp_pyc, doraise=True)
    except py_compile.PyCompileError as e:
        error_str = str(e)
        match = re.search(r"line (\d+)", error_str)
        line_num = int(match.group(1)) if match else 0
        code_lines = code.split("\n")
        snippet = code_lines[line_num - 1].rstrip() if 0 < line_num <= len(code_lines) else ""
        errors.append({
            "line": line_num,
            "error_type": "compile_error",
            "detail": f"Python 编译错误: {error_str}",
            "suggestion": f"请检查第 {line_num} 行附近的表达式、变量名和 Python 版本兼容性",
            "code_snippet": snippet,
        })
    except Exception:
        pass
    finally:
        for path in (tmp_src, tmp_pyc):
            if path:
                try:
                    os.unlink(path)
                except OSError:
                    pass
    return errors


def check_imports(code: str) -> List[Dict]:
    errors = []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return errors

    checked = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top in checked:
                    continue
                checked.add(top)
                if importlib.util.find_spec(top) is None:
                    errors.append({
                        "line": node.lineno,
                        "error_type": "import_error",
                        "detail": f"模块 '{alias.name}' 无法导入（环境中不存在）",
                        "suggestion": f"检查 '{alias.name}' 拼写或确认是否需要安装",
                        "code_snippet": "",
                    })
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                continue
            if node.module:
                top = node.module.split(".")[0]
                if top in checked:
                    continue
                checked.add(top)
                try:
                    if importlib.util.find_spec(top) is None:
                        errors.append({
                            "line": node.lineno,
                            "error_type": "import_error",
                            "detail": f"模块 '{node.module}' 无法导入（环境中不存在）",
                            "suggestion": f"检查 '{node.module}' 拼写或确认是否需要安装",
                            "code_snippet": "",
                        })
                except (ModuleNotFoundError, ValueError):
                    errors.append({
                        "line": node.lineno,
                        "error_type": "import_error",
                        "detail": f"模块 '{node.module}' 查找失败",
                        "suggestion": f"检查 '{node.module}' 拼写或确认是否需要安装",
                        "code_snippet": "",
                    })
    return errors


def check_stray_chinese(code: str) -> List[Dict]:
    errors = []
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(code).readline))
    except (tokenize.TokenError, IndentationError):
        return errors

    for tok in tokens:
        if tok.type in (tokenize.COMMENT, tokenize.STRING):
            continue
        if tok.type in (tokenize.NEWLINE, tokenize.NL, tokenize.INDENT,
                        tokenize.DEDENT, tokenize.ENDMARKER, tokenize.ENCODING):
            continue
        match = _CHINESE_RUN_RE.search(tok.string)
        if match:
            line_num = tok.start[0]
            errors.append({
                "line": line_num,
                "error_type": "stray_chinese_text",
                "detail": f"代码中混入了中文文本 '{match.group()}'，疑似未注释的中文描述",
                "suggestion": f"第 {line_num} 行包含非代码中文文本，请删除或改为注释",
                "code_snippet": "",
            })
    return errors


def check_dsl_compliance(code: str, dsl: str) -> List[Dict]:
    if not dsl.startswith("triton"):
        return []

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    errors: List[Dict] = []

    # A. 收集 @triton.jit kernel 函数名
    triton_kernels: set = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for dec in node.decorator_list:
                if _is_triton_decorator(dec):
                    triton_kernels.add(node.name)
                    break

    if not triton_kernels:
        errors.append({
            "line": 0,
            "error_type": "no_triton_kernel",
            "detail": (
                f"DSL 指定为 {dsl}，但代码中未找到任何 @triton.jit 装饰的 kernel 函数。"
                f"代码可能使用了 torch 高层 API 替代 triton kernel 实现。"
            ),
            "suggestion": "请确保代码包含至少一个 @triton.jit kernel，并通过 kernel[grid](...) 启动。",
            "code_snippet": "",
        })
        return errors

    # B. 检查 kernel 是否被 kernel[grid](...) 启动
    launched_kernels: set = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Subscript):
            value = node.func.value
            if isinstance(value, ast.Name) and value.id in triton_kernels:
                launched_kernels.add(value.id)

    kernels_not_launched = not launched_kernels
    if kernels_not_launched:
        errors.append({
            "line": 0,
            "error_type": "triton_kernel_not_called",
            "detail": (
                f"定义了 triton kernel {sorted(triton_kernels)}，"
                f"但未找到 kernel[grid](...) 形式的调用。"
            ),
            "suggestion": "请在 ModelNew.forward() 中通过 kernel_name[grid_size](...) 启动 kernel。",
            "code_snippet": "",
        })

    # C. 检查 forward() 中的 torch 高层计算 API
    model_cls = _find_model_new_class(tree)
    if model_cls is None:
        return errors
    forward_node = _find_forward(model_cls)
    if forward_node is None:
        return errors

    hard_calls: list = []
    soft_calls: list = []

    for node in ast.walk(forward_node):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            mod = node.func.value
            method = node.func.attr
            if isinstance(mod, ast.Name) and mod.id in _TORCH_CALL_PREFIXES:
                label = f"{mod.id}.{method}"
                if method in _TORCH_COMPUTE_OPS_HARD:
                    hard_calls.append((node.lineno, label))
                elif method in _TORCH_COMPUTE_OPS_SOFT:
                    soft_calls.append((node.lineno, label))
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
            hard_calls.append((node.lineno, "@ (matmul operator)"))

    def _fmt(calls, limit=5):
        s = ", ".join(f"{name}(第{line}行)" for line, name in calls[:limit])
        if len(calls) > limit:
            s += f" 等（共 {len(calls)} 处）"
        return s

    if hard_calls:
        errors.append({
            "line": hard_calls[0][0],
            "error_type": "torch_api_instead_of_kernel",
            "detail": (
                f"forward() 中使用了 {len(hard_calls)} 个不允许的 torch 高层计算 API: {_fmt(hard_calls)}。"
                f"这些操作必须在 DSL kernel 内实现。"
            ),
            "suggestion": "将这些核心计算移入 triton kernel，forward() 仅负责准备输入、启动 kernel 和返回输出。",
            "code_snippet": "",
        })

    if soft_calls and kernels_not_launched:
        errors.append({
            "line": soft_calls[0][0],
            "error_type": "torch_api_without_kernel",
            "detail": (
                f"forward() 使用了 {len(soft_calls)} 个 torch 计算 API: {_fmt(soft_calls)}，"
                f"同时 triton kernel 未被调用，代码很可能用 torch API 替代了 kernel 实现。"
            ),
            "suggestion": "请用 triton kernel 实现核心计算逻辑。",
            "code_snippet": "",
        })

    return errors


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def format_report(errors: List[Dict], code_lines: Optional[List[str]] = None) -> str:
    if not errors:
        return ""

    lines = [
        "## 静态代码检查报告",
        "",
        f"**发现 {len(errors)} 个问题，请修复后重新生成代码：**",
        "",
    ]

    for i, err in enumerate(errors, 1):
        eline = err["line"]
        lines.append(f"### 问题 {i}: 第 {eline} 行 [{err.get('error_type', 'unknown')}]")
        lines.append(f"  {err['detail']}")

        if code_lines is not None and eline > 0:
            start = max(1, eline - 3)
            end = min(len(code_lines), eline + 3)
            lines.append(f"  上下文（第 {start}-{end} 行）：")
            for n in range(start, end + 1):
                prefix = "  >>> " if n == eline else "      "
                lines.append(f"{prefix}{n:4d} | {code_lines[n - 1]}")

        elif err.get("code_snippet"):
            lines.append(f"  出错代码: {err['code_snippet']}")

        if err.get("suggestion"):
            lines.append(f"  建议：{err['suggestion']}")
        lines.append("")

    lines.append("**注意：语法检查每次只能定位到首个错误，修复后可能还有后续问题。**")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="静态代码检查（等价于 CodeChecker）")
    parser.add_argument("--code_file", required=True, help="待检查的代码文件路径")
    parser.add_argument("--backend", default="", help="后端（cuda / ascend / cpu）")
    parser.add_argument("--dsl", default="", help="DSL（triton_cuda / triton_ascend 等）")
    args = parser.parse_args()

    if not os.path.isfile(args.code_file):
        print(f"错误：文件不存在 {args.code_file}", file=sys.stderr)
        sys.exit(1)

    with open(args.code_file, "r", encoding="utf-8") as f:
        code = f.read()

    if not code.strip():
        print("错误：代码文件为空", file=sys.stderr)
        sys.exit(1)

    dsl = args.dsl.lower()

    # 依次执行 5 项检查
    errors: List[Dict] = []

    # 1. 语法检查
    errors.extend(check_syntax(code))

    # 2. 编译检查（语法通过后）
    if not errors:
        errors.extend(check_compile(code))

    # 3. import 可用性（编译通过后）
    if not errors:
        errors.extend(check_imports(code))

    # 4. 中文混入检测（始终执行）
    errors.extend(check_stray_chinese(code))

    # 5. DSL 合规性（语法正确时）
    has_syntax_err = any(e.get("error_type") in ("syntax_error", "compile_error") for e in errors)
    if not has_syntax_err:
        errors.extend(check_dsl_compliance(code, dsl))

    if not errors:
        print("静态检查通过")
        sys.exit(0)
    else:
        code_lines = code.split("\n")
        report = format_report(errors, code_lines)
        print(report)
        sys.exit(1)


if __name__ == "__main__":
    main()
