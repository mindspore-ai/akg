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
CodeChecker: 代码检查器

纯静态检查流程（不调用 LLM）：
1. ast.parse 语法检查
2. py_compile 编译检查
3. import 可用性检查
4. 中文文本混入检测
5. DSL 合规性检测（反作弊：确保代码真正使用了指定的 DSL）
"""

import re
import ast
import logging
import os
import py_compile
import importlib.util
import tempfile
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DSL 合规性检测 —— 常量与辅助函数
# ---------------------------------------------------------------------------

_TRITON_DECORATORS = frozenset({'jit', 'autotune', 'heuristics'})

# --- 分层黑名单 ---
# HARD: 高级复杂 API，无论 kernel 是否调用都硬失败（不应出现在 forward() 中）
# SOFT: 简单元素级/辅助操作，仅在 kernel 未调用时硬失败，已调用时仅警告
#       （融合算子的 pre/post 处理可能合理使用这些）

_TORCH_COMPUTE_OPS_HARD = frozenset({
    # Matrix / linear algebra — 这些是核心计算，必须由 kernel 完成
    'matmul', 'mm', 'bmm', 'addmm', 'addmv', 'addbmm', 'baddbmm',
    'einsum', 'dot', 'mv', 'inner', 'outer',
    'linear',
    # Convolution
    'conv1d', 'conv2d', 'conv3d',
    'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d',
    # Normalization — 整个 norm 语义应在 kernel 内实现
    'layer_norm', 'batch_norm', 'group_norm', 'instance_norm',
    # Compound activations — softmax 含 reduction + exp，属于完整子计算图
    'softmax', 'log_softmax', 'logsumexp',
    # Pooling
    'max_pool1d', 'max_pool2d', 'max_pool3d',
    'avg_pool1d', 'avg_pool2d', 'avg_pool3d',
    'adaptive_avg_pool2d',
    # Other complex ops
    'embedding', 'interpolate',
    'cumsum', 'cumprod',
})

_TORCH_COMPUTE_OPS_SOFT = frozenset({
    # Simple activations — 融合算子可能在 kernel 外做 pre/post 激活
    'relu', 'gelu', 'silu', 'sigmoid', 'tanh',
    'leaky_relu', 'elu', 'hardswish', 'mish',
    # Simple elementwise math — pre/post 处理可能合理使用
    'exp', 'log', 'sqrt', 'rsqrt', 'pow',
    'sin', 'cos', 'abs',
    'clamp', 'clamp_min', 'clamp_max',
    # Simple reductions — 可能用于 grid 计算或后处理
    'sum', 'mean', 'prod', 'norm',
    'amax', 'amin', 'argmax', 'argmin',
})

_TORCH_COMPUTE_OPS = _TORCH_COMPUTE_OPS_HARD | _TORCH_COMPUTE_OPS_SOFT

_TORCH_CALL_PREFIXES = frozenset({'torch', 'F'})


def _is_triton_decorator(node: ast.expr) -> bool:
    """判断 decorator 节点是否为 @triton.jit / @triton.autotune 等"""
    if isinstance(node, ast.Attribute):
        return (
            isinstance(node.value, ast.Name)
            and node.value.id == 'triton'
            and node.attr in _TRITON_DECORATORS
        )
    if isinstance(node, ast.Call):
        return _is_triton_decorator(node.func)
    return False


def _find_model_new_class(tree: ast.Module) -> Optional[ast.ClassDef]:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == 'ModelNew':
            return node
    return None


def _find_forward(cls_node: ast.ClassDef) -> Optional[ast.FunctionDef]:
    for item in cls_node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == 'forward':
            return item
    return None


@dataclass
class CheckError:
    """检查错误信息"""
    line: int
    error_type: str
    detail: str
    suggestion: str
    code_snippet: str


class CodeChecker:
    """
    代码检查器：在 Coder 生成代码后、Verifier 验证前，进行快速的纯静态检查

    检查流程：ast.parse → py_compile → import 验证 → 中文文本混入检测
    不调用 LLM，零额外成本。
    """

    def __init__(self, backend: str, dsl: str, config: Optional[dict] = None):
        self.backend = backend.lower() if backend else ""
        self.dsl = dsl.lower() if dsl else ""
        self.config = config or {}
        logger.info(f"CodeChecker initialized: backend={self.backend}, dsl={self.dsl}")

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    async def check(self, code: str, task_info: Optional[dict] = None) -> Tuple[bool, str, List[Dict]]:
        """
        检查代码（纯静态检查，不调用 LLM）

        检查流程：
        1. ast.parse 语法检查
        2. py_compile 编译检查（语法通过后执行，捕获额外编译问题）
        3. import 可用性检查（代码可编译时执行）
        4. 中文文本混入检测（独立于语法检查，始终执行）

        Args:
            code: 要检查的代码
            task_info: 任务信息（保留参数以兼容接口）

        Returns:
            Tuple[bool, str, List[Dict]]:
                - passed: 是否通过检查
                - error_message: 格式化的错误信息（用于传递给 Coder）
                - errors: 详细错误列表
        """
        if not code or not code.strip():
            logger.warning("CodeChecker: Empty code provided")
            empty_err = {
                "line": 0,
                "error_type": "empty_code",
                "detail": "代码为空，无法进行检查",
                "suggestion": "请生成有效的代码",
                "code_snippet": ""
            }
            return False, self._format_errors([empty_err]), [empty_err]

        # Step 1: Python 语法检查（ast.parse）
        errors = self._check_python_syntax(code)

        # Step 2: py_compile 编译检查（仅在语法检查通过时执行）
        if not errors:
            errors.extend(self._check_py_compile(code))

        # Step 3: import 可用性检查（仅在代码可编译时执行）
        if not errors:
            errors.extend(self._check_imports(code))

        # Step 4: 中文文本混入检测（独立于语法检查，始终执行）
        errors.extend(self._check_stray_chinese(code))

        # Step 5: DSL 合规性检测（仅在语法正确时执行，需要 AST 解析）
        has_syntax_err = any(
            e.get('error_type') in ('syntax_error', 'compile_error') for e in errors
        )
        if not has_syntax_err:
            errors.extend(self._check_dsl_compliance(code))

        # Step 6: Autotune 规范检测（仅 triton DSL，语法正确时执行）
        if not has_syntax_err and self.dsl.startswith("triton"):
            errors.extend(self._check_autotune_compliance(code))

        passed = len(errors) == 0
        code_lines = code.split('\n')
        error_message = self._format_errors(errors, code_lines) if errors else ""

        if errors:
            logger.warning(f"CodeChecker: Found {len(errors)} issue(s)")
            for err in errors:
                logger.warning(f"  Line {err['line']}: {err['detail']}")
        else:
            logger.info("CodeChecker: All checks passed")

        return passed, error_message, errors

    # ------------------------------------------------------------------
    # Step 1: ast.parse 语法检查
    # ------------------------------------------------------------------

    def _check_python_syntax(self, code: str) -> List[Dict]:
        """
        使用 ast.parse() 进行语法检查：
        括号不匹配、缩进错误、关键字拼写等。

        注意：ast.parse 遇到第一个 SyntaxError 就会停止，
        因此这里只返回首个错误，后续可能还有其他问题需要在修复后再次检查。
        """
        errors = []
        try:
            ast.parse(code)
        except SyntaxError as e:
            line_num = e.lineno or 0
            code_lines = code.split('\n')
            code_snippet = ""
            if 0 < line_num <= len(code_lines):
                code_snippet = code_lines[line_num - 1].rstrip()

            error_msg = e.msg or "语法错误"
            if e.offset:
                error_msg += f"（第 {e.offset} 列）"

            errors.append({
                "line": line_num,
                "error_type": "syntax_error",
                "detail": f"Python 语法错误: {error_msg}",
                "suggestion": f"""请检查第 {line_num} 行的语法：
  - 检查括号、引号是否匹配
  - 检查缩进是否正确
  - 检查关键字拼写是否正确
  - 检查冒号、逗号等符号是否遗漏""",
                "code_snippet": code_snippet
            })
            logger.warning(f"CodeChecker: Python syntax error at line {line_num}: {error_msg}")

        return errors

    # ------------------------------------------------------------------
    # Step 2: py_compile 编译检查
    # ------------------------------------------------------------------

    def _check_py_compile(self, code: str) -> List[Dict]:
        """
        使用 py_compile 进行编译级别检查。
        比 ast.parse 更严格，能捕获部分 ast.parse 遗漏的编译问题
        （如 SyntaxWarning 升级、重复关键字参数等）。
        """
        errors = []
        tmp_src = None
        tmp_pyc = None
        try:
            with tempfile.NamedTemporaryFile(
                mode='w', suffix='.py', delete=False, encoding='utf-8'
            ) as f:
                f.write(code)
                tmp_src = f.name

            # 临时文件写入系统临时目录（Linux: /tmp, Windows: %TEMP%），不在当前工作目录。
            # 用独立的临时文件接收 .pyc 输出，避免往 __pycache__ 写入导致权限问题。
            fd, tmp_pyc = tempfile.mkstemp(suffix='.pyc')
            os.close(fd)

            py_compile.compile(tmp_src, cfile=tmp_pyc, doraise=True)
        except py_compile.PyCompileError as e:
            line_num = 0
            error_str = str(e)
            match = re.search(r'line (\d+)', error_str)
            if match:
                line_num = int(match.group(1))

            code_lines = code.split('\n')
            code_snippet = ""
            if 0 < line_num <= len(code_lines):
                code_snippet = code_lines[line_num - 1].rstrip()

            errors.append({
                "line": line_num,
                "error_type": "compile_error",
                "detail": f"Python 编译错误: {error_str}",
                "suggestion": f"""请检查第 {line_num} 行附近的代码：
  - 检查是否有不合法的表达式或语法结构
  - 检查变量名、函数名是否合法
  - 检查是否有 Python 版本不兼容的写法""",
                "code_snippet": code_snippet
            })
            logger.warning(f"CodeChecker: py_compile error at line {line_num}: {error_str}")
        except Exception as e:
            logger.warning(f"CodeChecker: py_compile check failed unexpectedly: {e}")
        finally:
            for path in (tmp_src, tmp_pyc):
                if path:
                    try:
                        os.unlink(path)
                    except OSError:
                        pass

        return errors

    # ------------------------------------------------------------------
    # Step 3: import 可用性检查
    # ------------------------------------------------------------------

    def _check_imports(self, code: str) -> List[Dict]:
        """
        检查代码中 import 语句引用的模块是否可用。

        通过 AST 提取所有 import / from ... import 语句，
        使用 importlib.util.find_spec 验证顶层模块是否存在。
        """
        errors = []
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return errors

        checked = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top_module = alias.name.split('.')[0]
                    if top_module in checked:
                        continue
                    checked.add(top_module)
                    if not self._is_module_available(top_module):
                        errors.append({
                            "line": node.lineno,
                            "error_type": "import_error",
                            "detail": f"模块 '{alias.name}' 无法导入（环境中不存在此模块）",
                            "suggestion": f"请检查模块名 '{alias.name}' 是否拼写正确，或确认该模块是否需要安装",
                            "code_snippet": ""
                        })
                        logger.warning(
                            f"CodeChecker: import error at line {node.lineno}: "
                            f"module '{alias.name}' not found"
                        )

            elif isinstance(node, ast.ImportFrom):
                if node.level and node.level > 0:
                    continue
                if node.module:
                    top_module = node.module.split('.')[0]
                    if top_module in checked:
                        continue
                    checked.add(top_module)
                    if not self._is_module_available(top_module):
                        errors.append({
                            "line": node.lineno,
                            "error_type": "import_error",
                            "detail": f"模块 '{node.module}' 无法导入（环境中不存在此模块）",
                            "suggestion": f"请检查模块名 '{node.module}' 是否拼写正确，或确认该模块是否需要安装",
                            "code_snippet": ""
                        })
                        logger.warning(
                            f"CodeChecker: import error at line {node.lineno}: "
                            f"module '{node.module}' not found"
                        )

        return errors

    @staticmethod
    def _is_module_available(module_name: str) -> bool:
        """检查模块在当前环境中是否可用"""
        try:
            return importlib.util.find_spec(module_name) is not None
        except (ModuleNotFoundError, ValueError):
            return False

    # ------------------------------------------------------------------
    # Step 4: 中文文本混入检测
    # ------------------------------------------------------------------

    _CHINESE_RUN_RE = re.compile(r'[\u4e00-\u9fff]{3,}')

    def _check_stray_chinese(self, code: str) -> List[Dict]:
        """
        检测代码中混入的中文文本（LLM 常见问题）。

        规则：连续 >=3 个汉字出现在注释和字符串之外，视为误混入的中文描述。
        通过 tokenize 精确剥离注释和字符串，只扫描真正的代码 token。
        """
        import io
        import tokenize

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

            match = self._CHINESE_RUN_RE.search(tok.string)
            if match:
                line_num = tok.start[0]
                chinese_text = match.group()
                errors.append({
                    "line": line_num,
                    "error_type": "stray_chinese_text",
                    "detail": f"代码中混入了中文文本 '{chinese_text}'，疑似未注释的中文描述",
                    "suggestion": (
                        f"第 {line_num} 行包含非代码的中文文本，请删除或改为注释（在行首加 #）。"
                        f"如果是有意使用的中文变量名，请忽略此警告。"
                    ),
                    "code_snippet": ""
                })
                logger.warning(
                    f"CodeChecker: stray Chinese text at line {line_num}: '{chinese_text}'"
                )

        return errors

    # ------------------------------------------------------------------
    # Step 5: DSL 合规性检测（反作弊）
    # ------------------------------------------------------------------

    def _check_dsl_compliance(self, code: str) -> List[Dict]:
        """
        检测生成代码是否真正使用了指定的 DSL 实现（纯 AST 静态分析）。

        仅对 triton 系列 DSL 生效（triton_cuda / triton_ascend），检查三类问题：

        A. 没有定义任何 @triton.jit kernel            → 硬失败
        B. 定义了 kernel 但没有通过 kernel[grid](...) 调用 → 硬失败
        C. forward() 中使用了 torch 高层计算 API          → 结合 B 判断
           - kernel 未调用 + torch API → 硬失败（明确作弊）
           - kernel 已调用 + torch API → 仅日志警告（可能是合理辅助操作）
        """
        if not self.dsl.startswith("triton"):
            return []

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return []

        errors: List[Dict] = []

        # --- A. 收集所有 @triton.jit 装饰的 kernel 函数名 ---
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
                    f"DSL 指定为 {self.dsl}，但代码中未找到任何 @triton.jit 装饰的 kernel 函数。"
                    f"代码可能使用了 torch 高层 API 替代 triton kernel 实现。"
                ),
                "suggestion": (
                    "请确保代码中包含至少一个 @triton.jit 装饰的 kernel 函数，"
                    "并在 ModelNew.forward() 中通过 kernel[grid](...) 语法调用它。"
                ),
                "code_snippet": ""
            })
            return errors

        # --- B. 检查 kernel 是否在代码中被实际启动 (kernel[grid](...) 语法) ---
        #
        # 在 AST 中，kernel[grid](args) 解析为：
        #   Call(func=Subscript(value=Name(id='kernel_name'), slice=...))
        #
        # 扫描整个文件而非仅 ModelNew，以覆盖 helper 函数中的调用。
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
                    f"定义了 triton kernel 函数 {sorted(triton_kernels)}，"
                    f"但代码中未找到任何 kernel[grid](...) 形式的调用。"
                    f"kernel 函数可能只是装饰性的，实际计算未使用 triton。"
                ),
                "suggestion": (
                    "请在 ModelNew.forward() 或其辅助方法中，"
                    "通过 kernel_name[grid_size](...) 语法启动 triton kernel。"
                ),
                "code_snippet": ""
            })

        # --- C. 检查 forward() 中的 torch 高层计算 API 调用 ---
        #
        # 分两层处理：
        #   HARD 类（matmul/einsum/conv/norm/softmax/pooling 等）:
        #       无论 kernel 是否已调用都硬失败 —— 这些必须由 kernel 实现。
        #   SOFT 类（exp/sin/relu/sum 等简单元素级/辅助操作）:
        #       kernel 未调用 → 硬失败；kernel 已调用 → 仅日志警告（融合算子允许）。
        model_cls = _find_model_new_class(tree)
        if model_cls is None:
            return errors

        forward_node = _find_forward(model_cls)
        if forward_node is None:
            return errors

        hard_calls: List[tuple] = []
        soft_calls: List[tuple] = []

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

        def _fmt_calls(calls: List[tuple], limit: int = 5) -> str:
            summary = ", ".join(f"{name}(第{line}行)" for line, name in calls[:limit])
            if len(calls) > limit:
                summary += f" 等（共 {len(calls)} 处）"
            return summary

        # HARD 类：无论 kernel 是否调用，都硬失败
        if hard_calls:
            errors.append({
                "line": hard_calls[0][0],
                "error_type": "torch_api_instead_of_kernel",
                "detail": (
                    f"forward() 中使用了 {len(hard_calls)} 个不允许的 torch 高层计算 API: "
                    f"{_fmt_calls(hard_calls)}。"
                    f"这些操作（矩阵乘法、卷积、归一化、池化等）必须完全在 DSL kernel 内实现。"
                ),
                "suggestion": (
                    "请将这些核心计算操作移入 triton kernel 中实现，"
                    "forward() 仅负责准备输入、启动 kernel 和返回输出。"
                ),
                "code_snippet": ""
            })

        # SOFT 类：kernel 未调用 → 硬失败；kernel 已调用 → 仅警告
        if soft_calls:
            if kernels_not_launched:
                errors.append({
                    "line": soft_calls[0][0],
                    "error_type": "torch_api_without_kernel",
                    "detail": (
                        f"forward() 中使用了 {len(soft_calls)} 个 torch 计算 API: "
                        f"{_fmt_calls(soft_calls)}。"
                        f"同时 triton kernel 未被调用，代码很可能用 torch API 替代了 kernel 实现。"
                    ),
                    "suggestion": (
                        "请将核心计算逻辑用 triton kernel 实现。"
                        "这些简单操作（exp/relu/sum 等）如果只是 kernel 的前后处理可以保留，"
                        "但前提是必须有 kernel 承担主要计算。"
                    ),
                    "code_snippet": ""
                })
            else:
                logger.warning(
                    f"CodeChecker DSL compliance: forward() 调用了 triton kernel，"
                    f"同时包含 {len(soft_calls)} 处 torch 辅助计算 API: "
                    f"{_fmt_calls(soft_calls)}。（融合算子可能合理，仅记录警告）"
                )

        return errors

    # ------------------------------------------------------------------
    # Step 6: Autotune 规范检测
    # ------------------------------------------------------------------

    _AUTOTUNE_RE = re.compile(r'@triton\.autotune\s*\(', re.MULTILINE)
    _RESTORE_VALUE_RE = re.compile(r'restore_value\s*=')

    def _check_autotune_compliance(self, code: str) -> List[Dict]:
        """
        检查 @triton.autotune 使用是否符合规范：
        - 必须包含 restore_value 参数
        """
        errors = []

        autotune_match = self._AUTOTUNE_RE.search(code)
        if not autotune_match:
            return errors

        autotune_line = code[:autotune_match.start()].count('\n') + 1

        paren_depth = 0
        start = autotune_match.end() - 1
        end = start
        for i in range(start, len(code)):
            if code[i] == '(':
                paren_depth += 1
            elif code[i] == ')':
                paren_depth -= 1
                if paren_depth == 0:
                    end = i + 1
                    break
        autotune_block = code[start:end]

        if not self._RESTORE_VALUE_RE.search(autotune_block):
            errors.append({
                "line": autotune_line,
                "error_type": "autotune_missing_restore_value",
                "detail": (
                    "@triton.autotune 装饰器缺少 restore_value 参数。"
                    "autotune benchmark 会对每个 config 反复执行 kernel，"
                    "不同 config 之间的输出会互相污染，导致验证失败。"
                ),
                "suggestion": (
                    "在 @triton.autotune(...) 中添加 restore_value=['输出指针参数名']，"
                    "列出 kernel 的所有输出指针参数。例如：\n"
                    "  @triton.autotune(\n"
                    "      configs=[...],\n"
                    "      key=[...],\n"
                    "      restore_value=['output_ptr'],  # 必须添加\n"
                    "  )"
                ),
                "code_snippet": ""
            })
            logger.warning(
                f"CodeChecker: @triton.autotune at line {autotune_line} missing restore_value"
            )

        return errors

    # ------------------------------------------------------------------
    # 格式化输出
    # ------------------------------------------------------------------

    def _format_errors(self, errors: List[Dict], code_lines: Optional[List[str]] = None) -> str:
        """格式化错误信息，便于传递给 Coder"""
        if not errors:
            return ""

        lines = [
            "## CodeChecker 静态检查报告",
            "",
            f"**发现 {len(errors)} 个问题，请修复后重新生成代码：**",
            ""
        ]

        for i, err in enumerate(errors, 1):
            error_line = err['line']
            lines.append(f"### 问题 {i}: 第 {error_line} 行 [{err.get('error_type', 'unknown')}]")
            lines.append(f"  {err['detail']}")

            if code_lines is not None and error_line > 0:
                start_line = max(1, error_line - 3)
                end_line = min(len(code_lines), error_line + 3)

                lines.append(f"  上下文（第 {start_line}-{end_line} 行）：")
                for ctx_line_num in range(start_line, end_line + 1):
                    ctx_line = code_lines[ctx_line_num - 1]
                    if ctx_line_num == error_line:
                        lines.append(f"  >>> {ctx_line_num:4d} | {ctx_line}")
                    else:
                        lines.append(f"      {ctx_line_num:4d} | {ctx_line}")
            elif err.get('code_snippet'):
                lines.append(f"  出错代码: {err['code_snippet']}")

            if err.get('suggestion'):
                lines.append(f"  建议：")
                for sug_line in err['suggestion'].strip().split('\n'):
                    lines.append(f"    {sug_line}")

            lines.append("")

        lines.append("**注意：语法检查每次只能定位到首个错误，修复后可能还有后续问题，请仔细检查整段代码。**")

        return "\n".join(lines)

    def get_check_summary(self, errors: List[Dict]) -> str:
        """获取检查摘要（简短版本，用于日志）"""
        if not errors:
            return "代码检查通过"

        error_types = set(err.get("error_type", "unknown") for err in errors)
        return f"发现 {len(errors)} 个问题: {', '.join(error_types)}"
