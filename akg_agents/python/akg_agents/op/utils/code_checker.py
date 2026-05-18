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
import importlib.resources
import importlib.util
import tempfile
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Policy: single source of truth is akg_agents/op/config/code_checker.yaml.
# Loaded once at import; missing/malformed keys surface as KeyError / TypeError
# on first access (no redundant validation layer).
# ---------------------------------------------------------------------------

with importlib.resources.files("akg_agents.op.config").joinpath(
    "code_checker.yaml"
).open("r", encoding="utf-8") as _f:
    _POLICY = yaml.safe_load(_f)

_STRAY_TEXT_RE = re.compile(
    "[" + "".join(
        f"\\u{lo:04x}-\\u{hi:04x}" for lo, hi in _POLICY["stray_text"]["unicode_ranges"]
    ) + "]{" + str(_POLICY["stray_text"]["min_run"]) + ",}"
)
_AUTOTUNE_RE = re.compile(
    rf"@{re.escape(_POLICY['triton_module_name'])}\."
    rf"{re.escape(_POLICY['autotune']['decorator_attr'])}\s*\(",
    re.MULTILINE,
)
_RESTORE_VALUE_RE = re.compile(
    rf"{re.escape(_POLICY['autotune']['required_kwarg'])}\s*="
)

_A5_ARCH_PREFIX: str = _POLICY["a5_compliance"]["arch_prefix"]
_A5_DSL: str = _POLICY["a5_compliance"]["dsl"]
_A5_ENABLE_AFFINITY_CHECK: bool = bool(
    _POLICY["a5_compliance"]["enable_triton_ascend_affinity_check"]
)
_A5_AL_ALIAS: str = _POLICY["a5_compliance"]["aliases"]["al"]
_A5_BL_ALIAS: str = _POLICY["a5_compliance"]["aliases"]["bl"]
_A5_ONLY_APIS: frozenset = frozenset(_POLICY["a5_compliance"]["only_apis"])
_BL_APIS: frozenset = frozenset(_POLICY["a5_compliance"]["bl_apis"])

# TileLang compliance constants
_DSL_COMPLIANCE_PREFIXES: tuple = tuple(_POLICY["dsl_compliance_prefixes"])
_TL_MODULE_NAME: str = _POLICY["tilelang_compliance"]["module_name"]
_TL_DECORATORS: frozenset = frozenset(_POLICY["tilelang_compliance"]["decorators"])
_TL_PRIM_FUNC_ATTR: str = _POLICY["tilelang_compliance"]["prim_func_attr"]
_TL_NAMESPACE: str = _POLICY["tilelang_compliance"]["tl_namespace"]


def _find_model_new_class(tree: ast.Module) -> Optional[ast.ClassDef]:
    target = _POLICY["kernel_class_name"]
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == target:
            return node
    return None


def _find_forward(cls_node: ast.ClassDef) -> Optional[ast.FunctionDef]:
    target = _POLICY["kernel_forward_method"]
    for item in cls_node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == target:
            return item
    return None


def _dotted_name(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted_name(node.value)
        if base:
            return f"{base}.{node.attr}"
    return None


@dataclass
class CheckError:
    """检查错误信息"""
    line: int
    error_type: str
    detail: str
    suggestion: str
    code_snippet: str
    fix_strategy: str = "fix"  # "fix" 或 "rewrite"


class CodeChecker:
    """
    代码检查器：在 Coder 生成代码后、Verifier 验证前，进行快速的纯静态检查

    检查流程：ast.parse → py_compile → import 验证 → 中文文本混入检测
    不调用 LLM，零额外成本。
    """

    def __init__(self, backend: str, dsl: str, arch: str = "", config: Optional[dict] = None):
        self.backend = backend.lower() if backend else ""
        self.dsl = dsl.lower() if dsl else ""
        self.arch = arch.lower() if arch else ""
        # `config` 保留仅为兼容调用方签名；策略真源是 op/config/code_checker.yaml。
        self.config = config or {}
        # 暴露为实例属性便于测试与日志；来自 _POLICY 的浅复制（frozenset）。
        self.triton_decorators = frozenset(_POLICY["triton_decorators"])
        self.torch_compute_ops_hard = frozenset(_POLICY["torch_compute_ops_hard"])
        self.torch_compute_ops_soft = frozenset(_POLICY["torch_compute_ops_soft"])
        self.torch_call_prefixes = frozenset(_POLICY["torch_call_prefixes"])
        self._torch_call_prefixes_ordered = tuple(
            sorted(self.torch_call_prefixes, key=len, reverse=True)
        )
        logger.info(f"CodeChecker initialized: backend={self.backend}, dsl={self.dsl}, arch={self.arch}")

    @staticmethod
    def _collect_import_aliases(tree: ast.Module) -> Dict[str, str]:
        """Build a map of local-name → dotted-module-name from import statements.

        Handles:
          `from tilelang import jit`              → {"jit": "tilelang.jit"}
          `from tilelang import jit as j`         → {"j": "tilelang.jit"}
          `from triton import jit, autotune`       → {"jit": "triton.jit", "autotune": "triton.autotune"}
          `import tilelang as tl`                  → {"tl": "tilelang"}
          `import tilelang`                        → {"tilelang": "tilelang"}

        Only collects aliases that resolve to the Triton or TileLang module
        namespace — avoids false positives from unrelated `@jit` decorators.
        """
        _TARGET_MODULES = frozenset({_POLICY["triton_module_name"], _TL_MODULE_NAME})
        alias_map: Dict[str, str] = {}

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split('.')[0] in _TARGET_MODULES:
                        local_name = alias.asname if alias.asname else alias.name
                        alias_map[local_name] = alias.name

            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.split('.')[0] in _TARGET_MODULES:
                    for alias in node.names:
                        local_name = alias.asname if alias.asname else alias.name
                        dotted = f"{node.module}.{alias.name}"
                        alias_map[local_name] = dotted

        return alias_map

    def _is_triton_decorator(self, node: ast.expr, import_aliases: Optional[Dict[str, str]] = None) -> bool:
        """Check if an AST node is a triton kernel decorator.

        Recognizes three forms:
          1. @triton.jit           — dotted attribute
          2. @jit                  — bare name (when `from triton import jit` exists)
          3. @triton.jit(...)      — called decorator (unwraps to check inner func)

        `import_aliases` maps local bare names to their full dotted path
        (e.g. {"jit": "triton.jit"}).  Built by _collect_import_aliases().
        """
        if import_aliases is None:
            import_aliases = {}
        # Called decorator: @triton.jit(...) or @jit(...) → unwrap to func
        if isinstance(node, ast.Call):
            return self._is_triton_decorator(node.func, import_aliases)
        # Dotted form: @triton.jit
        if isinstance(node, ast.Attribute):
            return (
                isinstance(node.value, ast.Name)
                and node.value.id == _POLICY["triton_module_name"]
                and node.attr in self.triton_decorators
            )
        # Bare form: @jit — only matches if `jit` resolves to `triton.jit`
        if isinstance(node, ast.Name):
            resolved = import_aliases.get(node.id, "")
            parts = resolved.rsplit(".", 1)
            if len(parts) == 2 and parts[0] == _POLICY["triton_module_name"] and parts[1] in self.triton_decorators:
                return True
        return False

    def _is_tilelang_decorator(self, node: ast.expr, import_aliases: Optional[Dict[str, str]] = None) -> bool:
        """Check if an AST node is a tilelang kernel decorator.

        Recognizes three forms:
          1. @tilelang.jit          — dotted attribute
          2. @jit                   — bare name (when `from tilelang import jit` exists)
          3. @tilelang.jit(...)     — called decorator (unwraps to check inner func)

        `import_aliases` maps local bare names to their full dotted path
        (e.g. {"jit": "tilelang.jit"}).  Built by _collect_import_aliases().
        """
        if import_aliases is None:
            import_aliases = {}
        # Called decorator: @tilelang.jit(...) or @jit(...) → unwrap to func
        if isinstance(node, ast.Call):
            return self._is_tilelang_decorator(node.func, import_aliases)
        # Dotted form: @tilelang.jit
        if isinstance(node, ast.Attribute):
            return (
                isinstance(node.value, ast.Name)
                and node.value.id == _TL_MODULE_NAME
                and node.attr in _TL_DECORATORS
            )
        # Bare form: @jit — only matches if `jit` resolves to `tilelang.jit`
        if isinstance(node, ast.Name):
            resolved = import_aliases.get(node.id, "")
            parts = resolved.rsplit(".", 1)
            if len(parts) == 2 and parts[0] == _TL_MODULE_NAME and parts[1] in _TL_DECORATORS:
                return True
        return False

    def _is_prim_func_decorator(self, node: ast.expr) -> bool:
        """Check if an AST node is a @T.prim_func decorator.

        Accepts both `@T.prim_func` (bare) and `@T.prim_func(...)` (called).
        Also handles aliased imports like `import tilelang.language as TL`
        where the alias matches _TL_NAMESPACE.
        """
        if isinstance(node, ast.Attribute):
            return (
                isinstance(node.value, ast.Name)
                and node.value.id == _TL_NAMESPACE
                and node.attr == _TL_PRIM_FUNC_ATTR
            )
        if isinstance(node, ast.Call):
            return self._is_prim_func_decorator(node.func)
        return False

    def _find_tilelang_kernel_calls(self, tree: ast.Module, kernel_names: set) -> set:
        """Find tilelang kernel invocations in the AST.

        TileLang kernel calls appear in two common patterns:

        1. Direct call after factory construction:
           kernel = kernel_func(params)       # factory returns compiled kernel
           kernel(inputs, outputs)            # call the compiled kernel

           Or inlined:
           kernel_func(params)(inputs, outputs)

        2. Explicit compile step:
           compiled = tilelang.compile(func, target=...)
           compiled(inputs, outputs)

        We detect:
        - Direct calls to any name in kernel_names (pattern 1)
        - Calls to names assigned from tilelang.compile(...) (pattern 2)
        """
        launched: set = set()

        # Pattern 2: collect names assigned from `tilelang.compile(...)` calls
        compile_result_names: set = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        # RHS is `tilelang.compile(...)`
                        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute):
                            func = node.value.func
                            if (
                                isinstance(func.value, ast.Name)
                                and func.value.id == _TL_MODULE_NAME
                                and func.attr == "compile"
                            ):
                                compile_result_names.add(target.id)

        # Pattern 1 + 2: walk all Call nodes and check if callee is a kernel
        # name or a compile-result name
        all_callable_names = kernel_names | compile_result_names
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in all_callable_names:
                    if node.func.id in kernel_names:
                        launched.add(node.func.id)
                    elif node.func.id in compile_result_names:
                        # A compiled kernel was called — mark all as launched
                        # since we can't easily determine which one.
                        launched.update(kernel_names)

        return launched

    def _match_torch_call_prefix(self, call_name: str) -> Optional[str]:
        for prefix in self._torch_call_prefixes_ordered:
            if call_name.startswith(f"{prefix}."):
                return prefix
        return None

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def check(self, code: str, task_info: Optional[dict] = None) -> Tuple[bool, str, List[Dict]]:
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
                "code_snippet": "",
                "fix_strategy": "rewrite"
            }
            return False, self._format_errors([empty_err]), [empty_err]

        if self.dsl and self.dsl not in _POLICY.get("python_dsls", []):
            logger.info(
                f"CodeChecker: DSL '{self.dsl}' is not Python-based, skipping checks"
            )
            return True, "", []

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

        # Step 7: A5 API 合规性检测
        # 触发条件（全部满足）：
        #   1. 语法/编译均通过；
        #   2. arch.startswith(_A5_ARCH_PREFIX) 且 dsl == _A5_DSL —— 即在 A5 + triton_ascend 路径上；
        #   3. yaml 中 a5_compliance.enable_triton_ascend_affinity_check 为 True。
        if (
            not has_syntax_err
            and self.arch.startswith(_A5_ARCH_PREFIX)
            and self.dsl == _A5_DSL
            and _A5_ENABLE_AFFINITY_CHECK
        ):
            errors.extend(self._check_a5_api_compliance(code))
        elif (
            not has_syntax_err
            and self.arch.startswith(_A5_ARCH_PREFIX)
            and self.dsl == _A5_DSL
            and not _A5_ENABLE_AFFINITY_CHECK
        ):
            logger.info(
                f"CodeChecker A5: arch={self.arch}, dsl={self.dsl} — affinity "
                "enforcement disabled via "
                "a5_compliance.enable_triton_ascend_affinity_check=false; "
                "skipping Step 7."
            )

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
                "code_snippet": code_snippet,
                "fix_strategy": "fix"
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
                "code_snippet": code_snippet,
                "fix_strategy": "fix"
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
                            "code_snippet": "",
                            "fix_strategy": "fix"
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
                            "code_snippet": "",
                            "fix_strategy": "fix"
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
    # Step 4: 中文文本混入检测 —— regex 来自 op/config/code_checker.yaml
    # 的 stray_text.min_run / stray_text.unicode_ranges
    # ------------------------------------------------------------------

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

            match = _STRAY_TEXT_RE.search(tok.string)
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
                    "code_snippet": "",
                    "fix_strategy": "fix"
                })
                logger.warning(
                    f"CodeChecker: stray Chinese text at line {line_num}: '{chinese_text}'"
                )

        return errors

    # ------------------------------------------------------------------
    # Step 5: DSL 合规性检测（反作弊）—— triton / tilelang 统一入口
    # ------------------------------------------------------------------

    # DSL-specific metadata used by _check_dsl_compliance.
    # Each entry keys off the compliance-prefix; the rest of the method
    # is shared logic that only varies in these text / error_type templates.
    _DSL_META = {
        "triton": {
            "kernel_decor_check": "_is_triton_decorator",
            "no_kernel_type": "no_triton_kernel",
            "no_kernel_detail": (
                "DSL 指定为 {dsl}，但代码中未找到任何 @triton.jit 装饰的 kernel 函数。"
                "代码可能使用了 torch 高层 API 替代 triton kernel 实现。"
            ),
            "no_kernel_suggestion": (
                "请确保代码中包含至少一个 @triton.jit 装饰的 kernel 函数，"
                "并在 ModelNew.forward() 中通过 kernel[grid](...) 语法调用它。"
            ),
            "not_called_type": "triton_kernel_not_called",
            "not_called_detail": (
                "定义了 triton kernel 函数 {kernel_list}，"
                "但代码中未找到任何 kernel[grid](...) 形式的调用。"
                "kernel 函数可能只是装饰性的，实际计算未使用 triton。"
            ),
            "not_called_suggestion": (
                "请在 ModelNew.forward() 或其辅助方法中，"
                "通过 kernel_name[grid_size](...) 语法启动 triton kernel。"
            ),
            "hard_type": "torch_api_instead_of_kernel",
            "hard_detail": (
                "forward() 中使用了 {count} 个不允许的 torch 高层计算 API: "
                "{calls}。"
                "这些操作（矩阵乘法、卷积、归一化、池化等）必须完全在 triton kernel 内实现。"
            ),
            "hard_suggestion": (
                "请将这些核心计算操作移入 triton kernel 中实现，"
                "forward() 仅负责准备输入、启动 kernel 和返回输出。"
            ),
            "soft_no_kernel_type": "torch_api_without_kernel",
            "soft_no_kernel_detail": (
                "forward() 中使用了 {count} 个 torch 计算 API: {calls}。"
                "同时 triton kernel 未被调用，代码很可能用 torch API 替代了 kernel 实现。"
            ),
            "soft_no_kernel_suggestion": (
                "请将核心计算逻辑用 triton kernel 实现。"
                "这些简单操作（exp/relu/sum 等）如果只是 kernel 的前后处理可以保留，"
                "但前提是必须有 kernel 承担主要计算。"
            ),
            "soft_with_kernel_log": (
                "CodeChecker DSL compliance: forward() 调用了 triton kernel，"
                "同时包含 {count} 处 torch 辅助计算 API: {calls}。"
                "（融合算子可能合理，仅记录警告）"
            ),
        },
        "tilelang": {
            "kernel_decor_check": "_is_tilelang_decorator",
            "no_kernel_type": "no_tilelang_kernel",
            "no_kernel_detail": (
                "DSL 指定为 {dsl}，但代码中未找到任何 @tilelang.jit 装饰的 "
                "kernel 函数。代码可能使用了 torch 高层 API 替代 tilelang kernel "
                "实现（torch 退化）。"
            ),
            "no_kernel_suggestion": (
                "请确保代码中包含至少一个 @tilelang.jit 装饰的 kernel 函数，"
                "并在 ModelNew.forward() 中调用编译后的 kernel 执行计算。"
            ),
            "not_called_type": "tilelang_kernel_not_called",
            "not_called_detail": (
                "定义了 tilelang kernel 函数 {kernel_list}，"
                "但代码中未找到任何 kernel 调用。"
                "kernel 函数可能只是装饰性的，实际计算未使用 tilelang（torch 退化）。"
            ),
            "not_called_suggestion": (
                "请在 ModelNew.forward() 中调用编译后的 tilelang kernel 执行计算，"
                "例如：\n"
                "  kernel = my_kernel(M, N, K)\n"
                "  kernel(A, B, C)\n"
                "或使用 tilelang.compile 编译后调用：\n"
                "  compiled = tilelang.compile(func, target='npuir')\n"
                "  compiled(A, B, C)"
            ),
            "hard_type": "torch_api_instead_of_tilelang_kernel",
            "hard_detail": (
                "forward() 中使用了 {count} 个不允许的 torch 高层计算 API: "
                "{calls}。"
                "这些操作（矩阵乘法、卷积、归一化、池化等）必须完全在 tilelang kernel "
                "内实现，使用 torch 实现即为退化。"
            ),
            "hard_suggestion": (
                "请将这些核心计算操作移入 tilelang kernel 中实现，"
                "forward() 仅负责准备输入、启动 kernel 和返回输出。"
            ),
            "soft_no_kernel_type": "torch_api_without_tilelang_kernel",
            "soft_no_kernel_detail": (
                "forward() 中使用了 {count} 个 torch 计算 API: {calls}。"
                "同时 tilelang kernel 未被调用，代码很可能用 torch API 替代了 "
                "kernel 实现（torch 退化）。"
            ),
            "soft_no_kernel_suggestion": (
                "请将核心计算逻辑用 tilelang kernel 实现。"
                "这些简单操作（exp/relu/sum 等）如果只是 kernel 的前后处理可以保留，"
                "但前提是必须有 kernel 承担主要计算。"
            ),
            "soft_with_kernel_log": (
                "CodeChecker TileLang DSL compliance: forward() 调用了 tilelang kernel，"
                "同时包含 {count} 处 torch 辅助计算 API: {calls}。"
                "（融合算子可能合理，仅记录警告）"
            ),
        },
    }

    def _check_dsl_compliance(self, code: str) -> List[Dict]:
        """
        检测生成代码是否真正使用了指定的 DSL 实现（纯 AST 静态分析）。

        对 triton 和 tilelang 系列 DSL 统一生效，检查三类问题：

        A. 没有定义任何 DSL kernel                    → 硬失败
        B. 定义了 kernel 但没有调用                     → 硬失败
        C. forward() 中使用了 torch 高层计算 API          → 结合 B 判断
           - kernel 未调用 + torch API → 硬失败（明确作弊 / 退化）
           - kernel 已调用 + torch API → 仅日志警告（可能是合理辅助操作）
        """
        # Determine which DSL family we're checking
        dsl_family = None
        for prefix in _DSL_COMPLIANCE_PREFIXES:
            if self.dsl.startswith(prefix):
                dsl_family = prefix
                break
        if dsl_family is None:
            return []

        meta = self._DSL_META[dsl_family]

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return []

        errors: List[Dict] = []

        # Collect import aliases so we can recognise bare-name decorators
        # like `@jit` (from `from tilelang import jit`).
        import_aliases = self._collect_import_aliases(tree)

        # --- A. 收集 kernel 函数（使用 DSL-specific 装饰器检测方法）---
        decor_check_method = getattr(self, meta["kernel_decor_check"])
        kernel_names: set = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for dec in node.decorator_list:
                    if decor_check_method(dec, import_aliases):
                        kernel_names.add(node.name)
                        break

        if not kernel_names:
            errors.append({
                "line": 0,
                "error_type": meta["no_kernel_type"],
                "detail": meta["no_kernel_detail"].format(dsl=self.dsl),
                "suggestion": meta["no_kernel_suggestion"],
                "code_snippet": "",
                "fix_strategy": "rewrite"
            })
            return errors

        # --- B. 检查 kernel 是否在代码中被实际调用（DSL-specific 调用模式）---
        if dsl_family == "triton":
            # Triton: kernel[grid](...) → Call(func=Subscript(value=Name))
            launched_kernels: set = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Subscript):
                    value = node.func.value
                    if isinstance(value, ast.Name) and value.id in kernel_names:
                        launched_kernels.add(value.id)
        else:
            # TileLang: kernel(inputs) or compiled(inputs) after tilelang.compile
            launched_kernels = self._find_tilelang_kernel_calls(tree, kernel_names)

        kernels_not_launched = not launched_kernels

        if kernels_not_launched:
            errors.append({
                "line": 0,
                "error_type": meta["not_called_type"],
                "detail": meta["not_called_detail"].format(
                    kernel_list=sorted(kernel_names)
                ),
                "suggestion": meta["not_called_suggestion"],
                "code_snippet": "",
                "fix_strategy": "rewrite"
            })

        # --- C. 检查 forward() 中的 torch 高层计算 API 调用 ---
        # 分两层处理（triton / tilelang 共用同一套 torch API 分类）：
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
                call_name = _dotted_name(node.func)
                if not call_name or not self._match_torch_call_prefix(call_name):
                    continue
                method = node.func.attr
                if method in self.torch_compute_ops_hard:
                    hard_calls.append((node.lineno, call_name))
                elif method in self.torch_compute_ops_soft:
                    soft_calls.append((node.lineno, call_name))

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
                "error_type": meta["hard_type"],
                "detail": meta["hard_detail"].format(
                    count=len(hard_calls), calls=_fmt_calls(hard_calls)
                ),
                "suggestion": meta["hard_suggestion"],
                "code_snippet": "",
                "fix_strategy": "rewrite"
            })

        # SOFT 类：kernel 未调用 → 硬失败；kernel 已调用 → 仅警告
        if soft_calls:
            if kernels_not_launched:
                errors.append({
                    "line": soft_calls[0][0],
                    "error_type": meta["soft_no_kernel_type"],
                    "detail": meta["soft_no_kernel_detail"].format(
                        count=len(soft_calls), calls=_fmt_calls(soft_calls)
                    ),
                    "suggestion": meta["soft_no_kernel_suggestion"],
                    "code_snippet": "",
                    "fix_strategy": "rewrite"
                })
            else:
                logger.warning(
                    meta["soft_with_kernel_log"].format(
                        count=len(soft_calls), calls=_fmt_calls(soft_calls)
                    )
                )

        return errors

    # ------------------------------------------------------------------
    # Step 6: Autotune 规范检测
    # ------------------------------------------------------------------


    def _check_autotune_compliance(self, code: str) -> List[Dict]:
        """检查 @triton.autotune 必须包含 restore_value 参数。"""
        errors = []

        autotune_match = _AUTOTUNE_RE.search(code)
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

        if not _RESTORE_VALUE_RE.search(autotune_block):
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
                "code_snippet": "",
                "fix_strategy": "fix"
            })
            logger.warning(
                f"CodeChecker: @triton.autotune at line {autotune_line} missing restore_value"
            )

        return errors

    # ------------------------------------------------------------------
    # Step 7: A5 (Ascend950) API 合规性检测并使能亲和编程
    # ------------------------------------------------------------------

    @staticmethod
    def _kernel_uses_tl_dot(kernel: ast.AST) -> bool:
        """Return True if the kernel body contains any `tl.dot(...)` call.

        """
        for node in ast.walk(kernel):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            # `tl.dot(...)`
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "dot"
                and isinstance(func.value, ast.Name)
                and func.value.id == "tl"
            ):
                return True
            # `triton.language.dot(...)`
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "dot"
                and isinstance(func.value, ast.Attribute)
                and func.value.attr == "language"
                and isinstance(func.value.value, ast.Name)
                and func.value.value.id == "triton"
            ):
                return True
        return False

    def _check_a5_api_compliance(self, code: str) -> List[Dict]:
        """
        检测 Ascend950 (A5) 硬件上 triton_ascend 代码是否真正调用了
        Cube/Vector 协同编程的亲和接口。
        """
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return []

        import_aliases = self._collect_import_aliases(tree)

        triton_kernels: List[ast.FunctionDef] = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for dec in node.decorator_list:
                    if self._is_triton_decorator(dec, import_aliases):
                        triton_kernels.append(node)
                        break

        if not triton_kernels:
            return []

        cube_required = any(
            self._kernel_uses_tl_dot(k) for k in triton_kernels
        )
        if not cube_required:
            logger.info(
                f"CodeChecker A5: arch={self.arch}, dsl={self.dsl} — no tl.dot "
                "found in any kernel; treating as pure-vector op and skipping "
                "Cube/Vector affinity API checks."
            )
            return []

        has_al_scope = False
        has_fixpipe = False
        has_bl_alloc = False

        for kernel in triton_kernels:
            for node in ast.walk(kernel):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func

                # 直接命名空间调用：al.<method>(...) / bl.<method>(...)
                if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                    prefix = func.value.id
                    method = func.attr
                    if prefix == _A5_AL_ALIAS:
                        if method == "scope":
                            has_al_scope = True
                        elif method == "fixpipe":
                            has_fixpipe = True
                    elif prefix == _A5_BL_ALIAS:
                        if method == "alloc":
                            has_bl_alloc = True

                # 链式调用：al.<x>.<method>(...) —— 例如
                # `al.something.scope(...)`。受 `only_apis` 白名单约束，
                # 避免和无关的 `al.foo.bar()` 混淆。
                if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Attribute):
                    inner = func.value
                    if (isinstance(inner.value, ast.Name)
                            and inner.value.id == _A5_AL_ALIAS
                            and func.attr in _A5_ONLY_APIS):
                        if func.attr == "scope":
                            has_al_scope = True
                        elif func.attr == "fixpipe":
                            has_fixpipe = True

        errors: List[Dict] = []

        # al.scope 检测
        if not has_al_scope:
            errors.append({
                "line": 0,
                "error_type": "a5_missing_scope",
                "detail": (
                    f"目标架构为 {self.arch}（A5 硬件），但 kernel 中未使用 al.scope(core_mode=...) "
                    "划分 Cube/Vector 执行域。A5 的 Cube 和 Vector 核需要通过 al.scope 分别编排。"
                ),
                "suggestion": (
                    "请在 kernel 中使能亲和编程写法实现kernel内的Cube和Vector计算，可以使用 al.scope 划分计算域，例如：\n"
                    "  with al.scope(core_mode=\"cube\"):\n"
                    "      acc = tl.dot(a, b)\n"
                    "      al.fixpipe(acc, dst_buf, ...)\n"
                    "  with al.scope(core_mode=\"vector\"):\n"
                    "      c = bl.to_tensor(buf)\n"
                    "      tl.store(out_ptr, c)"
                ),
                "code_snippet": ""
            })

        # al.fixpipe 检测：只有进入 al.scope 后 fixpipe 才有意义
        if has_al_scope and not has_fixpipe:
            errors.append({
                "line": 0,
                "error_type": "a5_missing_fixpipe",
                "detail": (
                    f"目标架构为 {self.arch}（A5 硬件），kernel 使用了 al.scope 但未调用 al.fixpipe。"
                    "A5 Cube 域计算完成后通常需要通过 fixpipe 将 L0C 数据搬运到 UB/L1。"
                ),
                "suggestion": (
                    "如果使能了亲和编程写法，请在 Cube scope 内的 tl.dot 之后添加 al.fixpipe 调用，将结果搬运到UB，例如：\n"
                    "  al.fixpipe(acc, bl.to_buffer(c_ub, al.ascend_address_space.UB),\n"
                    "             al.FixpipeDMAMode.NZ2ND, al.FixpipeDualDstMode.ROW_SPLIT)"
                ),
                "code_snippet": ""
            })

        # bl.alloc 检测
        if not has_bl_alloc:
            errors.append({
                "line": 0,
                "error_type": "a5_missing_bl_alloc",
                "detail": (
                    f"目标架构为 {self.arch}（A5 硬件），但 kernel 中未使用 bl.alloc 分配片上 buffer。"
                    "A5 Cube/Vector 协同需要在 UB 或 L1 上分配 buffer 作为数据交换区域。"
                ),
                "suggestion": (
                    "如果使能了亲和编程写法，在 kernel 中申请使用 buffer的时候请使用 bl.alloc 分配 buffer，可以在UB、L1、L0C、L0A、L0B上分配，例如：\n"
                    "  c_ub = bl.alloc(tl.float32, (BLOCK_M, BLOCK_N), al.ascend_address_space.UB)\n"
                    "  c_l1 = bl.alloc(tl.float32, (BLOCK_M, BLOCK_N), al.ascend_address_space.L1)"
                ),
                "code_snippet": ""
            })

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