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
5. DSL/arch 合规性检测（反作弊：每个 DSL 各一个 _ComplianceCheck 类，
   各自独立 owns 自己的策略字段，``CodeChecker.__init__`` 不再感知任何
   单 DSL 的策略 schema）
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

# ---------------------------------------------------------------------------
# Module-level constants derived from _POLICY. These are SHARED across
# multiple compliance checks (triton uses them, catlass uses them when
# scanning forward() for forbidden torch ops, pypto uses the hard subset).
# Storing them once at module scope avoids each Check class re-loading them
# and keeps the per-Check __init__ focused on Check-specific literals.
# ---------------------------------------------------------------------------

_STRAY_TEXT_RE = re.compile(
    "[" + "".join(
        f"\\u{lo:04x}-\\u{hi:04x}" for lo, hi in _POLICY["stray_text"]["unicode_ranges"]
    ) + "]{" + str(_POLICY["stray_text"]["min_run"]) + ",}"
)

_TRITON_MODULE_NAME: str = _POLICY["triton_module_name"]
_TRITON_DECORATORS: frozenset = frozenset(_POLICY["triton_decorators"])
_TORCH_COMPUTE_OPS_HARD: frozenset = frozenset(_POLICY["torch_compute_ops_hard"])
_TORCH_COMPUTE_OPS_SOFT: frozenset = frozenset(_POLICY["torch_compute_ops_soft"])
_TORCH_CALL_PREFIXES: frozenset = frozenset(_POLICY["torch_call_prefixes"])
_TORCH_CALL_PREFIXES_ORDERED: tuple = tuple(
    sorted(_TORCH_CALL_PREFIXES, key=len, reverse=True)
)
_DSL_COMPLIANCE_PREFIXES: tuple = tuple(_POLICY["dsl_compliance_prefixes"])
_TL_MODULE_NAME: str = _POLICY["tilelang_compliance"]["module_name"]
_TL_DECORATORS: frozenset = frozenset(_POLICY["tilelang_compliance"]["decorators"])
_TL_PRIM_FUNC_ATTR: str = _POLICY["tilelang_compliance"]["prim_func_attr"]
_TL_NAMESPACE: str = _POLICY["tilelang_compliance"]["tl_namespace"]
_ASCENDC_TEXT_SUFFIXES: frozenset = frozenset(
    _POLICY["ascendc_anti_cheat"]["text_suffixes"]
)
_ASCENDC_TEXT_FILENAMES: frozenset = frozenset(
    _POLICY["ascendc_anti_cheat"]["text_filenames"]
)


# ---------------------------------------------------------------------------
# Free helpers — AST navigation + shared decorator/prefix matchers.
# ---------------------------------------------------------------------------

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


def _collect_import_aliases(tree: ast.Module) -> Dict[str, str]:
    """Build a map of local-name → dotted-module-name from import statements.

    Recognizes bare-name decorators like ``@jit`` (from ``from triton import jit``)
    by resolving the alias back to its origin module. Only collects aliases that
    resolve to the Triton or TileLang namespace — unrelated ``@jit`` from other
    libraries won't be misclassified."""
    targets = frozenset({_TRITON_MODULE_NAME, _TL_MODULE_NAME})
    aliases: Dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split('.')[0] in targets:
                    aliases[a.asname or a.name] = a.name
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split('.')[0] in targets:
                for a in node.names:
                    aliases[a.asname or a.name] = f"{node.module}.{a.name}"
    return aliases


def _is_triton_decorator(node: ast.expr,
                         import_aliases: Optional[Dict[str, str]] = None) -> bool:
    """True for ``@triton.jit`` / ``@triton.<dec>`` / ``@jit`` (when ``from
    triton import jit``). Handles bare name, dotted attribute, and called
    decorator forms."""
    if isinstance(node, ast.Call):
        return _is_triton_decorator(node.func, import_aliases)
    if isinstance(node, ast.Attribute):
        return (
            isinstance(node.value, ast.Name)
            and node.value.id == _TRITON_MODULE_NAME
            and node.attr in _TRITON_DECORATORS
        )
    if isinstance(node, ast.Name) and import_aliases:
        resolved = import_aliases.get(node.id, "")
        parts = resolved.rsplit(".", 1)
        if (len(parts) == 2 and parts[0] == _TRITON_MODULE_NAME
                and parts[1] in _TRITON_DECORATORS):
            return True
    return False


def _is_tilelang_decorator(node: ast.expr,
                           import_aliases: Optional[Dict[str, str]] = None) -> bool:
    """True for ``@tilelang.jit`` / ``@jit`` (when ``from tilelang import
    jit``). Mirrors :func:`_is_triton_decorator` for the TileLang
    namespace + decorator set defined in ``code_checker.yaml``."""
    if isinstance(node, ast.Call):
        return _is_tilelang_decorator(node.func, import_aliases)
    if isinstance(node, ast.Attribute):
        return (
            isinstance(node.value, ast.Name)
            and node.value.id == _TL_MODULE_NAME
            and node.attr in _TL_DECORATORS
        )
    if isinstance(node, ast.Name) and import_aliases:
        resolved = import_aliases.get(node.id, "")
        parts = resolved.rsplit(".", 1)
        if (len(parts) == 2 and parts[0] == _TL_MODULE_NAME
                and parts[1] in _TL_DECORATORS):
            return True
    return False


def _find_tilelang_kernel_calls(tree: ast.Module, kernel_names: set) -> set:
    """Find tilelang kernel invocations. Two patterns:

    1. ``kernel = kernel_func(params); kernel(inputs)`` (factory returns
       compiled kernel) or inlined ``kernel_func(params)(inputs)``.
    2. ``compiled = tilelang.compile(func, target=...); compiled(inputs)``.

    Returns the subset of ``kernel_names`` proven to be launched (or all
    of them when a tilelang.compile-result is called — we can't statically
    tell which one)."""
    launched: set = set()
    compile_result_names: set = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if not isinstance(target, ast.Name):
                    continue
                if (isinstance(node.value, ast.Call)
                        and isinstance(node.value.func, ast.Attribute)):
                    func = node.value.func
                    if (isinstance(func.value, ast.Name)
                            and func.value.id == _TL_MODULE_NAME
                            and func.attr == "compile"):
                        compile_result_names.add(target.id)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in kernel_names:
                launched.add(node.func.id)
            elif node.func.id in compile_result_names:
                launched.update(kernel_names)
    return launched


def _match_torch_call_prefix(call_name: str) -> Optional[str]:
    """Return the longest matching torch-namespace prefix for ``call_name``,
    or None. Longer prefixes win (``torch.nn.functional`` before ``torch``)."""
    for prefix in _TORCH_CALL_PREFIXES_ORDERED:
        if call_name.startswith(f"{prefix}."):
            return prefix
    return None


def _fmt_calls(calls: List[tuple], limit: int = 5) -> str:
    """Render ``[(line, name), ...]`` as ``name(第line行), ... 等（共 N 处）``."""
    summary = ", ".join(f"{name}(第{line}行)" for line, name in calls[:limit])
    if len(calls) > limit:
        summary += f" 等（共 {len(calls)} 处）"
    return summary


@dataclass
class CheckError:
    """检查错误信息"""
    line: int
    error_type: str
    detail: str
    suggestion: str
    code_snippet: str
    fix_strategy: str = "fix"  # "fix" 或 "rewrite"


# ===========================================================================
# Compliance checks — each Check class owns its own policy state. Adding a
# new DSL anti-cheat = new ``_<dsl>ComplianceCheck`` subclass + 1 line in
# ``CodeChecker._CHECKS``. ``CodeChecker.__init__`` does NOT know any DSL.
# ===========================================================================


def _forbidden_compute_in_forward(forward_node, *, hard_etype, kernel_label,
                                  kernel_present=True, soft_etype=None,
                                  skip_prefix=None) -> List[Dict]:
    """Shared 'forbid torch high-level compute in ModelNew.forward()' AST scan
    for the Python DSL checks (triton / tilelang / pypto / catlass).

    HARD ops (and the ``@`` matmul operator) always fail. When ``soft_etype``
    is given, SOFT ops also fail if no kernel is launched (``kernel_present``
    False), otherwise just warn — pypto omits ``soft_etype`` (hard-only).
    ``skip_prefix`` exempts the DSL's own custom-op calls (catlass's
    ``torch.ops.catlass.*``). Returns error dicts; the caller owns the rest.
    """
    hard_calls: List[tuple] = []
    soft_calls: List[tuple] = []
    for node in ast.walk(forward_node):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            call_name = _dotted_name(node.func)
            if skip_prefix and call_name and call_name.startswith(skip_prefix):
                continue
            if not call_name or not _match_torch_call_prefix(call_name):
                continue
            method = node.func.attr
            if method in _TORCH_COMPUTE_OPS_HARD:
                hard_calls.append((node.lineno, call_name))
            elif soft_etype and method in _TORCH_COMPUTE_OPS_SOFT:
                soft_calls.append((node.lineno, call_name))
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
            hard_calls.append((node.lineno, "@ (matmul operator)"))

    errors: List[Dict] = []
    if hard_calls:
        errors.append({
            "line": hard_calls[0][0],
            "error_type": hard_etype,
            "detail": (
                f"forward() 中使用了 {len(hard_calls)} 个不允许的 torch 高层计算 API: "
                f"{_fmt_calls(hard_calls)}。矩阵乘法/卷积/归一化等核心计算必须在 "
                f"{kernel_label} 内实现。"
            ),
            "suggestion": (
                f"请将这些核心计算移入 {kernel_label}，forward() 仅负责准备输入、"
                f"调用 kernel 和整理输出。"
            ),
            "code_snippet": "",
            "fix_strategy": "rewrite",
        })
    if soft_etype and soft_calls:
        if not kernel_present:
            errors.append({
                "line": soft_calls[0][0],
                "error_type": soft_etype,
                "detail": (
                    f"forward() 中使用了 {len(soft_calls)} 个 torch 计算 API: "
                    f"{_fmt_calls(soft_calls)}。同时未调用 {kernel_label}，"
                    f"代码很可能用 torch API 替代了 kernel 实现。"
                ),
                "suggestion": (
                    "请将核心计算逻辑用 kernel 实现；简单操作（exp/relu/sum 等）若只是 "
                    "kernel 的前后处理可保留，但必须有 kernel 承担主要计算。"
                ),
                "code_snippet": "",
                "fix_strategy": "rewrite",
            })
        else:
            logger.warning(
                f"CodeChecker compliance: forward() 调用了 {kernel_label}，同时包含 "
                f"{len(soft_calls)} 处 torch 辅助计算 API: {_fmt_calls(soft_calls)}。"
                f"（融合算子可能合理，仅记录警告）"
            )
    return errors


class _ComplianceCheck:
    """Base for per-DSL/per-arch static checks. Subclasses load their own
    policy literals in ``__init__`` (from ``_POLICY``), declare when to
    fire via ``applies(checker)``, and produce error dicts in ``run(code,
    checker)``. State is shared across all CodeChecker instances (the
    policy is module-immutable; checks have no per-instance mutable state)."""

    name: str = ""

    def applies(self, checker: "CodeChecker") -> bool:  # noqa: F821
        return True

    def run(self, code: str, checker: "CodeChecker") -> List[Dict]:  # noqa: F821
        raise NotImplementedError


def _err(line: int, error_type: str, detail: str, suggestion: str,
         *, snippet: str = "", fix: str = "rewrite") -> Dict:
    """Build a CodeChecker error dict — the shape shared by every check."""
    return {"line": line, "error_type": error_type, "detail": detail,
            "suggestion": suggestion, "code_snippet": snippet, "fix_strategy": fix}


def _decorated_functions(tree: ast.Module, predicate, aliases) -> set:
    """Names of functions carrying a decorator matched by ``predicate``."""
    out: set = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if any(predicate(dec, aliases) for dec in node.decorator_list):
                out.add(node.name)
    return out


class _KernelDSLComplianceCheck(_ComplianceCheck):
    """Shared skeleton for the AST-based per-DSL anti-cheat (triton / tilelang /
    pypto / catlass): a kernel must be DEFINED and USED, and ModelNew.forward()
    must not delegate core compute to a torch high-level API.

    A subclass supplies only DATA + two small strategy hooks: which dsl it matches
    (``dsl_prefix`` startswith, or ``dsl_exact``), how to spot kernels
    (:meth:`_find_kernels`) and their use (:meth:`_find_used`), the error-type
    names + messages, and the forbidden-compute etypes. The run() flow — parse →
    'no kernel' → 'kernel not called' → the shared ``_forbidden_compute_in_forward``
    scan — lives here once, so a new AST DSL is ~15 lines."""

    dsl_prefix: Optional[str] = None      # startswith match (triton / tilelang)
    dsl_exact: Optional[str] = None       # exact match (pypto / catlass)
    no_kernel_etype: str = ""
    not_called_etype: str = ""            # "" → skip the 'not called' stage
    hard_etype: str = ""
    soft_etype: Optional[str] = None
    kernel_label: str = ""
    skip_prefix: Optional[str] = None
    # (detail_template, suggestion). detail may reference {dsl} / {kernels}.
    no_kernel_msg: Tuple[str, str] = ("", "")
    not_called_msg: Tuple[str, str] = ("", "")

    def applies(self, checker: "CodeChecker") -> bool:  # noqa: F821
        if self.dsl_prefix is not None:
            return checker.dsl.startswith(self.dsl_prefix)
        return checker.dsl == self.dsl_exact

    def _find_kernels(self, tree: ast.Module, aliases: Dict[str, str]) -> set:
        """Return the set of kernel-bearing names (empty → 'no kernel')."""
        raise NotImplementedError

    def _find_used(self, tree: ast.Module, kernels: set) -> set:
        """Return the subset of ``kernels`` proven to be launched/called."""
        raise NotImplementedError

    def run(self, code: str, checker: "CodeChecker") -> List[Dict]:  # noqa: F821
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return []
        aliases = _collect_import_aliases(tree)

        kernels = self._find_kernels(tree, aliases)
        if not kernels:
            detail, suggestion = self.no_kernel_msg
            return [_err(0, self.no_kernel_etype,
                         detail.format(dsl=checker.dsl), suggestion)]

        errors: List[Dict] = []
        used = self._find_used(tree, kernels) if self.not_called_etype else kernels
        if self.not_called_etype and not used:
            detail, suggestion = self.not_called_msg
            errors.append(_err(0, self.not_called_etype,
                               detail.format(kernels=sorted(kernels)), suggestion))

        model_cls = _find_model_new_class(tree)
        if model_cls is None:
            return errors
        forward_node = _find_forward(model_cls)
        if forward_node is None:
            return errors
        errors.extend(_forbidden_compute_in_forward(
            forward_node, hard_etype=self.hard_etype, soft_etype=self.soft_etype,
            kernel_label=self.kernel_label, kernel_present=bool(used),
            skip_prefix=self.skip_prefix))
        return errors


class _TritonComplianceCheck(_KernelDSLComplianceCheck):
    """triton (triton_cuda / triton_ascend): a ``@triton.jit`` kernel must be
    defined AND launched via ``kernel[grid](...)``; forward() no hard torch API."""

    name = "triton_compliance"
    dsl_prefix = "triton"
    no_kernel_etype = "no_triton_kernel"
    not_called_etype = "triton_kernel_not_called"
    hard_etype = "torch_api_instead_of_kernel"
    soft_etype = "torch_api_without_kernel"
    kernel_label = "triton kernel"
    no_kernel_msg = (
        "DSL 指定为 {dsl}，但代码中未找到任何 @triton.jit 装饰的 kernel 函数。"
        "代码可能使用了 torch 高层 API 替代 triton kernel 实现。",
        "请确保代码中包含至少一个 @triton.jit 装饰的 kernel 函数，"
        "并在 ModelNew.forward() 中通过 kernel[grid](...) 语法调用它。")
    not_called_msg = (
        "定义了 triton kernel 函数 {kernels}，但代码中未找到任何 kernel[grid](...) "
        "形式的调用。kernel 函数可能只是装饰性的，实际计算未使用 triton。",
        "请在 ModelNew.forward() 或其辅助方法中，"
        "通过 kernel_name[grid_size](...) 语法启动 triton kernel。")

    def _find_kernels(self, tree, aliases):
        return _decorated_functions(tree, _is_triton_decorator, aliases)

    def _find_used(self, tree, kernels):
        # kernel[grid](args) -> Call(func=Subscript(value=Name)). Scan whole file
        # so a launch inside a helper (not just forward) still counts.
        used: set = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Subscript):
                value = node.func.value
                if isinstance(value, ast.Name) and value.id in kernels:
                    used.add(value.id)
        return used


class _TilelangComplianceCheck(_KernelDSLComplianceCheck):
    """tilelang (tilelang_cuda / tilelang_ascend / tilelang_npuir): a
    ``@tilelang.jit`` kernel must be defined AND called (factory ``kernel(...)``
    or a ``tilelang.compile`` result); forward() no hard torch API. Unlike triton
    there is no ``kernel[grid]`` launch — see :func:`_find_tilelang_kernel_calls`."""

    name = "tilelang_compliance"
    dsl_prefix = "tilelang"
    no_kernel_etype = "no_tilelang_kernel"
    not_called_etype = "tilelang_kernel_not_called"
    hard_etype = "torch_api_instead_of_tilelang_kernel"
    soft_etype = "torch_api_without_tilelang_kernel"
    kernel_label = "tilelang kernel"
    no_kernel_msg = (
        "DSL 指定为 {dsl}，但代码中未找到任何 @tilelang.jit 装饰的 kernel 函数。"
        "代码可能使用了 torch 高层 API 替代 tilelang kernel 实现（torch 退化）。",
        "请确保代码中包含至少一个 @tilelang.jit 装饰的 kernel 函数，"
        "并在 ModelNew.forward() 中调用编译后的 kernel 执行计算。")
    not_called_msg = (
        "定义了 tilelang kernel 函数 {kernels}，但代码中未找到任何 kernel 调用。"
        "kernel 函数可能只是装饰性的，实际计算未使用 tilelang（torch 退化）。",
        "请在 ModelNew.forward() 中调用编译后的 tilelang kernel 执行计算，例如：\n"
        "  kernel = my_kernel(M, N, K)\n  kernel(A, B, C)\n"
        "或使用 tilelang.compile 编译后调用：\n"
        "  compiled = tilelang.compile(func, target='npuir')\n  compiled(A, B, C)")

    def _find_kernels(self, tree, aliases):
        return _decorated_functions(tree, _is_tilelang_decorator, aliases)

    def _find_used(self, tree, kernels):
        return _find_tilelang_kernel_calls(tree, kernels)


class _PyptoComplianceCheck(_KernelDSLComplianceCheck):
    """PyPTO (deliberately lenient): a ``@pypto(...).jit`` kernel — or the factory
    wrapping it — must be defined AND called; forward() no hard torch API. No soft
    tier. The decorator namespace/attr come from ``pypto_compliance`` policy."""

    name = "pypto_compliance"
    no_kernel_etype = "no_pypto_kernel"
    not_called_etype = "pypto_kernel_not_called"
    hard_etype = "torch_api_instead_of_kernel"
    kernel_label = "pypto kernel"
    no_kernel_msg = (
        "DSL 指定为 {dsl}，但代码中未找到任何 @pypto.frontend.jit "
        "（或 @pypto.jit）装饰的 kernel，疑似直接用 torch 实现替代了 pypto。",
        "请用 @pypto.frontend.jit 定义 kernel，在 kernel 内通过 pypto.* "
        "算子完成核心计算，并在 ModelNew.forward() 中调用该 kernel。")
    not_called_msg = (
        "定义了 pypto kernel（或其工厂）{kernels}，"
        "但全文件未找到任何对它们的调用，kernel 可能只是摆设，实际计算落回了 torch。",
        "请在 ModelNew.forward() 中实际调用 pypto kernel（或构造它的工厂函数），"
        "由 kernel 承担核心计算，而不是用 torch 算子直接出结果。")

    def __init__(self):
        _p = _POLICY["pypto_compliance"]
        self.dsl_exact = _p["dsl"]
        self._dec_module = _p["decorator_module"]
        self._dec_attr = _p["decorator_attr"]

    def _is_pypto_kernel_decorator(self, node: ast.expr) -> bool:
        """True for ``@pypto.jit`` / ``@pypto.frontend.jit`` (any ``pypto.*.jit``
        spelling, with or without call args) — root module + trailing attr."""
        target = node.func if isinstance(node, ast.Call) else node
        name = _dotted_name(target)
        if not name:
            return False
        parts = name.split(".")
        return parts[0] == self._dec_module and parts[-1] == self._dec_attr

    def _find_kernels(self, tree, aliases):
        # kernel_bearing = the wrapping factory name (either it or the kernel
        # being called counts as used).
        out: set = set()
        for outer in ast.walk(tree):
            if not isinstance(outer, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for inner in ast.walk(outer):
                if not isinstance(inner, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if any(self._is_pypto_kernel_decorator(d) for d in inner.decorator_list):
                    out.add(outer.name)
                    break
        return out

    def _find_used(self, tree, kernels):
        called: set = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                called.add(node.func.id)
        return kernels & called


class _CatlassComplianceCheck(_ComplianceCheck):
    """ascendc_catlass 反作弊：ModelNew 中必须出现 torch.ops.catlass.xxx
    调用，forward() 不允许 torch 高层硬算子（除合法的 catlass 调用本身）。"""

    name = "catlass_compliance"

    def __init__(self):
        _c = _POLICY["catlass_compliance"]
        self._dsl: str = _c["dsl"]
        self._enabled: bool = bool(_c["enable_catlass_call_check"])
        self._call_ns: str = _c["call_namespace"]
        self._call_prefix: str = self._call_ns + "."

    def applies(self, checker: "CodeChecker") -> bool:  # noqa: F821
        return checker.dsl == self._dsl and self._enabled

    def run(self, code: str, checker: "CodeChecker") -> List[Dict]:  # noqa: F821
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return []

        model_cls = _find_model_new_class(tree)
        if model_cls is None:
            return []

        forward_node = _find_forward(model_cls)
        if forward_node is None:
            return []

        errors: List[Dict] = []

        # --- A. forward() must call torch.ops.catlass.xxx ---
        has_catlass_call = False
        for node in ast.walk(forward_node):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                call_name = _dotted_name(node.func)
                if call_name and call_name.startswith(self._call_prefix):
                    has_catlass_call = True
                    break

        if not has_catlass_call:
            errors.append(_err(
                0, "no_catlass_call",
                f"DSL 指定为 {checker.dsl}，但 ModelNew.forward() 中未找到任何 "
                f"{self._call_ns}.xxx 形式的调用。"
                f"代码可能使用了 torch 高层 API 替代 catlass kernel 实现。",
                "请确保 forward() 中通过 torch.ops.catlass.<op_name>(...) "
                "调用 catlass kernel，而非直接使用 torch 高层计算 API。"))

        errors.extend(_forbidden_compute_in_forward(
            forward_node, hard_etype="torch_api_instead_of_kernel",
            soft_etype="torch_api_without_kernel", kernel_label="catlass kernel",
            kernel_present=has_catlass_call, skip_prefix=self._call_prefix))
        return errors


# AscendC anti-cheat messages — kept here (not the policy YAML) to match the
# other DSL compliance checks, which build their detail/suggestion in code.
# Keyed by the pattern ``name`` in code_checker.yaml's ``forbidden_patterns``.
# Only the two BLOCKING patterns that bypass ATen dispatch entirely (raw ACL /
# torch_npu builtins). Everything reaching dispatch is disabled at runtime by the
# compute gate (runtime_guard/), not by source spelling.
_ASCENDC_PATTERN_MESSAGES: dict = {
    'torch_npu_builtin_compute': (
        'AscendC wrapper 调用了 torch_npu.npu_* 内置算子，等价于委托给现成 NPU op。',
        '改为 torch.ops.npu.<custom_op>(...) 调用自定义 direct-invoke 算子，核心计算写在 ascendc_op 中。'),
    'aclnn_builtin_compute': (
        'AscendC host extension 调用了 aclnn 内置高层计算 API。',
        '不要用 aclnn 内置算子代替自定义 AscendC kernel。'),
}


class _SourcePatternComplianceCheck(_ComplianceCheck):
    """Shared skeleton for the TEXT/regex source-scan anti-cheat — for calls that
    bypass ATen dispatch (raw ``aclnn*`` ACL API / ``torch_npu.npu_*`` builtins)
    which neither an AST check nor the runtime compute gate can see, so they must
    be caught in the source text. Regex- not AST-based, because it also scans
    ``.cpp/.h/.asc/CMake`` (non-Python) files.

    Comments and string literals are stripped first, so a forbidden name that
    appears only in a comment/string (``// avoid aclnnMatmul(...)``) is not a
    false hit; only a genuine call site matches. A subclass supplies: which dsl it
    matches (``dsl_set``), which file suffixes/names to scan, the error-type
    prefix, and the ``(name, compiled_regex, detail, suggestion)`` patterns."""

    dsl_set: frozenset = frozenset()
    text_suffixes: frozenset = frozenset()
    text_filenames: frozenset = frozenset()
    etype_prefix: str = ""
    patterns: tuple = ()

    def applies(self, checker: "CodeChecker") -> bool:  # noqa: F821
        return checker.dsl in self.dsl_set

    @staticmethod
    def _scan_line(line: str) -> bool:
        stripped = line.strip()
        if not stripped:
            return False
        return not stripped.startswith(("#", "//", "/*", "*", "*/"))

    @staticmethod
    def _sanitize_line(line: str) -> str:
        """Strip string/char literals and comments from a line before the scan.
        Single-pass, string-state aware so a ``//`` inside a string is not a
        comment and a quote inside a comment does not open a string; handles
        ``#``/``//`` line comments + single-line ``/* ... */`` blocks. Spacing is
        left intact so spaced-dodge patterns still match real calls."""
        out: List[str] = []
        i = 0
        n = len(line)
        quote = None
        while i < n:
            c = line[i]
            if quote is not None:
                if c == "\\" and i + 1 < n:
                    i += 2
                    continue
                if c == quote:
                    quote = None
                i += 1
                continue
            if c in ("\"", "'"):
                quote = c
                i += 1
                continue
            if c == "#":
                break
            if c == "/" and i + 1 < n and line[i + 1] == "/":
                break
            if c == "/" and i + 1 < n and line[i + 1] == "*":
                end = line.find("*/", i + 2)
                if end == -1:
                    break
                i = end + 2
                continue
            out.append(c)
            i += 1
        return "".join(out)

    def _should_scan_file(self, checker: "CodeChecker") -> bool:  # noqa: F821
        source_path = getattr(checker, "_source_path", "") or ""
        if not source_path:
            return True
        name = os.path.basename(source_path)
        suffix = os.path.splitext(name)[1]
        return suffix in self.text_suffixes or name in self.text_filenames

    def run(self, code: str, checker: "CodeChecker") -> List[Dict]:  # noqa: F821
        if not self._should_scan_file(checker):
            return []
        errors: List[Dict] = []
        for line_no, line in enumerate(code.splitlines(), 1):
            if not self._scan_line(line):
                continue
            # Scan the code-only text (comments / strings stripped); report the
            # original line.
            scan_target = self._sanitize_line(line)
            if not scan_target.strip():
                continue
            for name, pattern, detail, suggestion in self.patterns:
                if pattern.search(scan_target):
                    errors.append(_err(line_no, f"{self.etype_prefix}{name}",
                                       detail, suggestion, snippet=line.rstrip()))
        return errors


class _AscendCComplianceCheck(_SourcePatternComplianceCheck):
    """AscendC-family (ascendc + ascendc_catlass) source-scan for calls that
    bypass ATen dispatch (raw ``aclnn*`` / ``torch_npu.npu_*``). Both are
    directory-backed C++ and share the same 'no raw stock-kernel' rule, so one
    scan covers both. Everything reaching dispatch (Python / C++ ``torch::*`` /
    ``at::*`` nested in the candidate's own custom op) is disabled at runtime by
    the compute gate (runtime_guard/) — both adapters emit ``guarded_call``, so
    ascendc and catlass are now fully symmetric. Pattern messages live in
    ``_ASCENDC_PATTERN_MESSAGES`` above."""

    name = "ascendc_compliance"
    etype_prefix = "ascendc_anti_cheat_"

    def __init__(self):
        _a = _POLICY["ascendc_anti_cheat"]
        # Cover both AscendC-family DSLs so catlass's .cpp is scanned too (closes
        # the earlier asymmetry where catlass raw-aclnn was neither statically
        # scanned nor runtime-gated).
        self.dsl_set = frozenset({_a["dsl"], _POLICY["catlass_compliance"]["dsl"]})
        self.text_suffixes = _ASCENDC_TEXT_SUFFIXES
        self.text_filenames = _ASCENDC_TEXT_FILENAMES
        self.patterns = tuple(
            (item["name"], re.compile(item["pattern"]),
             *_ASCENDC_PATTERN_MESSAGES[item["name"]])
            for item in _a["forbidden_patterns"]
        )


class _AutotuneComplianceCheck(_ComplianceCheck):
    """triton 系列：``@triton.autotune`` 装饰器必须包含 ``restore_value``
    参数（否则 benchmark 重跑会跨 config 污染输出）。"""

    name = "autotune"

    def __init__(self):
        _a = _POLICY["autotune"]
        self._autotune_re = re.compile(
            rf"@{re.escape(_TRITON_MODULE_NAME)}\."
            rf"{re.escape(_a['decorator_attr'])}\s*\(",
            re.MULTILINE,
        )
        self._restore_value_re = re.compile(
            rf"{re.escape(_a['required_kwarg'])}\s*="
        )

    def applies(self, checker: "CodeChecker") -> bool:  # noqa: F821
        # ``@triton.autotune`` 是 triton-specific —— tilelang DSL 不走 autotune。
        return checker.dsl.startswith("triton")

    def run(self, code: str, checker: "CodeChecker") -> List[Dict]:  # noqa: F821
        errors: List[Dict] = []

        autotune_match = self._autotune_re.search(code)
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

        if not self._restore_value_re.search(autotune_block):
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


class _A5ComplianceCheck(_ComplianceCheck):
    """A5 (Ascend950) 硬件 + triton_ascend：含 tl.dot 的 kernel 必须
    使用 Cube/Vector 亲和接口 (al.scope / al.fixpipe / bl.alloc)。"""

    name = "a5_compliance"

    def __init__(self):
        _a = _POLICY["a5_compliance"]
        self._arch_prefix: str = _a["arch_prefix"]
        self._dsl: str = _a["dsl"]
        self._enabled: bool = bool(_a["enable_triton_ascend_affinity_check"])
        self._al_alias: str = _a["aliases"]["al"]
        self._bl_alias: str = _a["aliases"]["bl"]
        self._only_apis: frozenset = frozenset(_a["only_apis"])

    def applies(self, checker: "CodeChecker") -> bool:  # noqa: F821
        return (
            checker.arch.startswith(self._arch_prefix)
            and checker.dsl == self._dsl
        )

    @staticmethod
    def _kernel_uses_tl_dot(kernel: ast.AST) -> bool:
        """Return True if the kernel body contains any ``tl.dot(...)``
        call (or the spelled-out ``triton.language.dot(...)``)."""
        for node in ast.walk(kernel):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "dot"
                and isinstance(func.value, ast.Name)
                and func.value.id == "tl"
            ):
                return True
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

    def run(self, code: str, checker: "CodeChecker") -> List[Dict]:  # noqa: F821
        if not self._enabled:
            logger.info(
                f"CodeChecker A5: arch={checker.arch}, dsl={checker.dsl} — "
                "affinity enforcement disabled via "
                "a5_compliance.enable_triton_ascend_affinity_check=false; "
                "skipping check."
            )
            return []

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return []

        triton_kernels: List[ast.FunctionDef] = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for dec in node.decorator_list:
                    if _is_triton_decorator(dec):
                        triton_kernels.append(node)
                        break

        if not triton_kernels:
            return []

        cube_required = any(self._kernel_uses_tl_dot(k) for k in triton_kernels)
        if not cube_required:
            logger.info(
                f"CodeChecker A5: arch={checker.arch}, dsl={checker.dsl} — no tl.dot "
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
                    if prefix == self._al_alias:
                        if method == "scope":
                            has_al_scope = True
                        elif method == "fixpipe":
                            has_fixpipe = True
                    elif prefix == self._bl_alias:
                        if method == "alloc":
                            has_bl_alloc = True

                # 链式调用：al.<x>.<method>(...) —— 例如 ``al.something.scope(...)``。
                # 受 ``only_apis`` 白名单约束，避免和无关的 ``al.foo.bar()`` 混淆。
                if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Attribute):
                    inner = func.value
                    if (isinstance(inner.value, ast.Name)
                            and inner.value.id == self._al_alias
                            and func.attr in self._only_apis):
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
                    f"目标架构为 {checker.arch}（A5 硬件），但 kernel 中未使用 al.scope(core_mode=...) "
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
                    f"目标架构为 {checker.arch}（A5 硬件），kernel 使用了 al.scope 但未调用 al.fixpipe。"
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
                    f"目标架构为 {checker.arch}（A5 硬件），但 kernel 中未使用 bl.alloc 分配片上 buffer。"
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


# ===========================================================================
# CodeChecker class
# ===========================================================================

class CodeChecker:
    """
    代码检查器：在 Coder 生成代码后、Verifier 验证前，进行快速的纯静态检查

    检查流程：ast.parse → py_compile → import 验证 → 中文文本混入检测
    → DSL/arch 合规性检测。不调用 LLM，零额外成本。

    新增 per-DSL 合规检查的方法：实现一个 ``_<dsl>ComplianceCheck`` 子类
    （定义 ``applies(checker) / run(code, checker)``），在 ``_CHECKS`` 列
    表里追加一行；不要在 ``CodeChecker`` 类体内添加 per-DSL 字段或方法。
    """

    # Class-level singleton instances of each compliance check. State is
    # immutable (yaml policy frozensets / compiled regex) so sharing
    # across CodeChecker instances is safe.
    _triton_check = _TritonComplianceCheck()
    _tilelang_check = _TilelangComplianceCheck()
    _pypto_check = _PyptoComplianceCheck()
    _catlass_check = _CatlassComplianceCheck()
    _ascendc_check = _AscendCComplianceCheck()
    _autotune_check = _AutotuneComplianceCheck()
    _a5_check = _A5ComplianceCheck()

    # All compliance checks. Iteration order shows up in error rendering.
    _CHECKS: list = [
        _triton_check, _tilelang_check, _pypto_check, _catlass_check,
        _ascendc_check, _autotune_check, _a5_check,
    ]
    # Subset exposed via the ``_check_dsl_compliance`` public method
    # (called by autoresearch agent tools). Excludes autotune + A5,
    # which have their own dimensions (DSL prefix / arch+flag).
    _DSL_COMPLIANCE_CHECKS: list = [
        _triton_check, _tilelang_check, _pypto_check, _catlass_check,
        _ascendc_check,
    ]

    def __init__(self, backend: str, dsl: str, arch: str = "", config: Optional[dict] = None):
        self.backend = backend.lower() if backend else ""
        self.dsl = dsl.lower() if dsl else ""
        self.arch = arch.lower() if arch else ""
        # ``config`` accepted for caller-signature compat; policy 真源 is yaml.
        self.config = config or {}
        logger.info(
            f"CodeChecker initialized: backend={self.backend}, "
            f"dsl={self.dsl}, arch={self.arch}"
        )

    # ------------------------------------------------------------------
    # Compat surface — autoresearch agent tools + tests expect these
    # names. They are *not* per-instance state; the values are pinned
    # by op/config/code_checker.yaml at module load.
    # ------------------------------------------------------------------

    @property
    def triton_decorators(self) -> frozenset:
        return _TRITON_DECORATORS

    @property
    def torch_compute_ops_hard(self) -> frozenset:
        return _TORCH_COMPUTE_OPS_HARD

    @property
    def torch_compute_ops_soft(self) -> frozenset:
        return _TORCH_COMPUTE_OPS_SOFT

    @property
    def torch_call_prefixes(self) -> frozenset:
        return _TORCH_CALL_PREFIXES

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def _is_python_source(self, source_name: str) -> bool:
        """Whether this handoff file runs the Python-only steps (ast / compile /
        import / stray-text). A known non-.py text source — the C++/AscendC/
        CMake files of a directory-backed DSL — skips them and goes straight to
        the DSL compliance scan. Empty / unknown path defaults to Python."""
        if not source_name:
            return True
        suffix = os.path.splitext(source_name)[1]
        if suffix == ".py":
            return True
        return not (suffix in _ASCENDC_TEXT_SUFFIXES
                    or source_name in _ASCENDC_TEXT_FILENAMES)

    def check(self, code: str, task_info: Optional[dict] = None) -> Tuple[bool, str, List[Dict]]:
        """
        检查代码（纯静态检查，不调用 LLM）

        检查流程（步骤 1-4 仅对 Python 源执行；非 .py 文本源——目录式 DSL 的
        C++/AscendC/CMake 文件——直接跳到步骤 5，见 ``_is_python_source``）：
        1. ast.parse 语法检查
        2. py_compile 编译检查（语法通过后执行，捕获额外编译问题）
        3. import 可用性检查（代码可编译时执行）
        4. 中文文本混入检测
        5. DSL/arch 合规性检测：遍历 ``_CHECKS`` 调用每个 Check 实例的
           ``applies(self) / run(code, self)``（仅在无语法/编译错误时执行）

        Args:
            code: 要检查的代码
            task_info: 任务信息（``file``/``path`` 用于判定源类型）

        Returns:
            Tuple[bool, str, List[Dict]]:
                - passed: 是否通过检查
                - error_message: 格式化的错误信息（用于传递给 Coder）
                - errors: 详细错误列表
        """
        task_info = task_info or {}
        self._source_path = str(task_info.get("file") or task_info.get("path") or "")

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

        # Python-source gate via DSL adapter. Only ``ValueError`` from
        # the factory (unregistered DSL) is treated as "skip safely";
        # ImportError / AttributeError / 其它异常都是支持-中-DSL adapter
        # 自身的真问题，必须冒出来，不能被静默 swallow 成 "skip checker"。
        if self.dsl:
            from akg_agents.op.verifier.adapters.factory import get_dsl_adapter
            try:
                _adapter = get_dsl_adapter(self.dsl)
            except ValueError:
                _adapter = None
            if _adapter is None or not _adapter.static_check_via_python_ast:
                reason = "unknown DSL" if _adapter is None else "not Python-based"
                logger.info(f"CodeChecker: DSL '{self.dsl}' {reason}, skipping checks")
                return True, "", []

        # Steps 1-4 are Python-only (ast.parse / compile / import / stray-text).
        # A directory-backed DSL's non-.py sources (AscendC C++/AscendC/CMake)
        # skip straight to the DSL compliance scan in Step 5.
        errors: List[Dict] = []
        if self._is_python_source(os.path.basename(self._source_path)):
            errors = self._check_python_syntax(code)              # Step 1: ast.parse
            if not errors:
                errors.extend(self._check_py_compile(code))       # Step 2: py_compile
            if not errors:
                errors.extend(self._check_imports(code))          # Step 3: imports
            errors.extend(self._check_stray_chinese(code))        # Step 4: 中文混入

        # Step 5: DSL/arch compliance. Each Check owns its applies()/run()
        # (AscendC scans the C++/AscendC text; the others parse Python).
        has_syntax_err = any(
            e.get('error_type') in ('syntax_error', 'compile_error') for e in errors
        )
        if not has_syntax_err:
            for check in self._CHECKS:
                if check.applies(self):
                    errors.extend(check.run(code, self))

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
    # Public umbrella for autoresearch agent tools (op/autoresearch/
    # agent/tools.py). Runs ONLY the DSL anti-cheat subset (triton /
    # pypto / catlass) — excludes autotune and A5 which target other
    # dimensions.
    # ------------------------------------------------------------------

    def _check_dsl_compliance(self, code: str) -> List[Dict]:
        errors: List[Dict] = []
        for check in self._DSL_COMPLIANCE_CHECKS:
            if check.applies(self):
                errors.extend(check.run(code, self))
        return errors

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

    # Runtime modules that live on the eval target (NPU host), NOT on
    # the orchestrator that runs CodeChecker. Skip the find_spec gate
    # for them — a Windows / no-NPU orchestrator legitimately doesn't
    # have torch_npu / triton_ascend / etc. installed, and the kernel
    # is verified end-to-end by the remote worker anyway. Real typos
    # in user code surface there with a clear ImportError, not as a
    # silent reject here.
    _REMOTE_RUNTIME_MODULES = frozenset({
        "torch_npu",
        "triton_ascend",
        "tilelang",
        "swft",
        "pypto",
        "tbe",
        "te",
        "acl",
        "aclnnop",
    })

    # Some DSLs import a generic frontend package whose runtime is nevertheless
    # backend-specific.  In particular, Triton Ascend kernels spell their
    # imports as ``triton`` / ``triton.language`` even though that package only
    # exists in the remote Ascend environment.  Keep this policy keyed by DSL:
    # globally ignoring ``triton`` would also hide a broken local Triton-CUDA
    # setup, while requiring it for ``triton_ascend`` makes a Windows
    # orchestrator reject kernels that the remote worker can execute.
    _REMOTE_RUNTIME_MODULES_BY_DSL = {
        "triton_ascend": frozenset({"triton"}),
    }

    def _remote_runtime_modules(self) -> frozenset:
        """Modules whose availability is owned by the evaluation worker."""
        return self._REMOTE_RUNTIME_MODULES | self._REMOTE_RUNTIME_MODULES_BY_DSL.get(
            self.dsl, frozenset()
        )

    def _check_imports(self, code: str) -> List[Dict]:
        """
        检查代码中 import 语句引用的模块是否可用。

        通过 AST 提取所有 import / from ... import 语句，使用
        importlib.util.find_spec 验证顶层模块是否存在。``_REMOTE_RUNTIME_MODULES``
        里的模块跳过 —— 它们只在远端 NPU 测评机上有，本机 orchestrator
        缺它们不算 kernel 写错。
        """
        errors = []
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return errors

        checked = set()
        remote_runtime_modules = self._remote_runtime_modules()

        def _emit_error(line: int, module_name: str) -> None:
            errors.append({
                "line": line,
                "error_type": "import_error",
                "detail": f"模块 '{module_name}' 无法导入（环境中不存在此模块）",
                "suggestion": f"请检查模块名 '{module_name}' 是否拼写正确，或确认该模块是否需要安装",
                "code_snippet": "",
                "fix_strategy": "fix"
            })
            logger.warning(
                f"CodeChecker: import error at line {line}: "
                f"module '{module_name}' not found"
            )

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top_module = alias.name.split('.')[0]
                    if top_module in checked or top_module in remote_runtime_modules:
                        continue
                    checked.add(top_module)
                    if not self._is_module_available(top_module):
                        _emit_error(node.lineno, alias.name)

            elif isinstance(node, ast.ImportFrom):
                if node.level and node.level > 0:
                    continue
                if node.module:
                    top_module = node.module.split('.')[0]
                    if top_module in checked or top_module in remote_runtime_modules:
                        continue
                    checked.add(top_module)
                    if not self._is_module_available(top_module):
                        _emit_error(node.lineno, node.module)

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


# ---------------------------------------------------------------------------
# Back-compat module-level alias: ``op/agents/kernel_gen.py`` reads this
# at import time to pin its A5-affinity prompt branch. Defined after the
# class so it can resolve via ``CodeChecker._a5_check._enabled``.
# ---------------------------------------------------------------------------

_A5_ENABLE_AFFINITY_CHECK: bool = CodeChecker._a5_check._enabled
