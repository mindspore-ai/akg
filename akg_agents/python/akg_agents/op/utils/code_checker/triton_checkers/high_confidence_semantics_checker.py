from __future__ import annotations

import ast
from typing import Dict, List, Optional, Set, Tuple

from akg_agents.op.utils.code_checker.base import (
    CheckContext,
    Issue,
    Location,
    TritonDiagnosticChecker,
)

from .ast_utils import (
    extract_triton_info,
    is_constexpr_annotation,
    is_jit_kernel,
    resolve_canonical,
)


_TL_TENSOR_PRODUCERS = {
    "load",
    "dot",
    "arange",
    "zeros",
    "full",
    "where",
    "sum",
    "max",
    "min",
    "maximum",
    "minimum",
    "broadcast_to",
    "reshape",
    "expand_dims",
    "trans",
    "cumsum",
    "cumprod",
}

_ALLOC_FUNCS = {"zeros", "full", "empty"}


CHECKER_NAME = "high_confidence_semantics"
CHECKER_ID = "triton_high_confidence_semantics"
RULE_ID = "TRITON_HIGH_CONFIDENCE_SEMANTICS"


class TritonHighConfidenceSemanticsChecker(TritonDiagnosticChecker):
    name = CHECKER_NAME
    checker_id = CHECKER_ID
    rule_id = RULE_ID

    def run(self, code: str, ctx: CheckContext) -> List[Issue]:
        if not (ctx.dsl or "").lower().startswith("triton"):
            return []

        info = extract_triton_info(code)
        aliases = info.aliases
        star_imports = info.star_imports
        issues: List[Issue] = []

        for node in ast.walk(info.tree):
            if isinstance(node, ast.FunctionDef) and is_jit_kernel(node, aliases, star_imports):
                issues.extend(_KernelSemanticVisitor(aliases, star_imports).check(node))

        issues.extend(_check_duplicate_launch_args(info.kernels, info.uses))
        return issues


class _KernelSemanticVisitor:
    def __init__(self, aliases: Dict[str, str], star_imports: Set[str]) -> None:
        self.aliases = aliases
        self.star_imports = star_imports
        self.issues: List[Issue] = []
        self.constexpr_params: Set[str] = set()
        self.constexpr_locals: Dict[str, bool] = {}

    def check(self, kernel: ast.FunctionDef) -> List[Issue]:
        self.constexpr_params = self._collect_constexpr_params(kernel)
        self.constexpr_locals = self._collect_constexpr_locals(kernel)

        for node in ast.walk(kernel):
            if isinstance(node, (ast.Break, ast.Continue)):
                self._emit(
                    "TRITON_UNSUPPORTED_CONTROL_FLOW",
                    "Unsupported Triton kernel control flow",
                    "break/continue inside @triton.jit kernel is unsupported.",
                    node,
                    "Rewrite with structured if blocks or tensor masks.",
                    {"triton", "kernel", "control_flow"},
                )
            elif isinstance(node, ast.If):
                self._check_python_if(node)
            elif isinstance(node, ast.Call):
                self._check_program_id_axis(node)
                self._check_static_range(node)
                self._check_alloc_shape(node)
                self._check_expand_dims_axis(node)

        return self.issues

    def _collect_constexpr_params(self, kernel: ast.FunctionDef) -> Set[str]:
        out: Set[str] = set()
        for arg in list(kernel.args.args) + list(kernel.args.kwonlyargs):
            if is_constexpr_annotation(arg.annotation, self.aliases, self.star_imports):
                out.add(arg.arg)
        return out

    def _collect_constexpr_locals(self, kernel: ast.FunctionDef) -> Dict[str, bool]:
        out: Dict[str, bool] = {}
        for node in ast.walk(kernel):
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                out[node.targets[0].id] = self._is_constexpr_value(node.value)
        return out

    def _check_python_if(self, node: ast.If) -> None:
        if not self._test_involves_tensor(node.test):
            return
        self._emit(
            "TRITON_RUNTIME_PY_IF",
            "Tensor-dependent Python if in Triton kernel",
            "if condition involves a tl.tensor, which is data-dependent vector branching.",
            node,
            "Use tl.where for tensor-conditional values or masks for memory operations.",
            {"triton", "kernel", "control_flow"},
        )

    def _test_involves_tensor(self, test: ast.AST) -> bool:
        for child in ast.walk(test):
            if not isinstance(child, ast.Call):
                continue
            canonical = resolve_canonical(child.func, self.aliases, self.star_imports)
            if not canonical or not canonical.startswith("triton.language."):
                continue
            if canonical.split(".")[-1] in _TL_TENSOR_PRODUCERS:
                return True
        return False

    def _check_program_id_axis(self, call: ast.Call) -> None:
        if resolve_canonical(call.func, self.aliases, self.star_imports) != "triton.language.program_id":
            return

        axis = call.args[0] if call.args else None
        for kw in call.keywords:
            if kw.arg == "axis":
                axis = kw.value
                break
        if axis is None:
            return
        if isinstance(axis, ast.Constant) and isinstance(axis.value, int) and axis.value in (0, 1, 2):
            return
        self._emit(
            "TRITON_PROGRAM_ID_AXIS",
            "Invalid tl.program_id axis",
            "tl.program_id(axis) expects a literal 0, 1, or 2.",
            call,
            "Use tl.program_id(0/1/2); flatten/unflatten ids for higher-dimensional indexing.",
            {"triton", "kernel", "api"},
        )

    def _check_static_range(self, call: ast.Call) -> None:
        if resolve_canonical(call.func, self.aliases, self.star_imports) != "triton.language.static_range":
            return
        for index, arg in enumerate(call.args):
            if self._is_constexpr_value(arg):
                continue
            label = "stop" if len(call.args) == 1 and index == 0 else ("start" if index == 0 else "stop")
            self._emit(
                "TRITON_STATIC_RANGE_NON_CONSTEXPR",
                "tl.static_range with non-constexpr argument",
                f"tl.static_range {label} argument '{ast.unparse(arg)}' is not a compile-time constant.",
                call,
                "Pass the bound as tl.constexpr, compute it from constexpr values, or use plain range().",
                {"triton", "kernel", "constexpr"},
            )

    def _check_alloc_shape(self, call: ast.Call) -> None:
        canonical = resolve_canonical(call.func, self.aliases, self.star_imports)
        if not canonical or not canonical.startswith("triton.language."):
            return
        fn = canonical.split(".")[-1]
        if fn not in _ALLOC_FUNCS or not call.args:
            return

        bad_dims = self._find_non_constexpr_dims(call.args[0])
        if not bad_dims:
            return
        dims = ", ".join(f"dim {idx}: {ast.unparse(expr)}" for idx, expr in bad_dims)
        self._emit(
            "TRITON_DYNAMIC_SHAPE_IN_ALLOC",
            "Dynamic shape in Triton allocation",
            f"tl.{fn}() shape contains runtime-dependent dimension(s): {dims}.",
            call,
            "Use fixed constexpr BLOCK sizes for tensor shapes and handle boundaries with masks.",
            {"triton", "kernel", "shape"},
        )

    def _check_expand_dims_axis(self, call: ast.Call) -> None:
        if resolve_canonical(call.func, self.aliases, self.star_imports) != "triton.language.expand_dims":
            return

        axis = call.args[1] if len(call.args) >= 2 else None
        for kw in call.keywords:
            if kw.arg == "axis":
                axis = kw.value
                break
        if not isinstance(axis, (ast.Tuple, ast.List)):
            return
        self._emit(
            "TRITON_EXPAND_DIMS_TUPLE_AXIS",
            "tl.expand_dims with tuple/list axis",
            f"tl.expand_dims axis argument is {ast.unparse(axis)}; Triton expects a single integer axis.",
            call,
            "Chain multiple tl.expand_dims calls, one per axis, or reshape directly.",
            {"triton", "kernel", "shape"},
        )

    def _find_non_constexpr_dims(self, shape_node: ast.AST) -> List[Tuple[int, ast.AST]]:
        if isinstance(shape_node, (ast.Tuple, ast.List)):
            return [
                (index, item)
                for index, item in enumerate(shape_node.elts)
                if not self._is_constexpr_value(item)
            ]
        if not self._is_constexpr_value(shape_node):
            return [(0, shape_node)]
        return []

    def _is_constexpr_value(self, node: ast.AST) -> bool:
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return True
        if isinstance(node, ast.Name):
            if node.id in self.constexpr_params:
                return True
            if node.id in self.constexpr_locals:
                return self.constexpr_locals[node.id]
            return node.id.isupper() and len(node.id) >= 2
        if isinstance(node, ast.BinOp):
            return self._is_constexpr_value(node.left) and self._is_constexpr_value(node.right)
        if isinstance(node, ast.UnaryOp):
            return self._is_constexpr_value(node.operand)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"min", "max"}:
            return all(self._is_constexpr_value(arg) for arg in node.args)
        return False

    def _emit(
        self,
        rule_id: str,
        title: str,
        message: str,
        node: ast.AST,
        hint: str,
        tags: Set[str],
    ) -> None:
        self.issues.append(
            Issue(
                severity="ERROR",
                rule_id=rule_id,
                title=title,
                message=message,
                location=Location(
                    lineno=getattr(node, "lineno", -1),
                    col=getattr(node, "col_offset", 0),
                    end_lineno=getattr(node, "end_lineno", -1),
                    end_col=getattr(node, "end_col_offset", 0),
                ),
                hint=hint,
                tags=tags,
            )
        )


def _check_duplicate_launch_args(kernels: Dict[str, List[str]], uses) -> List[Issue]:
    issues: List[Issue] = []
    for use in uses:
        if use.kind != "launch" or use.kernel_name not in kernels:
            continue
        params = kernels[use.kernel_name]
        occupied = set(params[: use.args_count])
        duplicate = sorted({kw for kw in use.keywords if kw in occupied})
        for keyword in duplicate:
            index = params.index(keyword) + 1
            issues.append(
                Issue(
                    severity="ERROR",
                    rule_id="TRITON_DUPLICATE_KERNEL_ARGUMENT",
                    title="Triton kernel argument passed twice",
                    message=(
                        f"{use.kernel_name}[grid](...) passes {use.args_count} positional arguments, "
                        f"so parameter '{keyword}' at position {index} is already bound before "
                        "it is supplied again as a keyword."
                    ),
                    location=Location(lineno=use.lineno, col=use.col),
                    hint="Make the kernel signature and launch agree; pass each kernel parameter once.",
                    tags={"triton", "host", "launch"},
                )
            )
    return issues
