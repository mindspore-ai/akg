from __future__ import annotations

import ast
from typing import Dict, List, Set

from akg_agents.op.utils.code_checker.base import (
    CheckContext,
    Issue,
    Location,
    TritonDiagnosticChecker,
)

from .ast_utils import (
    extract_triton_info,
    is_jit_kernel,
)


CHECKER_NAME = "ascend_semantics"
CHECKER_ID = "triton_ascend_semantics"
RULE_ID = "TRITON_ASCEND_SEMANTICS"


class TritonAscendSemanticsChecker(TritonDiagnosticChecker):
    """Ascend-only Triton diagnostics for patterns known to fail lowering."""

    name = CHECKER_NAME
    checker_id = CHECKER_ID
    rule_id = RULE_ID

    def run(self, code: str, ctx: CheckContext) -> List[Issue]:
        if not _is_triton_ascend(ctx):
            return []

        info = extract_triton_info(code)
        aliases = info.aliases
        star_imports = info.star_imports
        issues: List[Issue] = []

        for node in ast.walk(info.tree):
            if isinstance(node, ast.FunctionDef) and is_jit_kernel(node, aliases, star_imports):
                issues.extend(_AscendKernelSemanticVisitor().check(node))

        return issues


class _AscendKernelSemanticVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.issues: List[Issue] = []

    def check(self, kernel: ast.FunctionDef) -> List[Issue]:
        self.visit(kernel)
        return self.issues

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if _is_mixed_scalar_slice_index(node.slice):
            self.issues.append(
                Issue(
                    severity="ERROR",
                    rule_id="TRITON_ASCEND_MIXED_SCALAR_SLICE_INDEX",
                    title="Ascend Triton mixed scalar/slice tensor indexing",
                    message=(
                        f"Ascend Triton lowering may reject mixed scalar and slice tensor indexing: "
                        f"{ast.unparse(node)}."
                    ),
                    location=Location(
                        lineno=getattr(node, "lineno", -1),
                        col=getattr(node, "col_offset", 0),
                        end_lineno=getattr(node, "end_lineno", -1),
                        end_col=getattr(node, "end_col_offset", 0),
                    ),
                    hint=(
                        "Avoid patterns like tensor[0, :] or tensor[:, i] inside @triton.jit; "
                        "keep vectors one-dimensional, preserve broadcast dimensions, or use tl.reshape."
                    ),
                    tags={"triton", "ascend", "kernel", "indexing"},
                )
            )
        self.generic_visit(node)


def _is_triton_ascend(ctx: CheckContext) -> bool:
    backend = (ctx.backend or "").lower()
    dsl = (ctx.dsl or "").lower()
    return backend == "ascend" or dsl == "triton_ascend"


def _is_mixed_scalar_slice_index(slice_node: ast.AST) -> bool:
    items = list(slice_node.elts) if isinstance(slice_node, ast.Tuple) else [slice_node]
    has_slice = any(isinstance(item, ast.Slice) for item in items)
    has_scalar = any(
        not isinstance(item, ast.Slice) and not _is_newaxis(item)
        for item in items
    )
    return has_slice and has_scalar


def _is_newaxis(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value is None
