from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


@dataclass
class TritonUse:
    canonical: str
    raw: str
    kind: str
    scope: str
    lineno: int
    col: int
    args_count: int = 0
    keywords: List[str] = field(default_factory=list)
    kernel_name: str = ""


@dataclass
class TritonAstInfo:
    tree: ast.AST
    aliases: Dict[str, str]
    star_imports: Set[str]
    kernels: Dict[str, List[str]]
    uses: List[TritonUse]


def collect_import_aliases(tree: ast.AST) -> tuple[Dict[str, str], Set[str]]:
    aliases: Dict[str, str] = {}
    star_imports: Set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                if item.name == "triton":
                    aliases[item.asname or "triton"] = "triton"
                elif item.name == "triton.language":
                    if item.asname:
                        aliases[item.asname] = "triton.language"
                    else:
                        aliases.setdefault("triton", "triton")
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module not in ("triton", "triton.language"):
                continue
            for item in node.names:
                if item.name == "*":
                    star_imports.add(node.module)
                    continue
                aliases[item.asname or item.name] = f"{node.module}.{item.name}"

    aliases.setdefault("triton", "triton")
    aliases.setdefault("tl", "triton.language")
    return aliases, star_imports


def attr_chain(node: ast.AST) -> Optional[List[str]]:
    parts: List[str] = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
        return list(reversed(parts))
    return None


def raw_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    chain = attr_chain(node)
    if chain:
        return ".".join(chain)
    return "<expr>"


def resolve_canonical(node: ast.AST, aliases: Dict[str, str], star_imports: Set[str]) -> Optional[str]:
    if isinstance(node, ast.Name):
        if node.id in aliases:
            return aliases[node.id]
        if "triton.language" in star_imports:
            return f"triton.language.{node.id}"
        if "triton" in star_imports:
            return f"triton.{node.id}"
        return None

    chain = attr_chain(node)
    if not chain:
        return None
    root = chain[0]
    if root not in aliases:
        return None

    base = aliases[root]
    if len(chain) == 1:
        return base
    return ".".join([base] + chain[1:])


def is_triton_api_name(canonical: Optional[str]) -> bool:
    return bool(canonical and (canonical == "triton" or canonical.startswith("triton.")))


def is_triton_jit_decorator(node: ast.AST, aliases: Dict[str, str], star_imports: Set[str]) -> bool:
    target = node.func if isinstance(node, ast.Call) else node
    canonical = resolve_canonical(target, aliases, star_imports)
    return canonical == "triton.jit"


def is_jit_kernel(fn: ast.FunctionDef, aliases: Dict[str, str], star_imports: Set[str]) -> bool:
    return any(is_triton_jit_decorator(dec, aliases, star_imports) for dec in fn.decorator_list)


def collect_kernel_params(fn: ast.FunctionDef) -> List[str]:
    params = []
    params.extend(arg.arg for arg in fn.args.posonlyargs)
    params.extend(arg.arg for arg in fn.args.args)
    params.extend(arg.arg for arg in fn.args.kwonlyargs)
    return params


def is_constexpr_annotation(node: Optional[ast.AST], aliases: Dict[str, str], star_imports: Set[str]) -> bool:
    if node is None:
        return False
    canonical = resolve_canonical(node, aliases, star_imports)
    return canonical == "triton.language.constexpr"


def extract_triton_info(code: str) -> TritonAstInfo:
    tree = ast.parse(code)
    aliases, star_imports = collect_import_aliases(tree)
    kernels: Dict[str, List[str]] = {}
    uses: List[TritonUse] = []

    class _Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.kernel_stack: List[str] = []

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            kernel = is_jit_kernel(node, aliases, star_imports)
            if kernel:
                kernels[node.name] = collect_kernel_params(node)

            for dec in node.decorator_list:
                self._record_decorator(dec, node.name if kernel else "")

            self.visit(node.args)
            if node.returns is not None:
                self.visit(node.returns)

            if kernel:
                self.kernel_stack.append(node.name)
            for stmt in node.body:
                self.visit(stmt)
            if kernel:
                self.kernel_stack.pop()

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            for dec in node.decorator_list:
                self._record_decorator(dec, "")
            self.visit(node.args)
            if node.returns is not None:
                self.visit(node.returns)
            for stmt in node.body:
                self.visit(stmt)

        def visit_Call(self, node: ast.Call) -> None:
            if isinstance(node.func, ast.Subscript):
                kernel_name = self._launch_kernel_name(node.func)
                if kernel_name:
                    uses.append(
                        TritonUse(
                            canonical=f"{kernel_name}.launch",
                            raw=f"{kernel_name}[grid]",
                            kind="launch",
                            scope=self._scope(),
                            lineno=getattr(node, "lineno", -1),
                            col=getattr(node, "col_offset", 0),
                            args_count=len(node.args),
                            keywords=[kw.arg or "**" for kw in node.keywords],
                            kernel_name=kernel_name,
                        )
                    )
                    self.visit(node.func.slice)
                    for arg in node.args:
                        self.visit(arg)
                    for kw in node.keywords:
                        self.visit(kw.value)
                    return

            canonical = resolve_canonical(node.func, aliases, star_imports)
            if is_triton_api_name(canonical):
                uses.append(
                    TritonUse(
                        canonical=canonical or "",
                        raw=raw_name(node.func),
                        kind="call",
                        scope=self._scope(),
                        lineno=getattr(node, "lineno", -1),
                        col=getattr(node, "col_offset", 0),
                        args_count=len(node.args),
                        keywords=[kw.arg or "**" for kw in node.keywords],
                        kernel_name=self.kernel_stack[-1] if self.kernel_stack else "",
                    )
                )

            for arg in node.args:
                self.visit(arg)
            for kw in node.keywords:
                self.visit(kw.value)

        def visit_Attribute(self, node: ast.Attribute) -> None:
            canonical = resolve_canonical(node, aliases, star_imports)
            if is_triton_api_name(canonical):
                uses.append(
                    TritonUse(
                        canonical=canonical or "",
                        raw=raw_name(node),
                        kind="attribute",
                        scope=self._scope(),
                        lineno=getattr(node, "lineno", -1),
                        col=getattr(node, "col_offset", 0),
                        kernel_name=self.kernel_stack[-1] if self.kernel_stack else "",
                    )
                )
            self.generic_visit(node)

        def _record_decorator(self, node: ast.AST, kernel_name: str) -> None:
            target = node.func if isinstance(node, ast.Call) else node
            canonical = resolve_canonical(target, aliases, star_imports)
            if not is_triton_api_name(canonical):
                return
            uses.append(
                TritonUse(
                    canonical=canonical or "",
                    raw=raw_name(target),
                    kind="decorator",
                    scope="kernel" if kernel_name else "host",
                    lineno=getattr(node, "lineno", -1),
                    col=getattr(node, "col_offset", 0),
                    args_count=len(node.args) if isinstance(node, ast.Call) else 0,
                    keywords=[kw.arg or "**" for kw in node.keywords] if isinstance(node, ast.Call) else [],
                    kernel_name=kernel_name,
                )
            )

        @staticmethod
        def _launch_kernel_name(func: ast.Subscript) -> str:
            value = func.value
            if isinstance(value, ast.Name):
                return value.id
            return ""

        def _scope(self) -> str:
            return "kernel" if self.kernel_stack else "host"

    _Visitor().visit(tree)
    return TritonAstInfo(
        tree=tree,
        aliases=aliases,
        star_imports=star_imports,
        kernels=kernels,
        uses=uses,
    )
