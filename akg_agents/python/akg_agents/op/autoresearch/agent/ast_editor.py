"""
AST-aware editing — a last-resort retry path for ``execute_edit`` when
string matching fails. Supports two families:

  * **Python** via LibCST: replace a whole function / method / class
    body by name, preserving surrounding comments and formatting.
  * **C / C++ / CUDA / Triton-kernel** via tree-sitter: locate a
    top-level function definition by name and splice new text into
    its byte range.

Both families are optional. If the underlying library is not
installed (tree-sitter wheels fail on some Windows setups; LibCST
is a small pure-Python package but still a dependency we want to
treat as optional for minimal installs), ``ASTEditor`` reports
``available=False`` and the caller falls back to block-fuzzy mode.

Usage:

    editor = ASTEditor(path)           # selects language by extension
    if not editor.available:
        return None                    # caller picks another strategy
    new_text = editor.replace_symbol("kernel_foo", replacement_body)
    if new_text is None:
        return None                    # symbol not found

This module NEVER raises on missing deps — it treats them as a
soft "try another retry" signal. That keeps the edit dispatcher
linear and deterministic.
"""

from __future__ import annotations

import os
from typing import Optional

# ---------------------------------------------------------------------------
# Optional-dep feature flags (evaluated once at import time)
# ---------------------------------------------------------------------------

try:  # LibCST (Python AST that preserves formatting)
    import libcst as _cst  # noqa: F401
    _HAS_LIBCST = True
except Exception:
    _HAS_LIBCST = False

try:  # tree-sitter (C/C++/CUDA/Triton — CUDA is grammar-compatible with C++)
    import tree_sitter  # noqa: F401
    _HAS_TREESITTER = True
except Exception:
    _HAS_TREESITTER = False


_PY_EXTS = {".py"}
_CLIKE_EXTS = {".c", ".h", ".cc", ".hh", ".cpp", ".hpp", ".cu", ".cuh"}


# ---------------------------------------------------------------------------
# Python — LibCST-based symbol replacement
# ---------------------------------------------------------------------------


def _py_replace_symbol(source: str, symbol: str,
                       replacement: str) -> Optional[str]:
    """Replace a top-level def/class ``symbol`` body with ``replacement``.

    ``replacement`` is expected to be a complete Python definition whose
    name matches ``symbol``. Returns the new source on success, or
    ``None`` if the symbol is not found or the replacement doesn't
    parse. Preserves surrounding code verbatim (comments, blank lines,
    decorators outside the matched node).
    """
    if not _HAS_LIBCST:
        return None
    import libcst as cst

    try:
        tree = cst.parse_module(source)
    except Exception:
        return None
    try:
        repl_tree = cst.parse_module(replacement)
    except Exception:
        return None
    if not repl_tree.body:
        return None
    repl_node = repl_tree.body[0]

    class _ReplaceVisitor(cst.CSTTransformer):
        def __init__(self) -> None:
            self.hit = False

        def leave_FunctionDef(self, original_node, updated_node):
            if updated_node.name.value == symbol and not self.hit:
                self.hit = True
                return repl_node
            return updated_node

        def leave_ClassDef(self, original_node, updated_node):
            if updated_node.name.value == symbol and not self.hit:
                self.hit = True
                return repl_node
            return updated_node

    visitor = _ReplaceVisitor()
    new_tree = tree.visit(visitor)
    if not visitor.hit:
        return None
    return new_tree.code


# ---------------------------------------------------------------------------
# C-like (C/C++/CUDA) — tree-sitter-based symbol replacement
# ---------------------------------------------------------------------------

_TS_LANG_CACHE: dict = {}


def _get_ts_language(ext: str):
    """Load the right tree-sitter language module lazily. Returns None
    on any failure (missing wheel, unsupported platform, etc)."""
    if not _HAS_TREESITTER:
        return None
    if ext in _TS_LANG_CACHE:
        return _TS_LANG_CACHE[ext]
    try:
        if ext in {".c", ".h"}:
            import tree_sitter_c as _m  # type: ignore
            lang = tree_sitter.Language(_m.language())
        else:
            import tree_sitter_cpp as _m  # type: ignore
            lang = tree_sitter.Language(_m.language())
    except Exception:
        _TS_LANG_CACHE[ext] = None
        return None
    _TS_LANG_CACHE[ext] = lang
    return lang


def _clike_replace_symbol(source: str, symbol: str,
                          replacement: str, ext: str) -> Optional[str]:
    """Find a top-level function whose declarator names ``symbol`` and
    replace its text (byte range) with ``replacement``.

    The node we target is ``function_definition`` (C) or the equivalent
    node in the C++ grammar — CUDA reuses C++'s grammar for host/device
    functions. We walk the top-level children only; nested methods in
    classes are out of scope for this simple editor.
    """
    lang = _get_ts_language(ext)
    if lang is None:
        return None
    parser = tree_sitter.Parser(lang)
    src_bytes = source.encode("utf-8")
    tree = parser.parse(src_bytes)
    for child in tree.root_node.children:
        if child.type != "function_definition":
            continue
        declarator = child.child_by_field_name("declarator")
        if declarator is None:
            continue
        # Walk declarators (pointer_declarator / function_declarator / identifier).
        name_node = declarator
        while name_node is not None and name_node.type != "identifier":
            found = name_node.child_by_field_name("declarator")
            if found is None:
                # Some grammars put the identifier directly as a child.
                id_children = [c for c in name_node.children if c.type == "identifier"]
                name_node = id_children[0] if id_children else None
            else:
                name_node = found
        if name_node is None:
            continue
        name = src_bytes[name_node.start_byte:name_node.end_byte].decode("utf-8")
        if name == symbol:
            new_bytes = (src_bytes[:child.start_byte]
                         + replacement.encode("utf-8")
                         + src_bytes[child.end_byte:])
            return new_bytes.decode("utf-8")
    return None


# ---------------------------------------------------------------------------
# Public facade
# ---------------------------------------------------------------------------


class ASTEditor:
    """Facade that picks the right backend by file extension.

    ``available`` is ``False`` when no backend can handle the file, or
    when the chosen backend's dependencies are missing. Callers should
    check it before ``replace_symbol``.
    """

    def __init__(self, path: str) -> None:
        ext = os.path.splitext(path)[1].lower()
        self.path = path
        self.ext = ext
        if ext in _PY_EXTS:
            self.kind = "python"
            self.available = _HAS_LIBCST
        elif ext in _CLIKE_EXTS:
            self.kind = "clike"
            self.available = _HAS_TREESITTER and _get_ts_language(ext) is not None
        else:
            self.kind = "unsupported"
            self.available = False

    def replace_symbol(self, source: str, symbol: str,
                       replacement: str) -> Optional[str]:
        """Return the new source on hit, ``None`` on miss / unavailable."""
        if not self.available:
            return None
        if self.kind == "python":
            return _py_replace_symbol(source, symbol, replacement)
        if self.kind == "clike":
            return _clike_replace_symbol(source, symbol, replacement, self.ext)
        return None


__all__ = ["ASTEditor"]
