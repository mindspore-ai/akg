"""
Post-edit semantic checks layered on top of guardrails and quick_check.

Contract: ``validate_patch_result(old, new, path, expected_delta_lines=None)``
returns ``None`` when the edit looks semantically plausible, or a short
error string when it doesn't. The caller (``execute_edit``) is expected
to roll back the file write when a non-None string is returned.

What this catches that ``_check_edit_guardrails`` and ``_qc_syntax``
don't:

- **Python syntax** is surfaced synchronously (``quick_check`` runs
  later, after the whole batch of edits — catching syntax here lets
  a single edit retry with an AST-aware fallback before eval burns).
- **Indentation sanity** — a patch that accidentally removes an
  ``if`` but keeps its body, or doubles a block's indent, is
  near-silent to ``py_compile`` (if indentation still parses) but
  obviously broken. We flag sudden jumps in max indent depth.
- **Line-count drift** — when the caller knows roughly how many
  lines the edit should add/remove (e.g. from the diff), a patch
  that explodes the file size is almost always a mis-paste.
- **Empty / no-op** — writing exactly the same content back wastes
  a turn; writing empty content tends to be a truncation bug.

Non-Python files skip the AST step. Tree-sitter-aware checks belong
in ``ast_editor`` and are called separately.
"""

from __future__ import annotations

import ast
import os
from typing import Optional

# Max jump in indentation depth (counted as "number of 4-space equivalents
# at deepest indented line") allowed between old and new. A jump of 2
# means: if old deepest was 3 levels, new may not exceed 5.
_MAX_INDENT_JUMP = 2

# Line-count drift check: if caller passes expected_delta_lines=+10 (10 lines
# added), actual delta must lie within [expected * 0.5 - 5, expected * 2 + 5].
_DELTA_LOWER_SCALE = 0.5
_DELTA_UPPER_SCALE = 2.0
_DELTA_TOLERANCE = 5


def _max_indent_depth(content: str) -> int:
    """Deepest indent level (in 4-space equivalents) across non-blank lines.

    Tabs count as 4 spaces. Lines with no indent contribute 0. Comments
    and blank lines are skipped.
    """
    best = 0
    for raw in content.splitlines():
        stripped = raw.lstrip(" \t")
        if not stripped or stripped.startswith("#"):
            continue
        lead = raw[: len(raw) - len(stripped)]
        # Normalize tabs to 4 spaces for counting.
        spaces = lead.replace("\t", "    ")
        depth = len(spaces) // 4
        if depth > best:
            best = depth
    return best


def _check_python_syntax(new_content: str, path: str) -> Optional[str]:
    """Return error message if AST parsing fails, else None."""
    try:
        ast.parse(new_content)
    except SyntaxError as e:
        # Keep the message tight — the agent sees it in tool_result.
        loc = f"line {e.lineno}" if e.lineno else "unknown location"
        return f"SyntaxError in {path} at {loc}: {e.msg}"
    return None


def _check_indent_jump(old_content: str, new_content: str,
                       path: str) -> Optional[str]:
    """Flag abrupt indentation explosions that parse but make no sense."""
    old_depth = _max_indent_depth(old_content)
    new_depth = _max_indent_depth(new_content)
    if new_depth - old_depth > _MAX_INDENT_JUMP:
        return (
            f"Indentation jumped from {old_depth} to {new_depth} levels in "
            f"{path} (>{_MAX_INDENT_JUMP}). Likely a copy/paste with extra "
            f"leading whitespace. Re-check the block's indent prefix."
        )
    return None


def _check_delta(old_content: str, new_content: str,
                 expected_delta_lines: Optional[int], path: str) -> Optional[str]:
    """Reject edits whose line-count change is far from expected."""
    if expected_delta_lines is None:
        return None
    old_lines = old_content.count("\n")
    new_lines = new_content.count("\n")
    actual = new_lines - old_lines
    if expected_delta_lines >= 0:
        lower = int(expected_delta_lines * _DELTA_LOWER_SCALE) - _DELTA_TOLERANCE
        upper = int(expected_delta_lines * _DELTA_UPPER_SCALE) + _DELTA_TOLERANCE
    else:
        # Symmetric bounds for negative deltas (line removal).
        lower = int(expected_delta_lines * _DELTA_UPPER_SCALE) - _DELTA_TOLERANCE
        upper = int(expected_delta_lines * _DELTA_LOWER_SCALE) + _DELTA_TOLERANCE
    if not (lower <= actual <= upper):
        return (
            f"Line-count drift in {path}: edit changed {actual:+d} lines, "
            f"expected near {expected_delta_lines:+d} (tolerance {lower:+d}..{upper:+d}). "
            f"Patch may have pasted or dropped extra content."
        )
    return None


def validate_patch_result(old_content: str, new_content: str,
                          path: str,
                          expected_delta_lines: Optional[int] = None) -> Optional[str]:
    """Run all post-write semantic checks. Returns ``None`` on success.

    Args:
        old_content:   File content before the edit.
        new_content:   File content after the edit.
        path:          Path (relative or absolute) used only for error text.
        expected_delta_lines: Optional signed line count the caller believes the
                        edit should produce (+10 for additions, -3 for
                        removals). When omitted the drift check is skipped.
    """
    if not new_content:
        return f"Patch produced empty content for {path}. Refusing write."
    if new_content == old_content:
        return f"Patch is a no-op for {path} (content unchanged)."

    is_python = path.endswith(".py")
    if is_python:
        err = _check_python_syntax(new_content, path)
        if err:
            return err

    err = _check_indent_jump(old_content, new_content, path)
    if err:
        return err

    err = _check_delta(old_content, new_content, expected_delta_lines, path)
    if err:
        return err

    return None
