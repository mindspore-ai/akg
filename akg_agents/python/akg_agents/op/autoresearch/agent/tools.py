"""
Tool definitions and execution for the programmatic agent.

Organization:
  The sole LLM-facing file-mutation tool is ``edit`` — an atomic
  multi-edit dispatcher. A single call carries a list of edits (or
  a single-edit shorthand); they run sequentially against an
  in-memory buffer and the file is written exactly once at the end.
  If any edit fails at any retry stage, the file is left untouched.
  This is the Claude-Code / MultiEdit idiom applied to one file.

  Per-edit mode routing (inside ``execute_edit``):

      mode="exact"    → exact substring replacement (1 match required)
      mode="block"    → exact + whitespace-tolerant fuzzy fallback
      mode="unified"  → unified-diff patch (multi-hunk, fuzzy context)
      mode="rewrite"  → full file rewrite (must be the only edit)

  Retry ladder per edit (no extra API turns): widen anchor window →
  whitespace-normalized match → AST-aware by ``symbol`` (LibCST for
  Python, tree-sitter for C/C++/CUDA/Triton). Errors include top-3
  similar-line suggestions. Post-batch: guardrail check + semantic
  validator (AST / indent / summed line delta); failure rolls back
  the entire batch.

  Schemas:

      READ_FILE_TOOL       → execute_read_file
      EDIT_TOOL            → execute_edit

  State-mutation tools (update_plan / compact / finish) have their
  schemas here but no local handler — TurnExecutor owns the handlers
  because they mutate turn-level state (feedback / counters / buffer)
  and need phase-based permission checks against runtime state:

      UPDATE_PLAN_TOOL     → TurnExecutor._handle_update_plan
      COMPACT_TOOL         → TurnExecutor._handle_compact
      FINISH_TOOL          → TurnExecutor inline branch

  ``TOOLS`` — the Anthropic-native list sent to ``messages.create``
  — is assembled at the bottom of the file from these per-tool
  constants. ``build_tool_handlers(task_dir, config)`` binds the
  file-op handlers into a dispatch dict with the task context
  closed over; state-mutation tools are not included because
  TurnExecutor intercepts them before reaching that dispatch map.

Domain-specific tools (eval, quick_check) are internal — triggered
automatically by the loop after edits, not by the LLM.
"""

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..framework.config import TaskConfig
    from ..framework.runner import ExperimentRunner


# ---------------------------------------------------------------------------
# Structured tool result — replaces the "OK"/"ERROR: ..." string protocol
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ToolResult:
    """Structured result from a tool execution.

    Attributes:
        ok:      True if the operation succeeded.
        message: Human-readable text sent to the LLM as tool_result content.
        kind:    Tool category — "read", "patch", "write", "check", "eval".
    """
    ok: bool
    message: str
    kind: str = ""


# ---------------------------------------------------------------------------
# Tool schemas + handler implementations (co-located per tool)
# ---------------------------------------------------------------------------


# -- read_file --------------------------------------------------------------

READ_FILE_TOOL = {
    "name": "read_file",
    "description": (
        "Read file contents. "
        "mode='full' (default): entire file with line/char count header. "
        "mode='range': specific line range (requires target='start-end'). "
        "Files already shown in conversation do NOT need re-reading."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file, relative to the task directory.",
            },
            "mode": {
                "type": "string",
                "enum": ["full", "range"],
                "description": (
                    "'full': entire file with metadata (default). "
                    "'range': specific line range, requires target."
                ),
                "default": "full",
            },
            "target": {
                "type": "string",
                "description": "Line range for mode='range', e.g. '50-100'. Ignored for 'full'.",
            },
        },
        "required": ["path"],
    },
}


def execute_read_file(path: str, task_dir: str,
                      mode: str = "full", target: str | None = None) -> ToolResult:
    """Read a file's contents. Sandboxed to repo root.

    Modes:
        full  — entire file, no truncation, with line/char count header.
        range — specific line range (1-based), requires *target* like '50-100'.
    """
    if not os.path.isabs(path):
        resolved = os.path.join(task_dir, path)
    else:
        resolved = path
    resolved = os.path.normpath(os.path.abspath(resolved))

    # Sandbox: must be within git repo root
    try:
        repo_root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=task_dir, stderr=subprocess.DEVNULL, text=True
        ).strip()
        repo_root = os.path.normpath(os.path.abspath(repo_root))
    except Exception:
        # Walk up from task_dir looking for .git; fall back to task_dir itself
        _d = os.path.normpath(os.path.abspath(task_dir))
        repo_root = _d  # conservative default
        while True:
            if os.path.isdir(os.path.join(_d, ".git")):
                repo_root = _d
                break
            _parent = os.path.dirname(_d)
            if _parent == _d:
                break
            _d = _parent

    try:
        common = os.path.commonpath([resolved, repo_root])
    except ValueError:
        common = ""
    if os.path.normpath(common) != os.path.normpath(repo_root):
        return ToolResult(ok=False, message=f"ERROR: Path '{path}' escapes project root.", kind="read")

    if not os.path.exists(resolved):
        return ToolResult(ok=False, message=f"ERROR: File not found: {path}", kind="read")
    if not os.path.isfile(resolved):
        return ToolResult(ok=False, message=f"ERROR: Not a file: {resolved}", kind="read")

    try:
        with open(resolved, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except Exception as e:
        return ToolResult(ok=False, message=f"ERROR reading {resolved}: {e}", kind="read")

    source_lines = content.splitlines()
    total_lines = len(source_lines)
    total_chars = len(content)
    rel_path = os.path.relpath(resolved, task_dir)

    # --- range mode: return specific line range ---
    if mode == "range":
        if not target:
            return ToolResult(ok=False,
                              message="ERROR: mode='range' requires target (e.g. '50-100')", kind="read")
        try:
            parts = target.split("-")
            start = int(parts[0])
            end = int(parts[1]) if len(parts) > 1 else start
        except (ValueError, IndexError):
            return ToolResult(ok=False,
                              message=f"ERROR: invalid range '{target}', expected 'start-end'", kind="read")
        start = max(1, start)
        end = min(end, total_lines)
        if start > total_lines:
            return ToolResult(ok=False,
                              message=f"ERROR: start line {start} exceeds file length ({total_lines} lines)",
                              kind="read")
        extracted = "\n".join(source_lines[start - 1: end])
        header = f"[{rel_path}: lines {start}-{end} of {total_lines}, {total_chars} chars total]"
        return ToolResult(ok=True, message=f"{header}\n{extracted}", kind="read")

    # --- full mode (default): entire file, no truncation, with metadata header ---
    header = f"[{rel_path}: {total_lines} lines, {total_chars} chars]"
    return ToolResult(ok=True, message=f"{header}\n{content}", kind="read")


def _diff_lines(old: str, new: str) -> list[str]:
    """Return lines that are in *new* but not in *old* (added/changed)."""
    old_lines = set(old.splitlines())
    return [l for l in new.splitlines() if l not in old_lines]


def _check_edit_guardrails(new_content: str, config: "TaskConfig",
                           old_content: str | None = None) -> str | None:
    """Check edit content against guardrails.

    config.forbidden_patterns is a dict with three keys:
      - "content":  list of regex patterns matched against full new file
                    content; reject if ANY pattern matches.
      - "diff":     list of regex patterns matched against changed lines;
                    reject if ALL changed lines match any pattern (catches
                    comment-only / whitespace-only edits).
      - "diff_any": list of regex patterns matched against changed lines;
                    reject if ANY changed line matches any pattern (catches
                    edits that touch banned args / APIs).

    Returns error message or None.
    """
    if len(new_content) > config.max_patch_size:
        return (
            f"Edit too large ({len(new_content)} chars). "
            f"Maximum: {config.max_patch_size} chars."
        )
    fp = config.forbidden_patterns
    # Backward compat: list → treat as content patterns
    if isinstance(fp, list):
        fp = {"content": fp}
    for pattern in fp.get("content", []):
        if re.search(pattern, new_content):
            return f"Edit matches forbidden content pattern '{pattern}'."
    if old_content is not None:
        changed = _diff_lines(old_content, new_content)
        for pattern in fp.get("diff", []):
            if changed and all(re.match(pattern, l) for l in changed):
                return f"Edit rejected: all changed lines match diff pattern '{pattern}'."
        for pattern in fp.get("diff_any", []):
            for line in changed:
                if re.search(pattern, line):
                    return (
                        f"Edit rejected: new line matches banned pattern "
                        f"'{pattern}'. Omit this arg from new code "
                        f"(removing existing instances is allowed)."
                    )
    return None


def _validate_editable_path(
    path: str, task_dir: str, config: "TaskConfig", kind: str,
) -> tuple[str | None, ToolResult | None]:
    """Resolve path and check editable_files whitelist.

    Returns (abs_path, None) on success, or (None, error_result) on failure.
    """
    if os.path.isabs(path):
        target_abs = os.path.normpath(path)
    else:
        target_abs = os.path.normpath(os.path.join(task_dir, path))

    allowed_abs = {
        os.path.normpath(os.path.join(task_dir, f))
        for f in config.editable_files
    }
    if target_abs not in allowed_abs:
        label = "Write" if kind == "write" else "Patch"
        return None, ToolResult(
            ok=False, kind=kind,
            message=(
                f"ERROR: {label} rejected. '{path}' not in editable_files.\n"
                f"Allowed: {list(config.editable_files)}"
            ),
        )
    return target_abs, None


# -- Edit diagnostic helpers -----------------------------------------------
#
# Historical note: ``patch_file`` and ``write_file`` used to be separate
# LLM-facing tools. They were unified into the single ``edit`` tool in
# the multi-edit refactor — ``execute_edit`` (further down) owns the
# retry ladder and the atomic write. The diagnostic helpers below are
# still named with the legacy "patch" prefix because the exact error
# phrasing is part of the agent's contract (stale-file marking keys on
# the literal phrase "old_str not found").


def _suggest_similar_lines(content: str, old_str: str,
                           top_k: int = 3,
                           threshold: float = 0.6) -> list[tuple[float, int, str]]:
    """Find lines most similar to ``old_str``'s first non-blank line.

    Returns a list of ``(ratio, line_number, line_text)`` sorted by
    descending similarity. Empty list if nothing crosses *threshold*.
    Used by ``_diagnose_patch_mismatch`` to turn "old_str not found"
    errors into actionable suggestions instead of dead ends.
    """
    from difflib import SequenceMatcher
    target_lines = [ln for ln in old_str.splitlines() if ln.strip()]
    if not target_lines:
        return []
    target = target_lines[0]
    scored: list[tuple[float, int, str]] = []
    for ln, line in enumerate(content.splitlines(), 1):
        if not line.strip():
            continue
        ratio = SequenceMatcher(None, target, line).ratio()
        if ratio >= threshold:
            scored.append((ratio, ln, line))
    scored.sort(key=lambda x: -x[0])
    return scored[:top_k]


def _format_suggestions(suggestions: list[tuple[float, int, str]],
                        max_chars: int = 80) -> str:
    """Render similar-line suggestions as a short markdown block."""
    if not suggestions:
        return ""
    lines = ["Did you mean one of these?"]
    for ratio, ln, text in suggestions:
        snippet = text.rstrip()
        if len(snippet) > max_chars:
            snippet = snippet[:max_chars] + "…"
        lines.append(f"  L{ln} ({ratio:.0%}): {snippet}")
    return "\n".join(lines)


def _diagnose_patch_mismatch(content: str, old_str: str, path: str) -> str:
    """Produce an actionable error when exact match fails.

    Stale-marking upstream (turn.py) keys on the literal phrase
    "old_str not found". We emit that phrase only when we genuinely
    cannot locate the target — whitespace-only and ambiguity failures
    use distinct wording so the agent can retry without a re-read.
    """
    exact = content.count(old_str)
    lines = content.splitlines()

    if exact > 1:
        positions: list[int] = []
        start = 0
        while True:
            idx = content.find(old_str, start)
            if idx < 0:
                break
            positions.append(content.count("\n", 0, idx) + 1)
            start = idx + 1
        locs = ", ".join(f"L{n}" for n in positions[:8])
        return (
            f"ERROR: old_str appears {exact} times in {path} (at {locs}). "
            f"Make it more specific or pass anchor_line=<expected_line> to "
            f"disambiguate."
        )

    # exact == 0: try whitespace-normalized search
    def _norm(s: str) -> str:
        return re.sub(r"\s+", " ", s).strip()

    norm_old = _norm(old_str)
    if not norm_old:
        suggestions = _format_suggestions(_suggest_similar_lines(content, old_str))
        tail = ("\n" + suggestions) if suggestions else ""
        return (
            f"ERROR: old_str not found in {path}. Match exactly (including "
            f"whitespace).{tail}"
        )

    old_line_count = old_str.count("\n") + 1
    matches: list[tuple[int, str]] = []
    if old_line_count == 1:
        for i, line in enumerate(lines, 1):
            if _norm(line) == norm_old:
                matches.append((i, line))
                if len(matches) > 3:
                    break
    else:
        for i in range(len(lines) - old_line_count + 1):
            block = "\n".join(lines[i: i + old_line_count])
            if _norm(block) == norm_old:
                matches.append((i + 1, block))
                if len(matches) > 3:
                    break

    if len(matches) == 1:
        line_no, snippet = matches[0]
        return (
            f"ERROR: old_str whitespace/indent mismatch in {path} at line "
            f"{line_no}. Actual content:\n{snippet}\n"
            f"Copy the exact whitespace and retry — no re-read needed."
        )
    if len(matches) > 1:
        locs = ", ".join(f"L{n}" for n, _ in matches[:4])
        return (
            f"ERROR: old_str ambiguous in {path} — whitespace-normalized "
            f"search found {len(matches)} candidates at {locs}. Make it "
            f"more specific or use anchor_line."
        )

    suggestions = _format_suggestions(_suggest_similar_lines(content, old_str))
    tail = ("\n" + suggestions) if suggestions else ""
    return (
        f"ERROR: old_str not found in {path}. Match exactly (including "
        f"whitespace), or read_file to refresh your view.{tail}"
    )


# ---------------------------------------------------------------------------
# Unified edit tool — the sole LLM-facing file-mutation surface.
#
# ``edit`` takes a list of edits and applies them atomically to one file:
# edits run sequentially against an in-memory buffer, and the file is
# written exactly once at the end. If any edit fails at any retry stage,
# the file is left untouched (no partial writes). This is the Claude-
# Code / MultiEdit idiom — it lets the agent compose small dependent
# changes (remove import → add helper → rewrite call site) without
# burning tool-call budget or having to reason about the interleaving
# of partial-write states.
# ---------------------------------------------------------------------------


EDIT_TOOL = {
    "name": "edit",
    "description": (
        "Apply one or more edits to a single file atomically. Edits run "
        "sequentially on an in-memory buffer — edit #2 sees edit #1's "
        "result — and the file is written once at the end. If ANY edit "
        "fails (at any retry stage), the file is left untouched.\n"
        "\n"
        "Preferred form for multiple changes:\n"
        "  edit(path='f.py', plan_item_id='p1', description='refactor',\n"
        "       edits=[\n"
        "         {mode: 'exact', old_str: 'import a', new_str: 'import b'},\n"
        "         {mode: 'exact', old_str: 'a.foo()', new_str: 'b.foo()'},\n"
        "       ])\n"
        "\n"
        "Shorthand for a single change (top-level mode/old_str/new_str/…):\n"
        "  edit(path='f.py', plan_item_id='p1', description='...',\n"
        "       mode='exact', old_str=..., new_str=...)\n"
        "\n"
        "Per-edit modes:\n"
        "  mode='exact'   — replace an exact substring (old_str must "
        "                   appear exactly once in the current buffer, "
        "                   or disambiguate with anchor_line). Default.\n"
        "  mode='block'   — like exact, but retries with whitespace-"
        "                   tolerant matching if the first pass misses.\n"
        "  mode='unified' — apply a unified diff (``diff`` field); "
        "                   multi-hunk, fuzz=±2 context lines.\n"
        "  mode='rewrite' — replace the ENTIRE buffer with ``new_str``. "
        "                   Must be the only edit in the batch.\n"
        "\n"
        "Retry ladder per edit (before surfacing an error): widen anchor "
        "window → whitespace-normalized match (block only) → AST-aware "
        "replacement by ``symbol`` (LibCST for Python, tree-sitter for "
        "C/C++/CUDA/Triton). Errors include top-3 similar-line "
        "suggestions by edit distance.\n"
        "\n"
        "Post-write batch validator: AST parse + indent-depth bound + "
        "summed line-delta drift. On any check failure the whole batch "
        "rolls back."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Relative path from task_dir. Must be in editable_files.",
            },
            "edits": {
                "type": "array",
                "description": (
                    "Ordered list of edits applied atomically. When "
                    "provided, top-level edit fields are ignored. Each "
                    "entry accepts the same per-edit fields as the "
                    "top-level shorthand."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "mode": {
                            "type": "string",
                            "enum": ["exact", "block", "unified", "rewrite"],
                            "default": "exact",
                        },
                        "old_str": {"type": "string"},
                        "new_str": {"type": "string"},
                        "diff": {"type": "string"},
                        "anchor_line": {"type": "integer"},
                        "symbol": {"type": "string"},
                        "expected_delta_lines": {"type": "integer"},
                    },
                },
            },
            "mode": {
                "type": "string",
                "enum": ["exact", "block", "unified", "rewrite"],
                "description": (
                    "Single-edit shorthand. Used when `edits` is absent. "
                    "Defaults to 'exact'."
                ),
                "default": "exact",
            },
            "old_str": {
                "type": "string",
                "description": (
                    "Single-edit shorthand. Substring to replace for "
                    "mode='exact'/'block'."
                ),
            },
            "new_str": {
                "type": "string",
                "description": (
                    "Single-edit shorthand. Replacement string for "
                    "exact/block; full content for 'rewrite'."
                ),
            },
            "diff": {
                "type": "string",
                "description": (
                    "Single-edit shorthand. Unified diff text for "
                    "mode='unified'."
                ),
            },
            "anchor_line": {
                "type": "integer",
                "description": (
                    "Single-edit shorthand. 1-based line number pinning "
                    "the match to ±5 lines."
                ),
            },
            "symbol": {
                "type": "string",
                "description": (
                    "Single-edit shorthand. Function/class name enabling "
                    "AST-aware retry."
                ),
            },
            "expected_delta_lines": {
                "type": "integer",
                "description": (
                    "Single-edit shorthand. Signed line-count delta "
                    "expected."
                ),
            },
            "description": {
                "type": "string",
                "description": (
                    "Specific change description: name the file/target/"
                    "technique. Kept in the round log and written into "
                    "plan history for traceability."
                ),
            },
            "plan_item_id": {
                "type": "string",
                "description": "ID of the active plan item this edit belongs to (e.g. 'p1').",
            },
        },
        "required": ["path", "description", "plan_item_id"],
    },
}


def _apply_unified_diff(content: str, diff_text: str,
                        fuzz: int = 2) -> tuple[Optional[str], Optional[str]]:
    """Apply a unified diff to ``content`` with ±fuzz context tolerance.

    Returns ``(new_content, None)`` on success, or ``(None, error)`` on
    failure. This is a deliberately minimal implementation (no git
    metadata, no rename detection) — just @@ hunks against a single
    file's text.
    """
    lines = content.splitlines(keepends=True)
    result = list(lines)
    hunk_header_re = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
    # Parse hunks from diff.
    hunks: list[tuple[int, list[str]]] = []
    current: list[str] | None = None
    current_start = 0
    for raw in diff_text.splitlines():
        m = hunk_header_re.match(raw)
        if m:
            if current is not None:
                hunks.append((current_start, current))
            current_start = int(m.group(1))
            current = []
            continue
        if current is None:
            # Skip ---/+++/index/diff headers before first hunk.
            continue
        if raw.startswith(("---", "+++")):
            continue
        current.append(raw)
    if current is not None:
        hunks.append((current_start, current))
    if not hunks:
        return None, "unified diff contained no @@ hunks"

    # Apply hunks bottom-up to keep line numbers stable.
    for start, hunk_lines in sorted(hunks, key=lambda h: -h[0]):
        # Build "before" and "after" line lists from the hunk body.
        before: list[str] = []
        after: list[str] = []
        for hl in hunk_lines:
            if not hl:
                before.append("")
                after.append("")
                continue
            tag, body = hl[0], hl[1:]
            if tag == " ":
                before.append(body)
                after.append(body)
            elif tag == "-":
                before.append(body)
            elif tag == "+":
                after.append(body)
            else:
                # Unknown marker — treat as context.
                before.append(hl)
                after.append(hl)
        # Locate the before-block in the file around the declared start.
        anchor = max(0, start - 1)
        window_lo = max(0, anchor - fuzz)
        window_hi = min(len(result), anchor + fuzz + 1)
        hit = -1
        for cand in range(window_lo, window_hi):
            slice_ = [l.rstrip("\n") for l in result[cand:cand + len(before)]]
            if slice_ == before:
                hit = cand
                break
        if hit < 0:
            # Widen the search to whole-file as a last resort.
            for cand in range(0, len(result) - len(before) + 1):
                slice_ = [l.rstrip("\n") for l in result[cand:cand + len(before)]]
                if slice_ == before:
                    hit = cand
                    break
        if hit < 0:
            return None, f"hunk at @@{start} did not match (context drift beyond fuzz={fuzz})"
        # Splice. Preserve trailing newlines by re-adding "\n" to after-lines.
        new_block = [(l + "\n") for l in after]
        result[hit:hit + len(before)] = new_block
    return "".join(result), None


# ---------------------------------------------------------------------------
# Pure-string edit primitives — no file I/O, no guardrail, no validator.
# Each returns (new_content, error_msg). These are the building blocks
# invoked by the per-edit retry ladder inside execute_edit.
# ---------------------------------------------------------------------------


def _s_exact_replace(content: str, old_str: str,
                     anchor_line: Optional[int],
                     path: str) -> tuple[Optional[str], Optional[str]]:
    """Exactly-one-match substring locate; returns (spliced_placeholder,
    error). The caller holds new_str — we return the position-aware
    slice offsets via a small tuple instead, keeping the helper minimal."""
    # Callers use _s_exact_splice below; this helper is kept separate so
    # the splice offsets can be returned without copying content.
    return None, "use _s_exact_splice"


def _s_exact_splice(content: str, old_str: str, new_str: str,
                    anchor_line: Optional[int],
                    path: str) -> tuple[Optional[str], Optional[str]]:
    """Attempt 1 of the retry ladder: straight exact match.

    Returns ``(new_content, None)`` on success, ``(None, error_msg)``
    otherwise. Error message uses the same phrasing as
    ``_diagnose_patch_mismatch`` so upstream stale-marking keys off the
    right substring.
    """
    if anchor_line is None:
        count = content.count(old_str)
        if count != 1:
            return None, _diagnose_patch_mismatch(content, old_str, path)
        return content.replace(old_str, new_str, 1), None
    candidates: list[tuple[int, int]] = []
    start = 0
    while True:
        idx = content.find(old_str, start)
        if idx < 0:
            break
        ln = content.count("\n", 0, idx) + 1
        candidates.append((ln, idx))
        start = idx + 1
    if not candidates:
        return None, _diagnose_patch_mismatch(content, old_str, path)
    within = [(ln, idx) for ln, idx in candidates
              if abs(ln - anchor_line) <= 5]
    if len(within) == 1:
        _, idx = within[0]
        return content[:idx] + new_str + content[idx + len(old_str):], None
    if not within:
        nearest = min(candidates, key=lambda c: abs(c[0] - anchor_line))
        return None, (
            f"ERROR: old_str found in {path} but not within ±5 lines of "
            f"anchor_line={anchor_line} (nearest match at L{nearest[0]}). "
            f"Adjust anchor_line or drop it."
        )
    locs = ", ".join(f"L{ln}" for ln, _ in within)
    return None, (
        f"ERROR: old_str ambiguous within ±5 lines of anchor_line="
        f"{anchor_line} in {path} (matches at {locs}). Tighten "
        f"anchor_line or make old_str more specific."
    )


def _s_widened_anchor(content: str, old_str: str, new_str: str,
                      anchor_line: int, path: str,
                      window: int = 15) -> tuple[Optional[str], Optional[str]]:
    """Attempt 2: widen the anchor search window to ±``window`` lines."""
    candidates: list[tuple[int, int]] = []
    start = 0
    while True:
        idx = content.find(old_str, start)
        if idx < 0:
            break
        ln = content.count("\n", 0, idx) + 1
        candidates.append((ln, idx))
        start = idx + 1
    within = [(ln, idx) for ln, idx in candidates
              if abs(ln - anchor_line) <= window]
    if len(within) != 1:
        return None, (f"ERROR: widened anchor (±{window}) still "
                      f"ambiguous or missed in {path}")
    _, idx = within[0]
    return content[:idx] + new_str + content[idx + len(old_str):], None


def _s_whitespace_fuzz(content: str, old_str: str, new_str: str,
                       path: str) -> tuple[Optional[str], Optional[str]]:
    """Attempt 3 (block mode only): whitespace-normalized match that
    preserves the file's indentation on splice."""
    def _norm(s: str) -> str:
        return re.sub(r"\s+", " ", s).strip()
    norm_old = _norm(old_str)
    if not norm_old:
        return None, "ERROR: empty old_str after normalization"
    old_line_count = old_str.count("\n") + 1
    lines = content.splitlines(keepends=True)
    char_idx_at_line: list[int] = []
    running = 0
    for l in lines:
        char_idx_at_line.append(running)
        running += len(l)
    matches: list[tuple[int, int]] = []
    if old_line_count == 1:
        for i, l in enumerate(lines):
            if _norm(l) == norm_old:
                matches.append((i, char_idx_at_line[i]))
    else:
        for i in range(len(lines) - old_line_count + 1):
            block = "".join(lines[i:i + old_line_count])
            if _norm(block) == norm_old:
                matches.append((i, char_idx_at_line[i]))
    if len(matches) != 1:
        return None, (f"ERROR: whitespace-fuzzy match count="
                      f"{len(matches)} in {path}")
    start_line, char_idx = matches[0]
    end_line = start_line + old_line_count
    end_char = (char_idx_at_line[end_line] if end_line < len(lines)
                else len(content))
    leading = re.match(r"^\s*", lines[start_line]).group(0)
    new_lines = new_str.splitlines(keepends=True) if new_str else []
    if new_lines:
        new_first_indent = re.match(r"^\s*", new_lines[0]).group(0)
        if new_first_indent != leading:
            common = new_first_indent
            new_str = "".join(
                (leading + l[len(common):]) if l.startswith(common) else l
                for l in new_lines
            )
    original_span = content[char_idx:end_char]
    if original_span.endswith("\n") and not new_str.endswith("\n"):
        new_str = new_str + "\n"
    return content[:char_idx] + new_str + content[end_char:], None


def _s_ast_replace(content: str, symbol: str, new_str: str,
                   path: str) -> tuple[Optional[str], Optional[str]]:
    """Attempt 4: AST-aware symbol body replacement. Returns
    ``(None, None)`` when no AST backend is available so the caller
    falls back to the prior error, or ``(None, msg)`` when the symbol
    is genuinely absent."""
    from .ast_editor import ASTEditor
    editor = ASTEditor(path)
    if not editor.available:
        return None, None
    result = editor.replace_symbol(content, symbol, new_str)
    if result is None:
        return None, f"ERROR: AST backend could not locate symbol '{symbol}' in {path}"
    return result, None


def _run_edit_step(content: str, ed: dict,
                   path: str) -> tuple[Optional[str], str, int]:
    """Apply one edit spec to ``content`` with the full retry ladder.

    Returns ``(new_content, message, retries)`` — new_content is None
    on total failure, otherwise the transformed string. ``message`` is
    either a success note or the final error. ``retries`` counts the
    number of fallback stages that executed (0 = first try succeeded).
    """
    mode = (ed.get("mode") or "exact").lower()
    old_str = ed.get("old_str", "") or ""
    new_str = ed.get("new_str", "") or ""
    anchor_line = ed.get("anchor_line")
    symbol = ed.get("symbol")

    if mode == "rewrite":
        return new_str, "rewrite", 0

    if mode == "unified":
        diff_text = ed.get("diff", "") or ""
        result, err = _apply_unified_diff(content, diff_text)
        if result is None:
            return None, f"ERROR: unified diff failed: {err}", 0
        return result, "unified", 0

    if mode not in ("exact", "block"):
        return None, f"ERROR: unknown edit mode '{mode}'", 0

    retries = 0
    # Attempt 1: exact (with optional anchor_line).
    new_content, err = _s_exact_splice(content, old_str, new_str,
                                       anchor_line, path)
    if new_content is not None:
        return new_content, f"{mode}", retries
    last_err = err

    # Attempt 2: widen anchor if one was given and we missed it.
    if anchor_line is not None and "not within ±5" in last_err:
        new_content, err = _s_widened_anchor(content, old_str, new_str,
                                             anchor_line, path)
        retries += 1
        if new_content is not None:
            return new_content, f"{mode}+widened", retries
        last_err = err

    # Attempt 3: whitespace-fuzzy — block mode only.
    if mode == "block" and ("whitespace/indent mismatch" in last_err
                            or "not found" in last_err):
        new_content, err = _s_whitespace_fuzz(content, old_str, new_str,
                                              path)
        retries += 1
        if new_content is not None:
            return new_content, f"{mode}+ws-fuzzy", retries
        last_err = err

    # Attempt 4: AST by symbol.
    if symbol:
        new_content, err = _s_ast_replace(content, symbol, new_str, path)
        if new_content is not None:
            retries += 1
            return new_content, f"{mode}+ast", retries
        if err is not None:
            retries += 1
            last_err = err

    prefix = f"[retries={retries}] " if retries else ""
    return None, prefix + last_err, retries


def execute_edit(path: str, task_dir: str,
                 config: "TaskConfig", *,
                 edits: Optional[list] = None,
                 mode: Optional[str] = None,
                 old_str: str = "", new_str: str = "",
                 diff: str = "", anchor_line: Optional[int] = None,
                 symbol: Optional[str] = None,
                 expected_delta_lines: Optional[int] = None,
                 **_ignored) -> ToolResult:
    """Atomic multi-edit dispatcher.

    Loads the file once, runs every edit spec through ``_run_edit_step``
    against the evolving in-memory buffer, then writes once at the end.
    Any step failure discards the whole batch and returns the offending
    error with the 1-based edit index. Post-batch guardrails and
    semantic validator run against the combined delta.

    Two input shapes:
      * ``edits=[{mode, old_str, new_str, ...}, ...]`` — preferred.
      * Top-level ``mode``/``old_str``/... — single-edit shorthand.
    """
    # Normalize input into a list of edit specs.
    if edits is None:
        edits = [{
            "mode": mode or "exact",
            "old_str": old_str, "new_str": new_str, "diff": diff,
            "anchor_line": anchor_line, "symbol": symbol,
            "expected_delta_lines": expected_delta_lines,
        }]
    if not edits or not isinstance(edits, list):
        return ToolResult(ok=False, kind="patch",
                          message="ERROR: no edits provided")
    # rewrite must be the only edit in a batch — otherwise earlier
    # edits would be silently discarded.
    modes = [(e.get("mode") or "exact").lower() for e in edits]
    if "rewrite" in modes and len(edits) > 1:
        return ToolResult(ok=False, kind="patch",
                          message=("ERROR: mode='rewrite' must be the "
                                   "only edit in the batch"))

    target_abs, err = _validate_editable_path(path, task_dir, config, "patch")
    if err:
        return err

    try:
        with open(target_abs, "r", encoding="utf-8") as f:
            original = f.read()
    except FileNotFoundError:
        # Allow rewrite to create a new file; other modes need the
        # file to exist.
        if modes == ["rewrite"]:
            original = ""
        else:
            return ToolResult(ok=False, kind="patch",
                              message=f"ERROR: file not found: {path}")
    except Exception as e:
        return ToolResult(ok=False, kind="patch",
                          message=f"ERROR reading {path}: {e}")

    # Apply each edit sequentially to the in-memory buffer.
    content = original
    total_retries = 0
    for i, ed in enumerate(edits):
        new_content, msg, retries = _run_edit_step(content, ed, path)
        total_retries += retries
        if new_content is None:
            loc = (f" (edit #{i + 1}/{len(edits)})"
                   if len(edits) > 1 else "")
            return ToolResult(ok=False, kind="patch",
                              message=f"{msg}{loc}")
        content = new_content

    # Guardrails + post-batch semantic validator on the final content.
    from .patch_validator import validate_patch_result
    guardrail_error = _check_edit_guardrails(content, config,
                                             old_content=original)
    if guardrail_error:
        return ToolResult(ok=False, kind="patch",
                          message=f"ERROR: {guardrail_error}")

    # Summed delta across all edits with an explicit expectation.
    agg_delta: Optional[int] = None
    for ed in edits:
        d = ed.get("expected_delta_lines")
        if d is None:
            continue
        agg_delta = (agg_delta or 0) + int(d)
    verr = validate_patch_result(original, content, path, agg_delta)
    if verr:
        return ToolResult(ok=False, kind="patch",
                          message=f"ERROR: post-batch validator: {verr}")

    # All clear — write once. For nested new-file creation via
    # mode='rewrite' (the only mode that accepts a missing target),
    # ensure the parent directory exists — the legacy execute_write_file
    # used to do this and dropping the behavior silently broke any
    # editable_files entry with a subdirectory.
    try:
        parent = os.path.dirname(target_abs)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(target_abs, "w", encoding="utf-8") as f:
            f.write(content)
    except Exception as e:
        return ToolResult(ok=False, kind="patch",
                          message=f"ERROR writing {path}: {e}")

    rel = os.path.relpath(target_abs, task_dir)
    count_tag = f" ({len(edits)} edits)" if len(edits) > 1 else ""
    retries_tag = f" [retries={total_retries}]" if total_retries else ""
    return ToolResult(ok=True, kind="patch",
                      message=f"OK: patched {rel}{count_tag}{retries_tag}")


def _qc_syntax(fpath: str, fname: str) -> Optional[str]:
    """py_compile the file. Returns error string or None."""
    import py_compile
    try:
        py_compile.compile(fpath, doraise=True)
        return None
    except py_compile.PyCompileError as e:
        return f"SyntaxError in {fname}: {e}"


def _qc_dsl_compliance(fpath: str, fname: str,
                       config: "TaskConfig") -> list[str]:
    """Run CodeChecker's DSL-compliance pass. Empty list if unavailable."""
    if not config.dsl:
        return []
    try:
        from akg_agents.op.utils.code_checker import CodeChecker
    except ImportError:
        return []  # standalone mode — no CodeChecker available
    checker = CodeChecker(backend=config.backend or "", dsl=config.dsl,
                          arch=config.arch or "")
    with open(fpath, "r", encoding="utf-8") as fh:
        code = fh.read()
    return [
        f"DSL compliance in {fname}: {err['detail']}"
        for err in checker._check_dsl_compliance(code)
    ]


def _qc_import(fpath: str, fname: str, task_dir: str,
               config: "TaskConfig",
               device_id: Optional[int]) -> list[str]:
    """Import the module in a subprocess with a timeout.

    Empty list when import_timeout <= 0 (AKG mode) — the caller
    short-circuits before this is reached, but we stay defensive.
    """
    if config.import_timeout <= 0:
        return []
    from ..framework.device import get_device_env
    import_env = get_device_env(device_id)
    try:
        result = subprocess.run(
            [sys.executable, "-c",
             f"import importlib.util; "
             f"spec = importlib.util.spec_from_file_location('_check', r'{fpath}'); "
             f"mod = importlib.util.module_from_spec(spec); "
             f"spec.loader.exec_module(mod)"],
            capture_output=True, text=True, timeout=config.import_timeout,
            cwd=task_dir, env=import_env,
        )
    except subprocess.TimeoutExpired:
        return [f"Import check timed out for {fname} (>{config.import_timeout}s)"]
    except Exception as e:
        return [f"Check failed for {fname}: {e}"]
    if result.returncode != 0:
        last_lines = result.stderr.strip().split("\n")[-3:]
        return [f"ImportError in {fname}: {chr(10).join(last_lines)}"]
    return []


def _qc_resolve_smoke_cmd(task_dir: str,
                          config: "TaskConfig") -> Optional[list[str]]:
    """Decide the smoke-test command. Returns None if no smoke test applies."""
    if config.smoke_test_script:
        smoke_path = os.path.join(task_dir, config.smoke_test_script)
        if os.path.exists(smoke_path):
            return [sys.executable, smoke_path]
        return None
    if not (config.dsl and config.framework and config.backend):
        return None
    try:
        from ..framework.eval_generator import generate_eval_script_file
    except ImportError:
        return None
    eval_path = generate_eval_script_file(
        dsl=config.dsl, framework=config.framework, backend=config.backend,
        output_dir=os.path.join(task_dir, ".eval_cache"),
    )
    return [sys.executable, eval_path, "--task-dir", task_dir, "--smoke"]


def _qc_smoke(task_dir: str, config: "TaskConfig",
              device_id: Optional[int]) -> list[str]:
    """Run the smoke test. Empty list when no smoke command applies."""
    smoke_cmd = _qc_resolve_smoke_cmd(task_dir, config)
    if smoke_cmd is None:
        return []
    from ..framework.device import get_device_env
    try:
        result = subprocess.run(
            smoke_cmd,
            capture_output=True, text=True,
            timeout=config.smoke_test_timeout,
            cwd=task_dir, env=get_device_env(device_id),
        )
    except subprocess.TimeoutExpired:
        return [f"Smoke test timed out after {config.smoke_test_timeout}s"]
    except Exception as e:
        return [f"Smoke test error: {e}"]

    if result.returncode != 0:
        limit = config.agent.smoke_output_limit
        stderr_tail = result.stderr[-limit:] if result.stderr else ""
        stdout_tail = result.stdout[-limit:] if result.stdout else ""
        return [
            f"Smoke test failed (exit {result.returncode}):\n"
            f"{stderr_tail}\n{stdout_tail}".strip()
        ]
    # Return code OK — scan stdout for the last JSON line and check correctness
    for line in reversed(result.stdout.strip().split("\n")):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not data.get("correctness", False):
            return [f"Smoke test correctness=false: {line}"]
        break
    return []


def execute_quick_check(task_dir: str, config: "TaskConfig",
                        device_id: Optional[int] = None) -> ToolResult:
    """
    Pre-flight check: syntax + DSL compliance + optional import + optional smoke.

    Per-file stages (syntax, DSL compliance, import) run for each
    editable .py file. If the per-file loop produces errors, the smoke
    test is skipped — those errors would only cascade.
    """
    per_file_errors: list[str] = []
    for fname in config.editable_files:
        fpath = os.path.join(task_dir, fname)
        if not fpath.endswith(".py") or not os.path.exists(fpath):
            continue
        syntax_err = _qc_syntax(fpath, fname)
        if syntax_err is not None:
            per_file_errors.append(syntax_err)
            # Syntax broken — DSL / import would also fail; try next file.
            continue
        per_file_errors.extend(_qc_dsl_compliance(fpath, fname, config))
        if config.import_timeout > 0:
            per_file_errors.extend(
                _qc_import(fpath, fname, task_dir, config, device_id)
            )

    if per_file_errors:
        return ToolResult(
            ok=False,
            message="quick_check failed:\n" + "\n".join(per_file_errors),
            kind="check",
        )

    if config.import_timeout <= 0:
        # Syntax-only mode (AKG): don't run the smoke test either.
        return ToolResult(
            ok=True, message="quick_check passed (syntax only)", kind="check",
        )

    smoke_errors = _qc_smoke(task_dir, config, device_id)
    if smoke_errors:
        return ToolResult(
            ok=False,
            message="quick_check failed:\n" + "\n".join(smoke_errors),
            kind="check",
        )
    return ToolResult(ok=True, message="OK", kind="check")


async def execute_run_eval(description: str, runner: "ExperimentRunner",
                           raw_output_tail: int = 2_048) -> str:
    """
    Run evaluation via runner.run_one_round().
    INTERNAL ONLY — not exposed to the LLM.
    Returns JSON string with round record summary.
    """
    record = await runner.run_one_round(description)
    r = record.result

    # Three-way status:
    #   KEEP    — accepted (correct + improved + constraints met)
    #   FAIL    — code is broken (correctness / constraint / infrastructure error)
    #   DISCARD — code is correct but not an improvement
    if record.accepted:
        status = "KEEP"
    elif not r.correctness or record.constraint_violations:
        status = "FAIL"
    else:
        status = "DISCARD"

    # Unified fail_reason for FAIL status (diagnostic detail for the agent)
    fail_reason = None
    if status == "FAIL":
        if r.error:
            fail_reason = r.error
        elif not r.correctness:
            fail_reason = "correctness mismatch"
        elif record.constraint_violations:
            fail_reason = "constraint: " + "; ".join(record.constraint_violations)

    result = {
        "round": record.round_num,
        "description": record.description,
        "accepted": record.accepted,
        "status": status,
        "correctness": r.correctness,
        "metrics": r.metrics,
        "commit": record.commit_hash,
        "fail_reason": fail_reason,
        "constraint_violations": record.constraint_violations,
        "duration_sec": round(record.duration_sec, 1),
    }
    raw = record.result.raw_output or ""
    if raw:
        result["raw_output_tail"] = raw[-raw_output_tail:]
    return json.dumps(result, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# State-mutation tool schemas — handlers live in TurnExecutor
#
# These three tools mutate turn-level state (feedback / counters / buffer)
# and are subject to phase-based permission checks against runtime state
# (e.g. finish requires eval_calls_made >= max_rounds // 2, phase ==
# replanning, and no edits in the same turn). That logic lives in
# TurnExecutor._dispatch_tools / _handle_update_plan / _handle_compact
# rather than here, so these schemas have no matching execute_* function.
# ---------------------------------------------------------------------------


UPDATE_PLAN_TOOL = {
    "name": "update_plan",
    "description": (
        "Submit a new optimization plan as `items=[...]`. Each item MUST "
        "include `text` (the change) and `rationale` (one sentence: "
        "name the bottleneck AND the expected effect). Generic phrases "
        "like 'optimize'/'improve performance' are rejected and the whole "
        "plan is discarded. Optional `keywords` request skill matching. "
        "Callable only in `no_plan` or `replanning` phase."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "description": (
                    "Structured plan items. Every item requires `text` and "
                    "`rationale`."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "Imperative description of the code change.",
                        },
                        "rationale": {
                            "type": "string",
                            "description": (
                                "One sentence: name the bottleneck and the "
                                "expected effect (e.g. 'BLOCK_K=32 forces 8 "
                                "K-iterations; doubling halves passes'). "
                                "Generic phrases are rejected."
                            ),
                        },
                        "keywords": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 1,
                            "maxItems": 5,
                            "description": (
                                "Optional technical tokens for skill matching "
                                "(e.g. ['matmul', 'tiling', 'make_block_ptr'])."
                            ),
                        },
                    },
                    "required": ["text", "rationale"],
                },
            },
        },
        "required": ["items"],
    },
}


SEARCH_SKILLS_TOOL = {
    "name": "search_skills",
    "description": (
        "Extend the pre-selected skill pool. Call when the current pool "
        "doesn't cover the direction you want to explore, or when the "
        "pool is exhausted. Provide a short natural-language "
        "hint (e.g. 'fused softmax for FP16', 'memory coalescing for "
        "matmul'). The framework combines hint + task_desc to generate "
        "fresh keywords, re-ranks the catalog, and appends new "
        "non-duplicate candidates to the pool. New skills become "
        "available for keyword-driven binding on your NEXT update_plan "
        "call - this tool does not modify the currently-active plan. "
        "Safe to call in any phase."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "hint": {
                "type": "string",
                "description": (
                    "Natural-language description of the optimization "
                    "direction. Combined with task_desc to drive keyword "
                    "generation."
                ),
            },
        },
        "required": ["hint"],
    },
}


ACKNOWLEDGE_SKILL_TOOL = {
    "name": "acknowledge_skill",
    "description": (
        "REQUIRED before ``edit`` on any plan item that has a "
        "backing_skill. The framework has just injected the SKILL.md "
        "content into your context — confirm you read it by producing "
        "TWO distinct analyses:\n"
        "\n"
        "  1. ``valuable_aspects`` — what is valuable in this skill as a "
        "     general pattern (algorithm, access pattern, known pitfalls). "
        "     Skill-level, not kernel-level.\n"
        "  2. ``kernel_application`` — how those valuable parts apply to "
        "     THIS kernel's current code, and what concrete change you "
        "     will make. Prefer STRUCTURAL edits (algorithm change, "
        "     memory-hierarchy rewrite, kernel fusion / split, access-"
        "     pattern rework) FIRST. Only fall back to parameter tuning "
        "     (BLOCK_SIZE / autotune configs / num_stages) when you have "
        "     a specific hypothesis linking the parameter to the skill's "
        "     structural claim. 'I will sweep X' without a structural "
        "     hypothesis is a weak ack.\n"
        "\n"
        "applicability:\n"
        "  - 'apply'  — you will use the skill (fully or in part) to "
        "    make the edit on this item.\n"
        "  - 'unbind' — the skill does not fit THIS item. The binding "
        "    is released for this item and the item becomes free "
        "    exploration. The skill STAYS available and may bind to a "
        "    future item; it is NOT permanently excluded. The item "
        "    stays active — proceed with ``edit``.\n"
        "\n"
        "Call once per (plan_item_id, backing_skill) activation. After "
        "this call, edits are unblocked. update_plan is blocked while "
        "an item is active; wait until this item settles before "
        "submitting a new plan."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "plan_item_id": {
                "type": "string",
                "description": (
                    "The active plan item id this acknowledgement is for "
                    "(e.g. 'p1'). Must match the currently-active item."
                ),
            },
            "valuable_aspects": {
                "type": "string",
                "description": (
                    "What this skill teaches as a general pattern — the "
                    "algorithms / access patterns / known pitfalls worth "
                    "taking away. SKILL-LEVEL, not kernel-level. 100-500 "
                    "characters."
                ),
            },
            "kernel_application": {
                "type": "string",
                "description": (
                    "Concrete plan for applying the valuable aspects to "
                    "THIS kernel's code. Start with structural edits "
                    "(algorithm / access pattern / memory hierarchy / "
                    "kernel fusion-split) and only mention parameter "
                    "tuning when backed by a specific structural "
                    "hypothesis. 100-500 characters."
                ),
            },
            "applicability": {
                "type": "string",
                "enum": ["apply", "unbind"],
                "description": (
                    "'apply' — proceed with edits using the skill. "
                    "'unbind' — skill doesn't fit THIS item; release "
                    "the binding but keep the skill available for "
                    "future items."
                ),
            },
        },
        "required": [
            "plan_item_id", "valuable_aspects", "kernel_application",
            "applicability",
        ],
    },
}


COMPACT_TOOL = {
    "name": "compact",
    "description": (
        "Manually trigger context compression. Call when you notice the "
        "conversation is getting long or history is being truncated. "
        "Summarizes history into a compact summary. Does NOT trigger eval."
    ),
    "input_schema": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}


FINISH_TOOL = {
    "name": "finish",
    "description": (
        "Signal that optimization is complete. Call when you have exhausted "
        "improvement ideas or the framework will stop you at max_rounds."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": "Brief summary of what was accomplished.",
            },
        },
        "required": [],
    },
}


# ---------------------------------------------------------------------------
# Anthropic-native tools list + dispatch map factory
#
# ``TOOLS`` is the list passed to ``client.messages.create(tools=...)``.
# It mixes file-op tools (with local execute_* handlers) and
# state-mutation tools (handled inside TurnExecutor). The LLM sees
# them uniformly; the dispatch split happens in TurnExecutor.
#
# ``build_tool_handlers(task_dir, config)`` returns the {name: callable}
# dispatch map for file-op tools only, with task_dir/config closed over.
# update_plan / compact / finish are NOT in this map — TurnExecutor
# intercepts them before reaching it.
# ---------------------------------------------------------------------------


TOOLS = [
    READ_FILE_TOOL,
    EDIT_TOOL,
    UPDATE_PLAN_TOOL,
    SEARCH_SKILLS_TOOL,
    ACKNOWLEDGE_SKILL_TOOL,
    COMPACT_TOOL,
    FINISH_TOOL,
]


def build_tool_handlers(task_dir: str, config: "TaskConfig") -> dict:
    """
    Build the file-op dispatch map bound to a specific task.

    Returns {name: callable(**input) -> ToolResult}. The LLM-facing
    surface is just ``read_file`` + ``edit``; ``edit`` dispatches
    internally to the exact / block / unified / rewrite backends.
    update_plan / compact / finish are NOT included — TurnExecutor
    handles them as state mutations before dispatching to this map.
    """
    return {
        "read_file": lambda **kw: execute_read_file(
            kw["path"], task_dir,
            mode=kw.get("mode", "full"),
            target=kw.get("target"),
        ),
        "edit": lambda **kw: execute_edit(
            path=kw["path"], task_dir=task_dir, config=config,
            edits=kw.get("edits"),
            mode=kw.get("mode"),
            old_str=kw.get("old_str", ""), new_str=kw.get("new_str", ""),
            diff=kw.get("diff", ""),
            anchor_line=kw.get("anchor_line"),
            symbol=kw.get("symbol"),
            expected_delta_lines=kw.get("expected_delta_lines"),
        ),
    }
