"""
Context compression for the autoresearch agent conversation.

Three layers:
  1. microcompact — clears old tool_result content (cheap, every turn)
  2. auto_compact — multi-step rebuild with two independent LLM calls
     (operator summary + plan analysis). Triggered at context threshold
     or via the agent's ``compact`` tool.
  3. force_rebuild — emergency (PTL escape), no LLM calls. Reuses the
     same attachment builders but with no-LLM fallbacks for the
     operator summary and plan analysis sections.

Post-compact message shape:
  [COMPACT_BOUNDARY]    — boundary marker
  [BOOTSTRAP]           — current plan vN items + operator summary
  [STATE_ATTACHMENT:KERNEL]   — editable files, full content (sanity cap only)
  [STATE_ATTACHMENT:PLAN]     — structured plan.md analysis (LLM or fallback)
  [STATE_ATTACHMENT:RANKING]  — ranking.md, full content
  <recent rounds from input>

The operator summary and plan analysis land on disk under
``agent_session/`` as ``op_summary.md`` and ``plan_analysis.md`` for
post-mortem inspection.
"""

import asyncio
import json
import logging
import os
import re
from collections import Counter
from typing import Iterable, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Message markers
# ---------------------------------------------------------------------------

COMPACT_BOUNDARY = "[COMPACT_BOUNDARY]"
BOOTSTRAP_MARKER = "[BOOTSTRAP]"
STATE_ATTACHMENT = "[STATE_ATTACHMENT]"  # generic prefix for backward-compat matches
STATE_KERNEL = "[STATE_ATTACHMENT:KERNEL]"
STATE_PLAN = "[STATE_ATTACHMENT:PLAN]"
STATE_RANKING = "[STATE_ATTACHMENT:RANKING]"
OPERATOR_SUMMARY_MARKER = "[OPERATOR_SUMMARY]"


# ---------------------------------------------------------------------------
# Normalization (for accurate serialization of internal message objects)
# ---------------------------------------------------------------------------

def _normalize_content(content):
    """Convert _TextBlock/_ToolUseBlock objects to plain dicts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        normalized = []
        for item in content:
            if isinstance(item, dict):
                normalized.append(item)
            elif hasattr(item, "type"):
                if item.type == "text":
                    normalized.append({"type": "text", "text": item.text})
                elif item.type == "tool_use":
                    normalized.append({
                        "type": "tool_use", "id": item.id,
                        "name": item.name, "input": item.input,
                    })
                else:
                    normalized.append(str(item))
            else:
                normalized.append(str(item))
        return normalized
    return content


def _normalize_messages(messages: list) -> list:
    return [
        {**msg, "content": _normalize_content(msg.get("content", ""))}
        for msg in messages
    ]


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

def estimate_tokens(messages: list, chars_per_token: int = 4) -> int:
    """Rough token estimate from messages only."""
    return len(json.dumps(_normalize_messages(messages))) // chars_per_token


def estimate_full_request_tokens(messages: list, system_prompt: str,
                                 tools: list = None,
                                 chars_per_token: int = 3) -> int:
    """Estimate total API request tokens (messages + system + tools)."""
    total = len(json.dumps(_normalize_messages(messages)))
    total += len(system_prompt)
    if tools:
        total += len(json.dumps(tools))
    return total // chars_per_token


# ---------------------------------------------------------------------------
# Microcompact
# ---------------------------------------------------------------------------

# Prefixes on user-role string messages that microcompact can replace
# with "[cleared]" once stale. Empty by default (the "[System] Skill
# guidance for " producer was removed with the supervisor system).
# Kept as a tuple so adding a future clearable prefix is one line.
_CLEARABLE_PREFIXES: tuple[str, ...] = ()


def microcompact(messages: list, min_chars: int = 200, keep_recent: int = 1):
    """Replace old tool_result content with "[cleared]"."""
    clearable: list = []
    for msg in messages:
        if msg["role"] != "user":
            continue
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "tool_result":
                    clearable.append(("tool_result", part, None))
        elif isinstance(content, str):
            if any(content.startswith(p) for p in _CLEARABLE_PREFIXES):
                clearable.append(("guidance", None, msg))

    if len(clearable) <= keep_recent:
        return
    for kind, part, msg in clearable[:-keep_recent]:
        if kind == "tool_result":
            if isinstance(part.get("content"), str) and len(part["content"]) > min_chars:
                part["content"] = "[cleared]"
        elif kind == "guidance":
            if isinstance(msg.get("content"), str) and len(msg["content"]) > min_chars:
                msg["content"] = "[cleared]"


# ---------------------------------------------------------------------------
# Transcript file IO
# ---------------------------------------------------------------------------

def _save_messages(messages: list, task_dir: str,
                   session_dir: str = "agent_session",
                   filename: str = "messages_latest.jsonl",
                   mode: str = "w"):
    msg_dir = os.path.join(task_dir, session_dir, "messages")
    os.makedirs(msg_dir, exist_ok=True)
    path = os.path.join(msg_dir, filename)
    with open(path, mode, encoding="utf-8") as f:
        for msg in _normalize_messages(messages):
            f.write(json.dumps(msg) + "\n")


def _load_messages(task_dir: str, session_dir: str = "agent_session",
                   filename: str = "messages_latest.jsonl") -> list | None:
    """Load messages from a JSONL file. Returns None if unavailable."""
    path = os.path.join(task_dir, session_dir, "messages", filename)
    if not os.path.exists(path):
        return None
    try:
        messages = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    messages.append(json.loads(line))
        return messages if messages else None
    except Exception:
        return None


def _read_safe(path: str) -> str:
    if not os.path.exists(path):
        return ""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except Exception:
        return ""


def _write_safe(path: str, content: str):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
    except Exception as exc:
        logger.debug("[compress] _write_safe(%s) failed: %r", path, exc)


def _read_file_full(task_dir: str, name: str, sanity_cap: int) -> tuple[str, bool]:
    """Read a file in full. Caps at ``sanity_cap`` chars with a warning
    flag (second return value) if the file is pathologically large.
    Returns ``(content, truncated_flag)``.
    """
    fpath = os.path.join(task_dir, name)
    if not os.path.exists(fpath):
        return "", False
    try:
        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except Exception:
        return "", False
    if len(content) > sanity_cap:
        logger.warning(
            "[compress] %s exceeds sanity cap (%d chars, cap=%d); truncating",
            name, len(content), sanity_cap,
        )
        return content[:sanity_cap] + (
            f"\n... [WARNING: file truncated at sanity cap {sanity_cap} chars]"
        ), True
    return content, False


# ---------------------------------------------------------------------------
# Message classification
# ---------------------------------------------------------------------------

def _find_last_boundary(messages: list):
    """Find the last compact boundary (search from end)."""
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if (msg["role"] == "user"
                and isinstance(msg.get("content"), str)
                and COMPACT_BOUNDARY in msg["content"]):
            return i
    return None


def _is_rebuild_message(msg: dict) -> bool:
    """True if this message is a compact-generated marker (boundary /
    bootstrap / any STATE_ATTACHMENT:* variant).

    NOTE: ``STATE_ATTACHMENT`` is ``"[STATE_ATTACHMENT]"`` — the
    bracketed form is NOT a substring of ``"[STATE_ATTACHMENT:KERNEL]"``
    and friends. Match against the open-bracket prefix so every
    current and future ``:VARIANT`` tag is recognized.
    """
    if msg["role"] != "user" or not isinstance(msg.get("content"), str):
        return False
    c = msg["content"]
    return (COMPACT_BOUNDARY in c
            or BOOTSTRAP_MARKER in c
            or "[STATE_ATTACHMENT" in c)


def _group_by_api_round(messages: list) -> list:
    """Group by API round-trip. New assistant message = new round.

    Preserves tool_use → tool_result pairing within a round.
    """
    rounds, current = [], []
    for msg in messages:
        if msg["role"] == "assistant" and current:
            rounds.append(current)
            current = []
        current.append(msg)
    if current:
        rounds.append(current)
    return rounds


# ---------------------------------------------------------------------------
# PTL detection
# ---------------------------------------------------------------------------

def _is_prompt_too_long(exc) -> bool:
    err = str(exc).lower()
    return any(k in err for k in (
        "input length", "exceeds the maximum",
        "prompt_too_long", "too long"))


# ---------------------------------------------------------------------------
# Session-dir helpers
# ---------------------------------------------------------------------------

def _op_summary_path(task_dir: str, session_dir: str) -> str:
    return os.path.join(task_dir, session_dir, "op_summary.md")


def _plan_analysis_path(task_dir: str, session_dir: str) -> str:
    return os.path.join(task_dir, session_dir, "plan_analysis.md")


# ---------------------------------------------------------------------------
# Keyword aggregation
# ---------------------------------------------------------------------------

def _collect_historical_keywords(feedback) -> list[tuple[str, int]]:
    """Aggregate every keyword the agent has submitted on any plan item
    across the entire run. Returns a list of ``(keyword, count)`` pairs
    sorted by frequency descending, then alphabetically.

    Sources:
      * feedback._plan_items[*].keywords   — current plan's items
      * feedback._settled_history[*].keywords — all past plan items,
        across every plan version, including those that were
        "superseded by replan" or "abandoned (diagnose)".
    """
    if feedback is None:
        return []
    counter: Counter = Counter()
    for item in getattr(feedback, "_plan_items", []) or []:
        for kw in item.get("keywords") or []:
            kw_norm = (kw or "").strip().lower()
            if kw_norm:
                counter[kw_norm] += 1
    for entry in getattr(feedback, "_settled_history", []) or []:
        for kw in entry.get("keywords") or []:
            kw_norm = (kw or "").strip().lower()
            if kw_norm:
                counter[kw_norm] += 1
    return sorted(counter.items(), key=lambda p: (-p[1], p[0]))


# ---------------------------------------------------------------------------
# LLM call wrapper
# ---------------------------------------------------------------------------

async def _call_llm_text(llm, *, system_prompt: str, user_prompt: str,
                         max_tokens: int, max_retries: int) -> str:
    """Call the LLM with compact=True (no thinking, text-only), retrying
    on PTL by truncating the user prompt. Returns the response text.
    Raises on non-PTL failures and on PTL exhaustion.
    """
    messages = [{"role": "user", "content": user_prompt}]
    attempt = 0
    while True:
        try:
            response = await llm.call(
                system_prompt=system_prompt,
                messages=messages,
                tools=[],
                compact=True,
                max_tokens=max_tokens,
            )
            text = llm.get_response_text(response) or ""
            return text.strip()
        except Exception as exc:
            if attempt >= max_retries or not _is_prompt_too_long(exc):
                raise
            # PTL: halve the user prompt and retry
            prompt = messages[0]["content"]
            if len(prompt) < 1000:
                raise
            messages = [{
                "role": "user",
                "content": prompt[: len(prompt) // 2]
                + "\n\n[... input truncated due to prompt-too-long ...]",
            }]
            attempt += 1


# ---------------------------------------------------------------------------
# Operator summary (LLM call #1)
# ---------------------------------------------------------------------------

def _load_compress_prompt(name: str) -> str:
    """Load a prompt template from ``agent/prompts/``.

    Externalising these keeps compress.py focused on pipeline
    mechanics (when to trigger, what to keep) and lets prompt edits
    land without touching Python. Missing file is treated as a bug:
    we fail loudly rather than silently emit an empty prompt, which
    would be far worse than a clean ImportError.
    """
    import os
    prompt_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "prompts", name,
    )
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read().rstrip()


_OP_SUMMARY_SYSTEM = _load_compress_prompt("compress_operator_summary.md")


def _op_summary_fallback(op_name: str,
                         keywords: list[tuple[str, int]]) -> str:
    """Plain-text fallback used when the LLM call fails."""
    lines = [
        f"## Operator Shape",
        f"Operator: {op_name}. (LLM summary unavailable — using raw keyword dump.)",
        "",
        "## Exploration Signals",
    ]
    if keywords:
        kw_str = ", ".join(
            f"{k}×{n}" if n > 1 else k for k, n in keywords[:30]
        )
        lines.append(f"Keywords seen so far (freq): {kw_str}")
    else:
        lines.append("No keywords recorded yet.")
    return "\n".join(lines)


async def _summarize_operator_from_keywords(llm, config, feedback,
                                            task_dir: str) -> str:
    """LLM call #1. Produces an operator-level summary from op_name +
    reference code head + historical keyword frequencies. Persists to
    ``agent_session/op_summary.md``. Falls back to a plain keyword dump
    on failure.
    """
    a = config.agent
    session_dir = getattr(a, "session_dir", "agent_session")

    op_name = getattr(config, "name", "") or ""
    # Honour ``TaskConfig.ref_file`` — tasks that don't come from the
    # default scaffolder may place the reference somewhere other than
    # ``reference.py``. ``ref_file`` is declared on ``TaskConfig``
    # (config root), NOT on ``AgentConfig`` — don't read it off ``a``.
    # Falls back to the scaffolder default for legacy callers that
    # don't set ref_file explicitly.
    ref_file = getattr(config, "ref_file", None) or "reference.py"
    ref_path = os.path.join(task_dir, ref_file)
    ref_head = _read_safe(ref_path)[:3000] if os.path.exists(ref_path) else ""
    keywords = _collect_historical_keywords(feedback)
    kw_list_str = (
        "\n".join(f"- {kw}: {count}" for kw, count in keywords[:60])
        or "(none recorded yet)"
    )

    user_prompt = (
        f"Operator name: {op_name}\n\n"
        f"Reference code (head excerpt):\n```python\n{ref_head}\n```\n\n"
        f"Keyword frequencies (agent-submitted on plan items, across the "
        f"whole run):\n{kw_list_str}\n"
    )

    try:
        text = await _call_llm_text(
            llm,
            system_prompt=_OP_SUMMARY_SYSTEM,
            user_prompt=user_prompt,
            max_tokens=getattr(a, "compact_op_summary_max_tokens", 500),
            max_retries=getattr(a, "compact_max_retries", 3),
        )
    except Exception as exc:
        logger.warning("[compress] operator summary LLM failed: %r", exc)
        text = _op_summary_fallback(op_name, keywords)

    if not text.strip():
        text = _op_summary_fallback(op_name, keywords)

    _write_safe(_op_summary_path(task_dir, session_dir), text)
    return text


# ---------------------------------------------------------------------------
# Plan analysis (LLM call #2)
# ---------------------------------------------------------------------------

_PLAN_ANALYSIS_SYSTEM = _load_compress_prompt("compress_plan_analysis.md")


def _plan_analysis_fallback(plan_md_content: str,
                            fallback_chars: int) -> str:
    """Return raw plan.md truncated, with a clear marker that the LLM
    summary is unavailable so the agent knows to read it carefully.
    """
    body = plan_md_content
    if len(body) > fallback_chars:
        body = body[:fallback_chars] + "\n...[truncated]"
    return (
        "## Plan (analysis unavailable, raw)\n\n"
        "LLM-based structured analysis failed — falling back to the\n"
        "raw plan.md content below (truncated). Read it carefully; the\n"
        "usual 5-section structure is not available this cycle.\n\n"
        f"{body}"
    )


async def _analyze_plan_md(llm, config, task_dir: str) -> str:
    """LLM call #2. Reads plan.md in full, sends to LLM, returns a
    5-section structured analysis. Persists to
    ``agent_session/plan_analysis.md``. Falls back to truncated raw
    plan.md on LLM failure.
    """
    a = config.agent
    session_dir = getattr(a, "session_dir", "agent_session")
    plan_path = os.path.join(task_dir, "plan.md")
    plan_md = _read_safe(plan_path)
    if not plan_md.strip():
        text = "## Plan (empty)\n\nNo plan.md content yet — nothing to analyze."
        _write_safe(_plan_analysis_path(task_dir, session_dir), text)
        return text

    user_prompt = f"Here is the current plan.md:\n\n{plan_md}\n"
    fallback_chars = getattr(a, "compact_plan_raw_fallback_chars", 6_000)

    try:
        text = await _call_llm_text(
            llm,
            system_prompt=_PLAN_ANALYSIS_SYSTEM,
            user_prompt=user_prompt,
            max_tokens=getattr(a, "compact_plan_analysis_max_tokens", 1_500),
            max_retries=getattr(a, "compact_max_retries", 3),
        )
    except Exception as exc:
        logger.warning("[compress] plan analysis LLM failed: %r", exc)
        text = _plan_analysis_fallback(plan_md, fallback_chars)

    if not text.strip():
        text = _plan_analysis_fallback(plan_md, fallback_chars)

    _write_safe(_plan_analysis_path(task_dir, session_dir), text)
    return text


# ---------------------------------------------------------------------------
# Attachment builders
# ---------------------------------------------------------------------------

def _build_kernel_attachment(task_dir: str, config,
                             *, max_chars_override: Optional[int] = None) -> list:
    """Full editable files. Capped at ``compact_kernel_sanity_cap``
    chars per file (default 80 000) with a warning flag appended.
    ``max_chars_override`` lets the PTL-recovery ``force_rebuild`` path
    request a much tighter cap (e.g. 20 000) to guarantee the rebuilt
    buffer is strictly smaller than the one that just tripped PTL."""
    a = config.agent
    sanity_cap = (
        max_chars_override
        if max_chars_override is not None
        else getattr(a, "compact_kernel_sanity_cap", 80_000)
    )
    parts: list[str] = []
    for fname in config.editable_files:
        content, _truncated = _read_file_full(task_dir, fname, sanity_cap)
        if not content:
            continue
        parts.append(f"## {fname}\n```\n{content}\n```")
    if not parts:
        return []
    return [{
        "role": "user",
        "content": f"{STATE_KERNEL}\n" + "\n\n".join(parts),
    }]


def _build_plan_attachment(plan_analysis_text: str) -> list:
    """Structured plan analysis (already produced by ``_analyze_plan_md``
    or the fallback path). Body is rendered verbatim."""
    if not plan_analysis_text.strip():
        return []
    return [{
        "role": "user",
        "content": f"{STATE_PLAN}\n{plan_analysis_text}",
    }]


def _build_ranking_attachment(task_dir: str,
                              *, max_chars: Optional[int] = None) -> list:
    """Full ranking.md content. No truncation in the normal auto_compact
    path (``max_chars=None``). Empty if the file is missing.

    ``max_chars`` is only set by the PTL-recovery ``force_rebuild`` path
    so the rebuilt buffer is guaranteed smaller than the one that
    tripped PTL; a long run's ranking.md can easily exceed the LLM
    window, and a compact-then-force-rebuild loop must not leave the
    same content in place.
    """
    content = _read_safe(os.path.join(task_dir, "ranking.md"))
    if not content.strip():
        return []
    if max_chars is not None and len(content) > max_chars:
        head = content[:max_chars].rstrip()
        content = (
            f"{head}\n\n"
            f"...[ranking.md truncated at {max_chars} chars during "
            f"force_rebuild; read the file directly for full history]"
        )
    return [{
        "role": "user",
        "content": f"{STATE_RANKING}\n## ranking.md\n{content}",
    }]


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def _render_current_plan_items(feedback) -> list[str]:
    """Render the current plan's items (no phase status, no history).
    Uses ``feedback_validation.render_item_line`` so the format matches
    the live plan.md header block.
    """
    if feedback is None:
        return []
    from . import feedback_validation as _fv
    items = getattr(feedback, "_plan_items", []) or []
    if not items:
        return []
    version = getattr(feedback, "plan_version", 0) or 0
    lines = [f"## Current Plan v{version}"]
    for item in items:
        lines.append(_fv.render_item_line(item))
    return lines


def _build_bootstrap(task_dir: str, config, feedback=None,
                     op_summary: str = "",
                     last_diagnosis: str | None = None,
                     best_metric_str: str = "") -> list:
    """Rebuild run contract after compact.

    Sections (top → bottom):
      1. Task header + goal anchor
      2. Current plan items (vN — no phase text, no history)
      3. ``[OPERATOR_SUMMARY]`` (from LLM #1 or fallback)
      4. must_replan warning if applicable
      5. Phase-aware next-action instruction
    """
    # Goal anchor — single source of truth is feedback.format_goal_anchor.
    fmt = (
        getattr(feedback, "format_goal_anchor", None)
        if feedback is not None else None
    )
    if callable(fmt):
        goal_line = fmt(best_metric_str)
    else:
        direction = "lower is better" if config.lower_is_better else "higher is better"
        goal_line = f"Goal: {config.primary_metric} ({direction})."

    lines = [
        f"# Continuing: {config.name}",
        goal_line,
        "",
        "Current code, ranking, and plan analysis are in the "
        "[STATE_ATTACHMENT:*] messages below.",
        "Do NOT assume initial prompt content is still available.",
    ]
    if os.path.isdir(os.path.join(task_dir, "docs")):
        lines.append("DSL docs available via read_file in docs/.")

    # Current plan items (no phase status / history — that's in plan.md analysis)
    plan_lines = _render_current_plan_items(feedback)
    if plan_lines:
        lines.append("")
        lines.extend(plan_lines)

    # Operator summary block
    if op_summary.strip():
        lines.append("")
        lines.append(OPERATOR_SUMMARY_MARKER)
        lines.append(op_summary.strip())

    # must_replan warning
    diag_limit = getattr(config.agent, "compact_diagnosis_truncate", 2_000)
    if feedback is not None and getattr(feedback, "must_replan", False):
        lines.append("")
        lines.append("⚠ DIRECTION CHANGE REQUIRED.")
        if last_diagnosis:
            lines.append(f"Diagnostic report:\n{last_diagnosis[:diag_limit]}")

    # Phase-aware next-action instruction.
    _phase = getattr(feedback, "phase", None) if feedback is not None else None
    _must_replan = (
        getattr(feedback, "must_replan", False) if feedback is not None else False
    )
    if _phase == "active" and not _must_replan:
        lines.append("")
        lines.append(
            "Continue editing the current active plan item; do NOT call "
            "update_plan (it is blocked while an item is active). "
            "Make the next code change with `edit`."
        )
    else:
        lines.append("")
        lines.append(
            "Submit a plan via update_plan(...), then edit code. "
            "Algorithmic changes FIRST, parameter tuning SECOND."
        )

    return [{
        "role": "user",
        "content": f"{BOOTSTRAP_MARKER}\n" + "\n".join(lines),
    }]


# ---------------------------------------------------------------------------
# auto_compact
# ---------------------------------------------------------------------------

async def auto_compact(messages: list, llm, task_dir: str, *, config, tools,
                       feedback=None, last_diagnosis=None,
                       keep_recent_rounds: int = 3,
                       best_metric_str: str = "") -> list:
    """Compress context. Returns a new messages list, or the input list
    unchanged (identity = "no-op signal").

    Flow:
      1. Split into (old rounds, recent rounds).
      2. Kick off two independent LLM summaries in parallel:
         - _summarize_operator_from_keywords (LLM #1)
         - _analyze_plan_md (LLM #2)
      3. Either task may fail independently — each has its own fallback.
      4. Build new buffer as:
           [COMPACT_BOUNDARY, BOOTSTRAP, KERNEL, PLAN, RANKING, *recent]
    """
    a = config.agent
    session_dir = getattr(a, "session_dir", "agent_session")

    # 1. Strip previous rebuild markers from the incremental range.
    bi = _find_last_boundary(messages)
    incremental = messages[bi + 1:] if bi is not None else messages
    live = [m for m in incremental if not _is_rebuild_message(m)]

    # 2. Group into rounds — need at least keep_recent + 1 to compress.
    rounds = _group_by_api_round(live)
    if keep_recent_rounds > 0 and len(rounds) <= keep_recent_rounds + 1:
        return messages  # identity = no-op

    if keep_recent_rounds > 0:
        recent = [m for r in rounds[-keep_recent_rounds:] for m in r]
    else:
        recent = []

    # 3. Log the compact event.
    _save_messages(
        [{"role": "system", "content": "[COMPACT]"}],
        task_dir, session_dir=session_dir,
        filename="messages_full.jsonl", mode="a")

    # 4. Launch both LLM summaries concurrently. Each helper already
    #    catches its own exceptions and returns a fallback string, so
    #    gather should never return exceptions in practice — the
    #    return_exceptions=True is belt-and-braces.
    op_task = _summarize_operator_from_keywords(llm, config, feedback, task_dir)
    plan_task = _analyze_plan_md(llm, config, task_dir)
    results = await asyncio.gather(op_task, plan_task, return_exceptions=True)

    op_summary = results[0] if isinstance(results[0], str) else ""
    plan_analysis = results[1] if isinstance(results[1], str) else ""

    # If gather surfaced an unexpected exception (should be rare since
    # helpers catch internally), degrade to in-function fallbacks here.
    if not isinstance(results[0], str):
        logger.warning(
            "[compress] operator summary task raised: %r", results[0],
        )
        op_summary = _op_summary_fallback(
            getattr(config, "name", "") or "",
            _collect_historical_keywords(feedback),
        )
        _write_safe(_op_summary_path(task_dir, session_dir), op_summary)
    if not isinstance(results[1], str):
        logger.warning(
            "[compress] plan analysis task raised: %r", results[1],
        )
        plan_md = _read_safe(os.path.join(task_dir, "plan.md"))
        plan_analysis = _plan_analysis_fallback(
            plan_md, getattr(a, "compact_plan_raw_fallback_chars", 6_000),
        )
        _write_safe(_plan_analysis_path(task_dir, session_dir), plan_analysis)

    # 5. Assemble the new buffer.
    boundary = {
        "role": "user",
        "content": f"{COMPACT_BOUNDARY}\n(compact complete — see BOOTSTRAP + attachments below)",
    }
    bootstrap = _build_bootstrap(
        task_dir, config, feedback,
        op_summary=op_summary,
        last_diagnosis=last_diagnosis,
        best_metric_str=best_metric_str,
    )
    kernel_att = _build_kernel_attachment(task_dir, config)
    plan_att = _build_plan_attachment(plan_analysis)
    ranking_att = _build_ranking_attachment(task_dir)

    return [boundary] + bootstrap + kernel_att + plan_att + ranking_att + recent


# ---------------------------------------------------------------------------
# force_rebuild (emergency — no LLM call)
# ---------------------------------------------------------------------------

def force_rebuild_minimal_context(task_dir: str, config, feedback=None,
                                  last_diagnosis=None,
                                  best_metric_str: str = "") -> list:
    """Emergency rebuild used on PTL recovery or when auto_compact fails.
    Reuses the same attachment builders as auto_compact but substitutes
    the LLM-produced summaries with no-LLM fallbacks:

      - operator summary ← plain keyword dump
      - plan analysis    ← truncated raw plan.md with "analysis unavailable"
    """
    a = config.agent
    session_dir = getattr(a, "session_dir", "agent_session")

    # Operator summary: keyword dump fallback.
    keywords = _collect_historical_keywords(feedback)
    op_summary = _op_summary_fallback(
        getattr(config, "name", "") or "", keywords,
    )
    _write_safe(_op_summary_path(task_dir, session_dir), op_summary)

    # Plan analysis: raw plan.md truncated.
    plan_md = _read_safe(os.path.join(task_dir, "plan.md"))
    fallback_chars = getattr(a, "compact_plan_raw_fallback_chars", 6_000)
    if plan_md.strip():
        plan_analysis = _plan_analysis_fallback(plan_md, fallback_chars)
    else:
        plan_analysis = "## Plan (empty)\n\nNo plan.md content yet."
    _write_safe(_plan_analysis_path(task_dir, session_dir), plan_analysis)

    boundary = {
        "role": "user",
        "content": f"{COMPACT_BOUNDARY}\n(force_rebuild — no LLM summary this cycle)",
    }
    bootstrap = _build_bootstrap(
        task_dir, config, feedback,
        op_summary=op_summary,
        last_diagnosis=last_diagnosis,
        best_metric_str=best_metric_str,
    )
    # Force_rebuild is the PTL escape hatch: the current buffer was just
    # rejected as too long by the LLM, so the rebuilt buffer MUST be
    # strictly smaller. Apply tighter per-attachment caps here — the
    # normal auto_compact path keeps the full 80k kernel cap and
    # uncapped ranking.md.
    rebuild_kernel_cap = getattr(
        a, "compact_rebuild_kernel_cap", 20_000,
    )
    rebuild_ranking_cap = getattr(
        a, "compact_rebuild_ranking_cap", 8_000,
    )
    kernel_att = _build_kernel_attachment(
        task_dir, config, max_chars_override=rebuild_kernel_cap,
    )
    plan_att = _build_plan_attachment(plan_analysis)
    ranking_att = _build_ranking_attachment(
        task_dir, max_chars=rebuild_ranking_cap,
    )

    return [boundary] + bootstrap + kernel_att + plan_att + ranking_att
