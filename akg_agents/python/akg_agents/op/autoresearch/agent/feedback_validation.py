# Copyright 2025-2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Stateless helpers lifted out of FeedbackBuilder.

FeedbackBuilder was a 1100-line god-object mixing plan-state mutation
with text sanitization, validation, reason classification, history
filtering, and line rendering. None of the latter touch ``self.*``
state — they are pure functions of their arguments — so they belong at
module scope rather than as private methods.

What lives here:

  * ``sanitize_text``        — trim + cap a string field.
  * ``validate_rationale``   — enforce the "not generic, not too short"
                               rule on plan-item rationales.
  * ``classify_reason``      — compress an eval-failure reason string
                               into one of a small vocabulary of short
                               labels (used in settled_history.reason
                               and plan.md rendering).
  * ``is_eval_signal``       — decide whether a settled-history entry
                               reflects a real KEEP/FAIL/DISCARD eval
                               outcome vs. a control event (replan
                               supersede, screening skip, …).
  * ``render_item_line``     — markdown one-liner for a single plan
                               item (shared by format_status and
                               format_plan_file).

FeedbackBuilder wraps these as thin methods so existing call sites
continue to work, but every new consumer should import from this
module directly.
"""

from typing import Optional


# -- Text sanitization --------------------------------------------------------


def sanitize_text(raw, max_chars: int) -> str:
    """Trim + cap a single text field. Returns ``""`` for non-string/empty."""
    if not isinstance(raw, str):
        return ""
    cleaned = raw.strip()
    if not cleaned:
        return ""
    if max_chars > 0 and len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars].rstrip() + "\u2026"
    return cleaned


# -- Rationale validation ----------------------------------------------------


RATIONALE_BANNED_PHRASES = (
    "optimize the kernel",
    "improve performance",
    "make it faster",
    "make the kernel faster",
    "better performance",
    "boost performance",
)

# -- Plan diversity: parameter-tuning detection --------------------------------
# Ported from claude-autoresearch's create_plan.py

PARAM_TUNING_WORDS = frozenset({
    "block", "tile", "tiling", "autotune", "config", "configs",
    "warps", "stages", "size", "tune", "adjust", "sweep",
    "parameter", "param", "group", "num",
})

PARAM_TUNING_PHRASES = frozenset({
    "block_size", "block_m", "block_n", "block_k", "block_size_m",
    "block_size_n", "block_size_k", "num_warps", "num_stages",
    "group_size", "group_size_m",
})


def is_param_tuning_only(keywords: list[str]) -> bool:
    """Return True if the keyword set indicates pure parameter tuning."""
    if not keywords:
        return False
    words = set()
    phrases = set()
    for kw in keywords:
        phrase = kw.strip().lower().replace("-", "_").replace(" ", "_")
        phrases.add(phrase)
        for w in phrase.split("_"):
            if w:
                words.add(w)
    if not words:
        return False
    has_param_phrase = bool(phrases & PARAM_TUNING_PHRASES)
    non_param_words = words - PARAM_TUNING_WORDS - {""}
    return has_param_phrase or not non_param_words


def check_plan_diversity(items_keywords: list[list[str]],
                         max_param_items: int = 1) -> tuple[bool, str]:
    """Reject plans where too many items are parameter-tuning only.

    Returns (ok, error_message). ok=True means plan is diverse enough.
    """
    if not items_keywords:
        return True, ""
    param_count = sum(1 for kws in items_keywords if is_param_tuning_only(kws))
    allowed = max(max_param_items, len(items_keywords) - 1)
    # Reject if all-but-one (or more) are param-only
    if param_count > allowed:
        return False, (
            f"Diversity rejected: {param_count}/{len(items_keywords)} items are "
            f"parameter tuning. Bundle parameter sweeps into ONE item. Other "
            f"items must be structurally different (algorithmic changes, fusion, "
            f"memory access patterns, data layout)."
        )
    return True, ""


def validate_rationale(raw, *, min_chars: int,
                       max_chars: int) -> tuple[Optional[str], str]:
    """Validate + normalize a plan-item rationale.

    Returns ``(cleaned, error)`` where:
      - cleaned is the validated string when accepted, ``None`` when rejected
      - error is a one-line reason suitable to surface in the
        "Plan rejected" message; empty when accepted

    Rules:
      1. Required (non-empty after strip)
      2. Length >= ``min_chars``
      3. Not dominated by a banned generic phrase (unless length >=
         ``2 * min_chars``, at which point the specific detail
         presumably outweighs the generic framing)
      4. Capped at ``max_chars``
    """
    cleaned = sanitize_text(raw, max_chars)
    if not cleaned:
        return None, (
            "rationale is required (one sentence: name the bottleneck "
            "and the expected effect)"
        )
    if len(cleaned) < min_chars:
        return None, (
            f"rationale too short (< {min_chars} chars); name the "
            f"specific bottleneck and the expected effect"
        )
    lowered = cleaned.lower()
    for phrase in RATIONALE_BANNED_PHRASES:
        if phrase in lowered and len(cleaned) < min_chars * 2:
            return None, (
                f"rationale too generic ({phrase!r}); state the actual "
                f"bottleneck (which loop / which memory / which op) "
                f"and the expected effect"
            )
    return cleaned, ""


# -- Reason classification ---------------------------------------------------


def classify_reason(reason: str) -> str:
    """Classify a failure reason into a short label.

    Labels are deliberately coarse — they drive the plan.md "status"
    column and pattern detection in ``format_history_signal``, so new
    labels should only be added when a user-visible distinction is
    worth rendering.
    """
    if not reason:
        return "fail"
    r = reason.lower()
    if "no improvement" in r:
        return "no improvement"
    if "timed out" in r:
        return "timeout"
    # Compile-time errors are distinct from numerical correctness:
    # a typo / unresolved name is likely transient (fixable on retry
    # with the same plan intent).
    if ("compilationerror" in r or "nameerror" in r
            or "attributeerror" in r or "syntaxerror" in r
            or "is not defined" in r):
        return "compile error"
    if "correctness" in r or "err_cnt" in r or "验证失败" in r:
        return "correctness"
    if "ub overflow" in r:
        return "ub overflow"
    if "syntax" in r:
        return "syntax error"
    return "fail"


# -- Eval-signal classifier --------------------------------------------------


CONTROL_REASONS_EXACT = frozenset({
    "superseded by replan",
    "abandoned (diagnose)",
})
# Prefix-based control-event matches. Empty in the post-supervisor
# tree (the "screened:" producer was removed with the pre-eval
# screener). Kept as a tuple so a future producer is a one-line
# addition; ``is_eval_signal`` iterates regardless of length.
CONTROL_REASON_PREFIXES: tuple[str, ...] = ()


def is_eval_signal(ok: bool, reason: str) -> bool:
    """True if the entry came from an actual KEEP/FAIL/DISCARD eval.

    Successful settles (KEEP) are always eval signals. Failures only
    count when the ``reason`` does not match a control-event pattern
    (replan supersede, diagnose-forced replan).
    """
    if ok:
        return True
    r = reason or ""
    if r in CONTROL_REASONS_EXACT:
        return False
    for prefix in CONTROL_REASON_PREFIXES:
        if r.startswith(prefix):
            return False
    return True


# -- Plan-item rendering -----------------------------------------------------


def render_item_line(item: dict) -> str:
    """Render a single plan item as a one-line markdown bullet.

    Shared by ``format_status`` and ``format_plan_file`` so the
    marker mapping and backing-skill suffix stay single-sourced.
    """
    status = item["status"]
    item_id = item["id"]
    if status == "active":
        marker = f">>> [{item_id}]"
    elif status == "done_ok":
        marker = f"[O] [{item_id}]"
    elif status == "done_fail":
        marker = f"[X] [{item_id}]"
    elif status == "skipped":
        marker = f"[-] [{item_id}]"
    else:  # pending
        marker = f"[ ] [{item_id}]"
    bs = item.get("backing_skill")
    if bs:
        details = [f"backing: {bs} (read: skills/{bs}/SKILL.md)"]
    else:
        details = ["unbound"]
    keywords = list(item.get("keywords") or [])
    if keywords:
        details.append("keywords: " + ", ".join(keywords))
    suffix = f" ({'; '.join(details)})"
    return f"- {marker} {item['text']}{suffix}"
