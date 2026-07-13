# Copyright 2026 Huawei Technologies Co., Ltd
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

"""Plan/diagnose validators + plan.md parser.

Validators (each: "is this artifact OK enough to advance the phase?"):

  - validate_plan: structural check on plan.md (≥3 items, rationale length,
    exactly one ACTIVE).
  - validate_diagnose: marker + sections on diagnose_v<N>.md.

Reference validation lives elsewhere: the AST symbol check is in
[utils/ref_ast.py](../utils/ref_ast.py) and runs once at scaffold time;
runtime behaviour is validated by scaffold's `--run-baseline` (run_verify
in eval_kernel.py tags `error_source="ref"` on ref-side failure).

Plan.md parsing (`parse_plan_text`, `get_plan_items`, `has_pending_items`,
`get_active_item`, `is_settled_table_header`) is the single source of
truth for plan-file structure; phase_policy, guidance, create_plan.py
and pipeline.py's inlined settle step all consume it from here.
"""
import os
import re
from typing import NamedTuple, Optional

from .state_store import (  # noqa: E402
    plan_path,
    diagnose_artifact_path, diagnose_marker,
    load_progress, DIAGNOSE_ATTEMPTS_CAP,
)


# ---------------------------------------------------------------------------
# plan.md parser + structural validation
# ---------------------------------------------------------------------------

_PLAN_ITEM_RE = re.compile(r'\s*-\s*\[([ x])\]\s*\*\*(\w+)\*\*\s*(.*)')
_PLAN_TAG_RE = re.compile(r'^\[([^\]]*)\]:?\s*(.*)')
_PLAN_SKILL_RE = re.compile(r"[\w.-]+[/\\]SKILL\.md\b", re.IGNORECASE)


def is_settled_table_header(line: str) -> bool:
    """True iff `line` is the Settled-History markdown table header.

    Both create_plan.py and pipeline.py's inlined settle step find the
    same row to either append a new settled entry (settle) or carry
    forward existing rows (create_plan). Centralising the predicate
    keeps the table format defined in one place.
    """
    s = line.strip()
    return s.startswith("|") and "Item" in s and "Outcome" in s


def parse_plan_text(text: str, include_meta: bool = False) -> list:
    """Canonical plan.md parser, on already-loaded text. Returns
    [{id, description, done, active, tag}, ...]. With include_meta=True
    also captures the `- rationale:` and `- skill:` sub-lines.

    Every plan reader in the codebase must go through this function or
    `get_plan_items` — no ad-hoc regex scans (drift risk)."""
    lines = text.split("\n")

    out = []
    i = 0
    while i < len(lines):
        m = _PLAN_ITEM_RE.match(lines[i])
        if not m:
            i += 1
            continue
        done = m.group(1) == 'x'
        pid = m.group(2)
        rest = m.group(3).strip()
        is_active = "(ACTIVE)" in rest
        tag = ""
        tm = _PLAN_TAG_RE.match(rest)
        if tm:
            tag = tm.group(1).strip()
            rest = tm.group(2)
        desc = rest.replace("(ACTIVE)", "").strip().lstrip(": ").strip()
        item = {"id": pid, "description": desc, "done": done,
                "active": is_active, "tag": tag}

        if include_meta:
            rationale = ""
            skill = ""
            j = i + 1
            while j < len(lines):
                sub = lines[j].strip()
                if sub.startswith("- rationale:"):
                    rationale = sub.split(":", 1)[1].strip()
                elif sub.startswith("- skill:"):
                    skill = sub.split(":", 1)[1].strip()
                elif sub.startswith("- ") and not sub.startswith("- ["):
                    # other sub-fields (hand-written notes) are skipped
                    # silently
                    pass
                else:
                    break
                j += 1
            item["rationale"] = rationale
            item["skill"] = skill

        out.append(item)
        i += 1
    return out


def get_plan_items(task_dir: str, include_meta: bool = False) -> list:
    """Canonical plan.md parser by task_dir. Thin wrapper over
    `parse_plan_text` so file-loading lives in one place."""
    if not os.path.exists(plan_path(task_dir)):
        return []
    with open(plan_path(task_dir), "r", encoding="utf-8") as f:
        text = f.read()
    return parse_plan_text(text, include_meta=include_meta)


def has_pending_items(task_dir: str) -> bool:
    """True iff plan.md has at least one unchecked item."""
    return any(not it["done"] for it in get_plan_items(task_dir))


def get_active_item(task_dir: str) -> Optional[dict]:
    """Return the (ACTIVE) pending item, or None. include_meta=True so the
    item carries its bound `skill` for EDIT guidance."""
    for it in get_plan_items(task_dir, include_meta=True):
        if it["active"] and not it["done"]:
            return {"id": it["id"], "description": it["description"],
                    "skill": it.get("skill", "")}
    return None


_DIAGNOSE_REQUIRED_SECTIONS = ("Root cause", "Fix directions", "What to avoid")

# DIAGNOSE has three host-visible sub-states. Keep this as the single branch
# used by guidance + Task/Bash/Stop hooks so they cannot drift.
DIAGNOSE_NEED_DIAGNOSIS = "NEED_DIAGNOSIS"
DIAGNOSE_READY = "DIAGNOSIS_READY"
DIAGNOSE_MANUAL_FALLBACK = "MANUAL_FALLBACK"


def validate_diagnose(task_dir: str, plan_version: int) -> tuple:
    """Validate the DIAGNOSE artifact for `plan_version`.

    Contract (in lockstep with `.claude/agents/ar-diagnosis.md` and the
    DIAGNOSE guidance in `phase_machine/guidance.py`):
      1. File `<task_dir>/.ar_state/diagnose_v<plan_version>.md` exists and
         is non-empty.
      2. Contains the magic marker `[AR DIAGNOSE COMPLETE marker_v<N>]`.
      3. Contains the three required sections: "Root cause",
         "Fix directions", "What to avoid". Match is substring (so either
         "## Root cause" or "Root cause:" passes — generous on heading style,
         strict on content presence).
    Returns (ok, reason). On failure, `reason` is a short user-facing
    string suitable for an `[AR Phase: DIAGNOSE retry]` message.
    """
    if plan_version is None or plan_version < 0:
        return False, f"invalid plan_version {plan_version!r}"

    path = diagnose_artifact_path(task_dir, plan_version)
    if not os.path.exists(path):
        return False, (
            f"missing artifact {os.path.basename(path)} — the ar-diagnosis "
            f"subagent must Write its report to that exact path")
    try:
        with open(path, "r", encoding="utf-8") as f:
            body = f.read()
    except OSError as e:
        return False, f"cannot read {os.path.basename(path)}: {e}"

    if not body.strip():
        return False, f"{os.path.basename(path)} is empty"

    marker = diagnose_marker(plan_version)
    if marker not in body:
        return False, (f"missing required marker line {marker!r} — the "
                       f"subagent must include this exact string in the "
                       f"artifact (recommended on its own line near the end)")

    missing_sections = [s for s in _DIAGNOSE_REQUIRED_SECTIONS if s not in body]
    if missing_sections:
        return False, (f"missing required section(s): "
                       f"{', '.join(missing_sections)}. Required headings: "
                       f"{', '.join(_DIAGNOSE_REQUIRED_SECTIONS)}.")

    return True, ""


class DiagnoseState(NamedTuple):
    """Snapshot of DIAGNOSE-phase state for the hook callers.

    `action` is the next legal high-level action. `attempts` is the
    per-plan_version Task-failure count. `artifact_reason` comes from
    `validate_diagnose` and explains the NEED_DIAGNOSIS state.
    """
    plan_version: int
    attempts: int
    action: str
    artifact_reason: str


def diagnose_state(task_dir: str,
                   progress: Optional[dict] = None) -> DiagnoseState:
    """Single read of all DIAGNOSE-relevant state needed by hooks.

    Pass `progress` if you've already loaded it; otherwise this loads it.
    The artifact validation is always run because the next legal action
    depends on both artifact validity and the attempt cap.
    """
    if progress is None:
        progress = load_progress(task_dir) or {}
    pv = progress.get("plan_version", 0) or 0
    if progress.get("diagnose_attempts_for_version") == pv:
        attempts = progress.get("diagnose_attempts", 0) or 0
    else:
        attempts = 0
    artifact_ok, artifact_reason = validate_diagnose(task_dir, pv)
    exhausted = attempts >= DIAGNOSE_ATTEMPTS_CAP
    if artifact_ok:
        action = DIAGNOSE_READY
    elif exhausted:
        action = DIAGNOSE_MANUAL_FALLBACK
    else:
        action = DIAGNOSE_NEED_DIAGNOSIS
    return DiagnoseState(
        plan_version=pv,
        attempts=attempts,
        action=action,
        artifact_reason=artifact_reason,
    )


def validate_plan(task_dir: str) -> tuple:
    """Validate plan.md structure. Returns (ok, error_message).

    Delegates item parsing to `get_plan_items` (canonical parser) and only
    enforces invariants here: ≥3 items, rationale length within bounds,
    exactly one ACTIVE pending item.
    """
    if not os.path.exists(plan_path(task_dir)):
        return False, "plan.md does not exist"

    items = get_plan_items(task_dir, include_meta=True)
    if len(items) < 3:
        return False, f"Plan must have ≥ 3 items, found {len(items)}"

    pending = [it for it in items if not it["done"]]
    for it in pending:
        rat = it.get("rationale", "")
        if len(rat) < 30:
            return False, f"Item {it['id']}: rationale too short ({len(rat)} chars, need ≥ 30)"
        if len(rat) > 400:
            return False, f"Item {it['id']}: rationale too long ({len(rat)} chars, max 400)"

        skill = it.get("skill", "")
        if not _PLAN_SKILL_RE.search(skill):
            return False, f"Item {it['id']}: skill must be `<skill-dir>/SKILL.md`"

    active_items = [it for it in pending if it["active"]]
    if len(active_items) != 1:
        return False, f"Must have exactly 1 (ACTIVE) pending item, found {len(active_items)}"

    return True, ""
