#!/usr/bin/env python3
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

"""
Create or replace plan.md from structured XML input.

Claude provides content, this script handles format. XML is preferred over JSON
because LLMs hallucinate fewer structural/escape errors in tag-delimited text.

Usage:
    python scripts/engine/create_plan.py <task_dir>
    python scripts/engine/create_plan.py <task_dir> '<items_xml>'

The single-arg form (no <items_xml>) is preferred — it reads from the
canonical path `<task_dir>/.ar_state/plan_items.xml`. The recommended
flow is:

  1. Write the XML to <task_dir>/.ar_state/plan_items.xml using the
     Write tool (Claude already has $AR_TASK_DIR; the path is fixed).
  2. Run `create_plan.py "<task_dir>"` — no second arg.

This eliminates a class of LLM drift where the path the model wrote to
and the path it later passed to create_plan.py disagreed (`@/tmp/x.xml`
vs the actual `.ar_state/plan_items.xml` write target, etc.). With one
hardcoded canonical path, the model can't transcribe it wrong because
the model never types it twice.

See `phase_machine.guidance._PLAN_XML_EXAMPLE` for the canonical schema
(with inline schema-reminder comments). That constant is the single
source of truth — keep this file's parsing rules in lockstep with it.
It is private to the `guidance` submodule and intentionally not
re-exported from `phase_machine` itself; import from the submodule
directly if you need to reach it.

Behavior:
  Every successful run REPLACES plan.md's `## Active Items` with the new
  XML items. Any pending pid from the previous plan that hadn't run yet is
  silently dropped (no fake DISCARD record, no Settled History row). pids
  remain monotonic — `next_pid` keeps advancing, dropped pids are not
  reused — so the audit chain via plan_version + history.jsonl is still
  unambiguous: a pid that exists only in plan_version N's plan.md and has
  no history.jsonl entry was abandoned at the N → N+1 transition.

  In practice this only affects DIAGNOSE (which fires mid-plan after 3
  consecutive failures); REPLAN by construction only fires when every
  item has already settled, so old_pending is empty there.

  If a past DISCARD/FAIL idea looks promising again, just re-propose it
  as a new item with a fresh pid. The desc text carries the audit story;
  pid reuse adds no information.

If <items_xml> begins with '@', the remainder is treated as a path and the
XML is read from that file. If <items_xml> is exactly '-', XML is read from
stdin. Prefer these over inline argv — on Windows, multi-line XML passed
through bash argv can be silently truncated by the shell / CreateProcess,
producing misleading "missing <desc>" style errors that look like schema
bugs but are actually IPC truncation.

Output: writes plan.md, prints JSON status.
"""
import json
import os
import re
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase_machine import PLAN_ITEMS_FILE
from task_handle import (
    open_task, Role,
    TaskOwnershipError, TaskConsistencyError, TaskCorrupted, TaskPhaseError,
)


# Words that indicate parameter tuning, used by `_check_diversity` to flag
# plans where every item is a parameter sweep.
_PARAM_WORDS = {
    "block", "tile", "tiling", "autotune", "config", "configs",
    "warps", "stages", "size", "tune", "adjust", "sweep",
    "parameter", "param", "group", "num",
}
_PARAM_PHRASES = {
    "block_size", "block_m", "block_n", "block_k", "block_size_m",
    "block_size_n", "block_size_k", "num_warps", "num_stages",
    "group_size", "group_size_m",
}
_STOPWORDS = {"the", "a", "to", "of", "in", "for", "and", "with", "from", "by",
              "on", "is", "it", "as", "at", "or", "an", "be", "was", "that"}
_SKILL_NAME_RE = re.compile(r"[\w.-]+[/\\]SKILL\.md\b", re.IGNORECASE)

# Tracks where the XML payload came from so error messages can steer the
# caller toward a robust input channel when argv looks suspicious. Set in
# main() before any _fail() call that depends on parsed content.
_SOURCE_MODE = "argv"  # one of: "argv", "file", "stdin", "default-file"

# Canonical path the slash command's PLAN/DIAGNOSE/REPLAN guidance steers
# the LLM to write to. Hardcoded so create_plan.py can default to it when
# called with just <task_dir> — no path transcription required.
_DEFAULT_XML_RELPATH = os.path.join(".ar_state", PLAN_ITEMS_FILE)


def _fail(msg: str):
    hint = ""
    if _SOURCE_MODE == "argv":
        # If this trips, the model almost certainly got here by inline-quoting
        # multi-line XML — which is exactly the Windows-argv failure mode. Say
        # so loudly so retries don't loop on "fix the schema".
        hint = (" [hint: payload was passed inline via argv. On Windows this "
                "is often truncated by the shell, producing errors that look "
                "like schema bugs. Write the XML to a file and pass "
                "'@<path>', or pipe it via stdin with '-' as the 2nd arg.]")
    print(json.dumps({"ok": False, "error": msg + hint}))
    sys.exit(1)


def _skill_required() -> bool:
    try:
        from utils.external_paths import skills_dir
        from utils.settings import skill_dsl
        subtree = os.path.join(os.path.abspath(skills_dir()), skill_dsl())
        return os.path.isdir(subtree)
    except Exception:
        return False


_ALLOWED_ITEM_TAGS = {"desc", "rationale", "skill"}


def _parse_items_xml(xml_str: str) -> list:
    """Parse <items><item>...</item>...</items> into a list of dicts.

    Recognized child elements under <item>: desc, rationale, skill.
    Unknown tags are rejected so typos surface loudly rather than silently
    dropping fields.
    """
    try:
        root = ET.fromstring(xml_str)
    except ET.ParseError as e:
        _fail(f"Invalid XML: {e}")
    if root.tag != "items":
        _fail(f"Root element must be <items>, got <{root.tag}>")
    if root.attrib:
        _fail(f"<items> must have no attributes, got {sorted(root.attrib)}")
    items = []
    for i, child in enumerate(list(root)):
        if child.tag != "item":
            _fail(f"Unexpected <{child.tag}> under <items> (only <item> allowed)")
        # <item> takes no attributes — pids are auto-assigned, and the
        # XML example's inline comments say so explicitly. Reject anything
        # the model invents (id="p1", pid="p1", priority="high", ...) so
        # the lesson lands instead of slipping through silently.
        if child.attrib:
            _fail(f"Item {i}: <item> must have no attributes, got "
                  f"{sorted(child.attrib)} — pids are auto-assigned, do "
                  f"not supply them")
        d = {}
        for sub in list(child):
            if sub.tag not in _ALLOWED_ITEM_TAGS:
                _fail(f"Item {i}: unknown element <{sub.tag}> "
                      f"(allowed: {sorted(_ALLOWED_ITEM_TAGS)})")
            if sub.tag in d:
                _fail(f"Item {i}: duplicate <{sub.tag}>")
            d[sub.tag] = (sub.text or "").strip()
        items.append(d)
    return items


def _validate_items(items, *, require_skill: bool = False):
    if not isinstance(items, list) or len(items) < 3:
        _fail(f"Need >= 3 items, got {len(items) if isinstance(items, list) else 'non-list'}")
    for i, item in enumerate(items):
        for field in ("desc", "rationale"):
            if field not in item:
                _fail(f"Item {i}: missing <{field}>")
            if not isinstance(item[field], str) or not item[field].strip():
                _fail(f"Item {i}: '{field}' must be a non-empty string")
        if require_skill and "skill" not in item:
            _fail(f"Item {i}: missing <skill>")

        # desc must be a short prose sentence, not a snake_case identifier —
        # the history table and plan table in the dashboard surface this
        # field directly, and "fuse_swiglu_epilogue" is unreadable next to
        # "Fuse the SwiGLU epilogue into the matmul kernel".
        desc = item["desc"].strip()
        item["desc"] = desc
        if len(desc) < 12:
            _fail(f"Item {i}: desc too short ({len(desc)} chars, need >= 12 — "
                  f"write a short sentence, not a label)")
        if " " not in desc:
            _fail(f"Item {i}: desc looks like an identifier ({desc!r}) — "
                  f"write a short sentence describing the change instead "
                  f"(e.g. 'Fuse SwiGLU into the matmul epilogue')")

        rat = item["rationale"].strip()
        if len(rat) < 30:
            _fail(f"Item {i}: rationale too short ({len(rat)} chars, need >= 30)")
        if len(rat) > 400:
            item["rationale"] = rat[:397] + "..."
            rat = item["rationale"]
        skill = (item.get("skill") or "").strip()
        if skill:
            if not _SKILL_NAME_RE.search(skill):
                _fail(
                    f"Item {i}: <skill> must be a concrete "
                    f"`<skill-dir>/SKILL.md` filename."
                )
            item["skill"] = skill


def _check_diversity(items):
    """Reject plans where all but one item are pure parameter tuning.

    Tokenizes <desc> directly. The earlier schema carried a separate
    <keywords> field for this signal; it was removed because every
    keyword token already appears in desc verbatim (you cannot describe
    a parameter sweep without using "block"/"tile"/"size"/etc.) and
    forcing the model to write keywords twice was pure friction.

    Detection rule per item: classify as parameter-only if its desc
    contains a known parameter phrase (block_size, num_warps, ...) OR
    if the only content tokens it has come from `_PARAM_WORDS`.
    """
    word_sets = []   # per-item: content tokens after stopword filter
    raw_descs = []   # per-item: lower+normalized for phrase matching
    for item in items:
        raw = item["desc"].lower().replace("-", "_")
        raw_descs.append(raw)
        words = set()
        for tok in raw.replace("_", " ").split():
            tok = tok.strip(".,;:()[]{}\"'")
            if tok and tok not in _STOPWORDS:
                words.add(tok)
        word_sets.append(words)

    param_only = 0
    for words, raw in zip(word_sets, raw_descs):
        has_param_phrase = any(p in raw for p in _PARAM_PHRASES)
        non_param = words - _PARAM_WORDS - {""}
        if (has_param_phrase or not non_param) and words:
            param_only += 1

    if param_only >= len(items) - 1:
        detected = _PARAM_WORDS & set().union(*word_sets) if word_sets else set()
        _fail(
            f"Diversity rejected: {param_only}/{len(items)} items are parameter tuning. "
            f"Bundle parameter sweeps into ONE item. Other items must be structurally "
            f"different (algorithmic changes, fusion, memory access patterns, data layout). "
            f"Param-only words detected in desc: {sorted(detected)}"
        )


def main():
    global _SOURCE_MODE
    if len(sys.argv) < 2:
        _fail("Usage: create_plan.py <task_dir> [<items_xml> | @<path> | -]")
    task_dir = sys.argv[1]
    arg = sys.argv[2] if len(sys.argv) > 2 else None

    if arg is None:
        # Single-arg form: read from the canonical path. This is the form
        # the slash command guidance now recommends — Claude Writes to the
        # fixed location, then runs create_plan with just <task_dir>, never
        # transcribing the path twice.
        _SOURCE_MODE = "default-file"
        path = os.path.join(task_dir, _DEFAULT_XML_RELPATH)
        if not os.path.exists(path):
            _fail(
                f"No XML payload provided and the default file does not "
                f"exist: {path!r}.\nWrite the <items>...</items> XML to "
                f"that path with the Write tool, then re-run "
                f"`create_plan.py \"{task_dir}\"` — or pass the XML "
                f"explicitly as a second arg (inline / '@<path>' / '-' "
                f"for stdin)."
            )
        try:
            with open(path, "r", encoding="utf-8") as f:
                xml_str = f.read()
        except OSError as e:
            _fail(f"Cannot read XML from default {path!r}: {e}")
    elif arg == "-":
        _SOURCE_MODE = "stdin"
        xml_str = sys.stdin.read()
    elif arg.startswith("@"):
        _SOURCE_MODE = "file"
        path = arg[1:]
        try:
            with open(path, "r", encoding="utf-8") as f:
                xml_str = f.read()
        except OSError as e:
            _fail(f"Cannot read XML from {path!r}: {e}")
    else:
        _SOURCE_MODE = "argv"
        xml_str = arg

    items = _parse_items_xml(xml_str)
    _validate_items(items, require_skill=_skill_required())
    _check_diversity(items)

    # Plan transaction is now journaled + committed atomically inside
    # Task.commit_plan (intent → plan.md → state.json → clear). A
    # SIGKILL between the body and the state write is healed by
    # replay_intent at the next open_task; no manual recovery
    # required. XML parsing / validation stays in this script — the
    # diversity rules + XML schema enforcement are LLM-prompt-shaped
    # and belong with the CLI entry, not the Task primitive.
    try:
        with open_task(task_dir, role=Role.AGENT) as t:
            result = t.commit_plan(items)
    except TaskConsistencyError as e:
        _fail(f"state inconsistent on entry: {e}")
        return
    except TaskOwnershipError as e:
        _fail(f"cannot run: {e}")
        return
    except (TaskCorrupted, TaskPhaseError) as e:
        _fail(f"cannot commit plan: {e}")
        return

    print(json.dumps({
        "ok":      True,
        "version": result["version"],
        "items":   result["item_ids"],
        "active":  result["active"],
        "dropped": result["dropped"],
        "path":    result["path"],
    }))


if __name__ == "__main__":
    main()
