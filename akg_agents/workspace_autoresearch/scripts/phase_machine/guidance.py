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

"""Phase-specific guidance - what the LLM should do next.

`get_guidance(task_dir)` is the only public API; it reads phase + progress
+ task config + plan, then returns the `[AR Phase: …]` message that hooks
inject into Claude's context after every state-changing event.

The XML schema example for plan creation (`_PLAN_XML_EXAMPLE`) and the
field-rules tail (`_PLAN_FIELD_RULES`) live here - they're prompt content
shared between PLAN, DIAGNOSE, and REPLAN guidance.
"""
import json
import os
import sys
from typing import Optional

from .state_store import (
    INIT, BASELINE, PLAN, EDIT,
    DIAGNOSE, REPLAN, FINISH,
    PLAN_ITEMS_FILE, DIAGNOSE_ATTEMPTS_CAP,
    diagnose_artifact_path, diagnose_marker,
    history_path, load_progress, read_phase, state_path,
)
from .validators import (
    get_active_item, diagnose_state,
    DIAGNOSE_READY, DIAGNOSE_MANUAL_FALLBACK,
)
# state_store (imported above) puts scripts/ on sys.path, so utils resolves.
from utils.external_paths import skills_dir  # noqa: E402


def _format_fail_record(rec: dict,
                        progress: Optional[dict] = None) -> str:
    """Compact per-FAIL block for the DIAGNOSE subagent prompt.

    Surfaces what kept-or-discard now persists per FAIL:
    `failure_signals` (kind + extracted params + hint) and a trimmed
    `raw_output_tail` when the pattern matchers found nothing. The
    earlier prompt only listed `R<n>: <description>`, so the subagent
    had to Read history.jsonl to learn what actually broke - which
    didn't help anyway because the structured signals weren't in
    history before this change.

    `progress` is passed through to `_failed_shapes_block` so per-shape
    correctness failures (eval_client's `correctness_failed_cases` etc.)
    get rendered with their stored per-shape descriptions inline.
    """
    rnd = rec.get("round", "?")
    desc = (rec.get("description") or "")[:80]
    out = [f"  R{rnd}: {desc}"]

    sig = rec.get("failure_signals") or {}
    signals = sig.get("signals") or []
    python_error = sig.get("python_error")

    for s in signals[:2]:  # at most two distinct kinds, the rest is noise
        kind = s.get("kind", "?")
        params = ", ".join(
            f"{k}={v}"
            for k, v in s.items()
            if k not in ("kind", "excerpt", "hint") and v is not None
        )
        marker = f"      → {kind}"
        if params:
            marker += f"  [{params}]"
        out.append(marker)
        if s.get("hint"):
            out.append(f"        hint: {s['hint']}")
    if python_error:
        out.append(f"      python_error: {python_error[:200]}")
    if not signals and not python_error:
        # FAIL row with no failure_signals (log matched no pattern).
        # Surface the raw error message and a trimmed tail if either is
        # available.
        err = (rec.get("error") or "").strip()
        if err and "verify failed (kernel broken)" not in err:
            out.append(f"      error: {err[:160]}")
        tail = (sig.get("tail_excerpt") or rec.get("raw_output_tail") or "").strip()
        if tail:
            # Prefer failure_extractor's filtered tail when present; fall
            # back to the raw tail for old history rows. In either case,
            # keep the final lines because Python tracebacks and ACL/CANN
            # runtime errors usually land there after noisy warnings.
            keep = [l for l in tail.splitlines()[-8:] if l.strip()]
            if keep:
                out.append("      tail:")
                for line in keep:
                    out.append(f"        {line[:160]}")

    # Per-shape correctness failure detail. Empty for compile/profile-style
    # failures (no per-shape signal) and for single-shape ops; see
    # _failed_shapes_block for the gating.
    fs = _failed_shapes_block(rec.get("metrics"), progress)
    if fs:
        out.append(fs)

    return "\n".join(out)


# Shared plan-item scaffolding shown in PLAN / DIAGNOSE / REPLAN guidance.
#
# Design notes:
#
# 1. THREE concrete items in the example, not one + "repeat" hint. Agents
#    consistently copy-as-shown - a single-item example produces single-
#    item submissions that fail the ">=3 items" check immediately.
#
# 2. Schema rules live in a plain bullet block ABOVE the example, not as
#    inline <!-- XML comments -->. Comments inside the structure get
#    treated as part of the shape and either leak into the agent's output
#    or train the agent to think the schema is "this with prose".
#
# 3. Wrong-vs-right pairs cover the most common drifts (attributes,
#    snake_case desc, all-parameter-tuning plans). Negative-only rules
#    underperform - pair each "don't" with a concrete "do" alternative.
#
# 4. Example items deliberately avoid every word in create_plan.py's
#    `_PARAM_WORDS` / `_PARAM_PHRASES` (block, tile, num_warps, etc.) so
#    the diversity check passes when the agent generalises the shape to
#    their own task. The three items represent: kernel fusion / memory
#    layout / data alignment - structural changes, not parameter sweeps.
#
# 5. XML stays the required format (tag-delimited beats JSON for LLMs -
#    no commas to forget, no brace balance).
_PLAN_XML_RULES = (
    "Plan item schema (each rule below maps to a create_plan.py check):\n"
    "  - Root <items> has NO attributes.\n"
    "  - At least 3 <item> children. NO attributes on <item> (pid is auto-assigned).\n"
    "  - Each <item> has EXACTLY three children: <desc>, <rationale>, <skill>.\n"
    "    NO <id>, <pid>, <keywords>, <priority>, or any other tag.\n"
    "  - <desc>: short prose sentence, >=12 chars, MUST contain spaces.\n"
    "  - <rationale>: 30-400 chars, explains WHY the change should help.\n"
    "  - <skill>: concrete skill filename, e.g. `ascendc-direct-invoke/SKILL.md`.\n"
    "  - At most ONE item may be pure parameter tuning (block size / num_warps /\n"
    "    num_stages / autotune sweep). The rest must be structural changes:\n"
    "    algorithmic / fusion / memory layout / data movement.\n"
    "\n"
    "Common drifts (these get rejected):\n"
    "  WRONG: <item id=\"p1\">...</item>          -> <item>...</item>     (no attributes)\n"
    "  WRONG: <desc>fuse_swiglu_epilogue</desc> -> <desc>Fuse the SwiGLU epilogue</desc>\n"
    "         (snake_case label fails the 'must contain spaces' check)\n"
    "  WRONG: 3 items all named 'tune block size to N' -> mix in a fusion or\n"
    "         layout change (diversity check rejects param-only plans)\n"
    "  WRONG: <keywords>fuse,matmul</keywords>  -> drop it, _check_diversity\n"
    "         tokenises <desc> directly, no separate keyword tag exists\n"
    "  Escape special chars in text: '&'->'&amp;', '<'->'&lt;', '>'->'&gt;'\n"
    "  (or wrap the field body in <![CDATA[...]]>)."
)
_PLAN_XML_EXAMPLE = (
    '<items>\n'
    '  <item>\n'
    '    <desc>Fuse the activation into the matmul epilogue to avoid a second '
    'kernel launch</desc>\n'
    '    <rationale>The separate activation kernel re-reads the matmul output '
    'from DRAM; folding it into the epilogue removes one round-trip and one '
    'launch overhead.</rationale>\n'
    '    <skill>ascendc-direct-invoke/SKILL.md</skill>\n'
    '  </item>\n'
    '  <item>\n'
    '    <desc>Transpose the input layout so the reduction axis is contiguous '
    'in memory</desc>\n'
    '    <rationale>Current reduction stride is 16380 bytes, which traps the '
    'vector core because it needs 256-byte-aligned access. Making the reduce '
    'axis contiguous gives aligned vectorised loads.</rationale>\n'
    '    <skill>ascendc-profiling-optimization/SKILL.md</skill>\n'
    '  </item>\n'
    '  <item>\n'
    '    <desc>Pad the inner dimension to a multiple of 64 elements</desc>\n'
    '    <rationale>The current inner dim is 4095, one short of the 4096 '
    'alignment the vector unit needs; padding to the next multiple lets the '
    'main loop drop its tail-handling branch entirely.</rationale>\n'
    '    <skill>ascendc-direct-invoke/SKILL.md</skill>\n'
    '  </item>\n'
    '</items>'
)
# One-line schema reminder for re-plans (REPLAN / DIAGNOSE / repeat PLAN).
# The agent already authored plan.md once and has it on disk as a live
# example, so the full _PLAN_XML_RULES + _PLAN_XML_EXAMPLE blocks are pure
# repetition — create_plan.py re-validates everything anyway. The freed
# prompt budget goes to _skills_hint(), which is where re-plans actually
# need help.
_PLAN_SCHEMA_ONELINE = (
    "Schema (same as your existing plan.md, create_plan.py re-checks it): "
    ">= 3 <item>, each with EXACTLY <desc> (prose, has spaces) / <rationale> "
    "(30-400 chars) / <skill> (a SKILL.md filename); no attributes on any "
    "tag; at most ONE pure parameter-tuning item, the rest structural."
)
# Single-item shape shown on re-plans — enough to anchor the tag nesting
# without the full 3-item block. The rationale deliberately cites a
# reference doc, modelling the skills-hint rule "name the deeper doc in
# <rationale>". No XML comments inside the structure (design note 2: the
# agent treats them as part of the shape and leaks them) — the "repeat"
# instruction lives in the prose around this block.
_PLAN_XML_EXAMPLE_ONE = (
    '<items>\n'
    '  <item>\n'
    '    <desc>Reuse the second calc buffer for the sigmoid output instead '
    'of a third allocation</desc>\n'
    '    <rationale>ascendc-ub-budget reduce-buffer note: freeing one '
    'TILE*4 fp32 slot lets BUFFER_NUM go 2->3 within the 192KB UB, '
    'enabling MTE2/V overlap.</rationale>\n'
    '    <skill>ascendc-ub-budget/SKILL.md</skill>\n'
    '  </item>\n'
    '</items>'
)
# Kept as a named constant because callers (create_plan.py docstring,
# tests) reference the rules block by attribute. The rules text is the
# primary content; _PLAN_FIELD_RULES is now an alias to keep the public
# name stable.
_PLAN_FIELD_RULES = _PLAN_XML_RULES


def _create_plan_instruction(task_dir: str, *, first_plan: bool = True) -> str:
    """'How to invoke create_plan.py' block for PLAN / DIAGNOSE / REPLAN.

    Emits the canonical two-step flow (write XML to the FIXED path, then
    run create_plan.py with just <task_dir>). The fixed path eliminates
    the LLM-drift class where the model wrote to one path then passed a
    different `@<path>` to create_plan.

    first_plan controls the schema teaching:
      - first_plan=True (the very first PLAN, plan_version 0): print the
        full _PLAN_XML_RULES + the worked 3-item _PLAN_XML_EXAMPLE so the
        agent learns the format once.
      - first_plan=False (every re-plan: REPLAN, DIAGNOSE, or a PLAN
        revisited with plan_version >= 1): the agent already wrote plan.md
        and has it on disk as a live example. Re-printing the rules +
        example is pure repetition (create_plan.py re-validates anyway),
        so collapse to a one-line schema reminder and spend the budget on
        the skills/reference guidance instead.
    """
    xml_path = state_path(task_dir, PLAN_ITEMS_FILE)
    invoke = (
        f"To create the plan, do EXACTLY these two steps:\n"
        f"  1. Use the Write tool to write your <items>...</items> XML to:\n"
        f"       {xml_path}\n"
        f"     (Path is fixed - do NOT invent a different path, do NOT use "
        f"/tmp/, do NOT pass it as a CLI arg later.)\n"
        f"  2. Run: python scripts/engine/create_plan.py \"{task_dir}\"\n"
        f"     (No second argument. The script reads .ar_state/"
        f"{PLAN_ITEMS_FILE} automatically.)\n"
    )
    if not first_plan:
        return (
            f"{invoke}\n"
            f"{_PLAN_SCHEMA_ONELINE}\n"
            f"\n"
            f"Shape of a single <item> (repeat per change; count + "
            f"constraints are in the schema line above):\n"
            f"{_PLAN_XML_EXAMPLE_ONE}\n"
        )
    return (
        f"{invoke}"
        f"\n"
        f"{_PLAN_XML_RULES}\n"
        f"\n"
        f"Canonical example (copy the SHAPE - three items, three children "
        f"each;\nreplace the contents with items that fit YOUR task):\n"
        f"{_PLAN_XML_EXAMPLE}\n"
    )


def _load_config_safe(task_dir: str):
    """Load TaskConfig, return None on any failure.

    task_config lives in scripts/ root (one level up from this package);
    insert the parent dir into sys.path so the import resolves no matter
    who's importing us.
    """
    try:
        _scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if _scripts_dir not in sys.path:
            sys.path.insert(0, _scripts_dir)
        from task_config import load_task_config
        return load_task_config(task_dir)
    except Exception:
        return None


# Skills tree root. The new layout is DSL-partitioned
# (`triton-ascend/`, `triton-cuda/`, `pypto/`, `cpp/`, `cuda-c/`,
# `tilelang-cuda/`, `tilelang-ascend/`, `ascendc/`,
# `ascendc-catlass/`) with
# per-topic SKILL.md files under each DSL's `fundamentals/`, `guides/`,
# `cases/`, `examples/`, and `evolved-*/` subdirs. The hint instructs
# the LLM to Glob inside the relevant DSL subtree. Returns "" when the
# skills root itself is missing so the prompt doesn't dispatch the
# agent to a dead Glob.


def _skills_subtree(dsl: str) -> str:
    root = os.path.abspath(skills_dir())
    # Skill dirs are kebab-case (e.g. `triton-ascend`), but skill_dsl may arrive
    # with underscores (operators copy the DSL adapter id `triton_ascend`).
    # Prefer whichever variant actually exists so the Glob hint points at a real
    # directory instead of returning zero matches.
    for cand in (dsl, dsl.replace("_", "-")):
        p = os.path.join(root, cand)
        if os.path.isdir(p):
            return p.replace(os.sep, "/")
    return os.path.join(root, dsl.replace("_", "-")).replace(os.sep, "/")


def _skills_catalog(subtree: str, *, limit: int = 20) -> str:
    """Comma-joined list of the actual skill names under the subtree (the
    leaf dir of each SKILL.md, e.g. `ascendc-ub-budget`).

    Used for the 'match your bottleneck' hint. Built from the tree on disk
    so it stays DSL-correct — the old hardcoded list (UB budget, reduce,
    aliasing...) was AscendC vocabulary that would mislead a triton / cpp
    plan. Names are kept verbatim (no prefix stripping) so each one is
    usable as-is in a `<skill>` tag (`<name>/SKILL.md`). '' when none."""
    names: set[str] = set()
    try:
        for dirpath, _dirnames, filenames in os.walk(subtree):
            if "SKILL.md" in filenames:
                names.add(os.path.basename(dirpath))
    except OSError:
        return ""
    ordered = sorted(names)
    if not ordered:
        return ""
    shown = ordered[:limit]
    tail = ", ..." if len(ordered) > limit else ""
    return ", ".join(shown) + tail


def _skills_hint() -> str:
    """Recommend reading the DSL-specific skill docs when authoring plan
    items.

    Used by PLAN and REPLAN (parent-voice — the parent agent reads
    skills directly and writes the plan). DIAGNOSE has its own inline
    skills section because the subagent's framing differs: it's
    diagnosing failures, not opening a plan, and the prompt wording
    reflects that.

    The Glob path is expanded with `settings.skill_dsl()`
    (config.yaml `defaults.skill_dsl`; kebab-case, e.g.
    `triton-ascend`) so the LLM gets a literal path to the resolved
    skills tree and not a "<dsl>" placeholder that the Glob tool would
    treat as a literal dir name and return zero matches for.
    """
    if not os.path.isdir(skills_dir()):
        return ""
    from utils.settings import skill_dsl as _skill_dsl
    dsl = _skill_dsl()
    subtree = _skills_subtree(dsl)
    catalog = _skills_catalog(subtree)
    # Bare leaf names; each sits under some category subdir, so locate via
    # the `**/` glob above, not a flat `{subtree}/<name>` guess.
    catalog_clause = (
        f" Names for `{dsl}` (each lives under some category subdir — reach it "
        f"via the `**/` glob above, don't guess the parent): {catalog}."
    ) if catalog else ""
    return (
        f"\nSkills & references (DO THIS BEFORE writing items - most weak "
        f"rounds skipped it):\n"
        f"  1. Glob {subtree}/**/SKILL.md and Read the 1-3 whose description "
        f"matches your bottleneck.{catalog_clause}\n"
        f"  2. SKILL.md files are INDEXES. Follow their `references/` / "
        f"`reference/` links and Read the specific doc for your case - the "
        f"concrete technique (the byte budget, the API signature, the "
        f"bound-type strategy) lives in the linked doc, not the index.\n"
        f"  3. Each <skill> tag = the SKILL.md you actually read; name the "
        f"deeper reference doc inside that item's <rationale> so EDIT knows "
        f"where the technique came from. An item whose rationale cites no "
        f"doc is usually a guess."
    )


def _active_skill_hint(skill: str) -> str:
    """EDIT hint to read THIS item's bound skill doc. `<skill>` is a bare
    `<dir>/SKILL.md` (no category parent), so anchor with `**/` to match
    whatever subdir it lives under. '' when no skill / no skills tree."""
    skill = (skill or "").strip()
    if not skill or not os.path.isdir(skills_dir()):
        return ""
    from utils.settings import skill_dsl as _skill_dsl
    subtree = _skills_subtree(_skill_dsl())
    return (
        f"\nBound skill: Glob `{subtree}/**/{skill}` and Read the match "
        f"before editing — it's the technique source for this item's "
        f"approach. SKILL.md is an index; follow its `references/` links to "
        f"the doc with the concrete signature / byte budget / write-up."
    )


def _target_dsl_safe() -> tuple[str, str, str]:
    """Return (KernelVerifier DSL, skills DSL dir, backend) for prompts."""
    try:
        from utils.settings import (
            skill_dsl as _skill_dsl,
            target_backend as _target_backend,
            target_dsl as _target_dsl,
        )
        return _target_dsl(), _skill_dsl(), _target_backend()
    except Exception:
        return "<configured-dsl>", "<configured-skill-dsl>", "<backend>"


def _editable_paths(task_dir: str, editable: list[str]) -> list[str]:
    return [f"{task_dir}/{name}" for name in editable]


def _editable_scope_text(editable: list[str]) -> str:
    if editable:
        return ", ".join(editable)
    return "task.yaml editable_files"


def _multi_shape_plan_note(progress: Optional[dict],
                           task_dir: Optional[str] = None) -> str:
    """One-line note for the PLAN phase: say the op is multi-shape and point
    at the actual file(s) holding the shape spec. Deliberately does NOT
    list individual shapes - plan items are coarse-grained decisions, a
    30-line case dump in the planning prompt makes the agent over-engineer
    for shape generality at the expense of writing good plan items.

    NPUKernelBench-style refs read shapes from a sidecar JSON (the ref's
    `get_input_groups()` opens a same-directory `<basename>.json`). When
    that JSON is present in `task_dir`, this note names it explicitly -
    pointing at reference.py alone is not enough because the .py file is
    just a loader; the actual shape list lives in the JSON.

    Returns "" for single-shape ops (progress.num_cases <= 1) and when
    progress isn't initialized yet (pre-BASELINE).
    """
    if not progress:
        return ""
    n = progress.get("num_cases")
    if not isinstance(n, int) or n <= 1:
        return ""

    # Locate sidecar JSONs that scaffold copied into task_dir at startup.
    # Skip task.yaml / *.lock / dotfiles and only surface JSONs at the
    # task_dir root (sidecars live there per scaffold.py:106-116).
    sidecar_names: list[str] = []
    if task_dir and os.path.isdir(task_dir):
        try:
            for fname in sorted(os.listdir(task_dir)):
                if not fname.endswith(".json"):
                    continue
                if fname.startswith("."):
                    continue
                if os.path.isfile(os.path.join(task_dir, fname)):
                    sidecar_names.append(fname)
        except OSError:
            pass

    # Fully qualify sidecar paths with task_dir so the agent can Read
    # them without first having to glob / guess where they live.
    if sidecar_names:
        full_paths = [f"{task_dir}/{name}" for name in sidecar_names]
        if len(full_paths) == 1:
            where = f"shape list: {full_paths[0]}"
        else:
            where = "shape lists:\n  - " + "\n  - ".join(full_paths)
    else:
        # Inline get_input_groups() - shapes are constructed in reference.py
        # itself, not a sidecar. Point at the loader function in the ref.
        where = (
            f"shape list: {task_dir}/reference.py "
            f"(in the get_input_groups() body)"
        )

    return (
        f"Note: multi-shape op - reference exposes {n} input groups "
        f"via get_input_groups(). {where}\n"
        f"Plan items must hold across all shapes; rely on shape-aware "
        f"logic (read shape at runtime, dispatch on dtype/rank, adapt "
        f"tile size) rather than constants pinned to one shape."
    )


def _last_failure_metrics(task_dir: str) -> Optional[dict]:
    """Return the metrics dict of the most recent FAIL/SEED record in
    history.jsonl whose `correctness` is False. Returns None when no
    failed record exists or history is missing.
    """
    hpath = history_path(task_dir)
    if not os.path.exists(hpath):
        return None
    last = None
    try:
        with open(hpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if rec.get("correctness") is False:
                    last = rec
    except OSError:
        return None
    if not last:
        return None
    metrics = last.get("metrics")
    return metrics if isinstance(metrics, dict) else None


def _failed_shapes_block(metrics: Optional[dict],
                         progress: Optional[dict],
                         *, max_listed: int = 5) -> str:
    """Render the per-shape failure detail used in DIAGNOSE's FAIL records.

    Pulls from the metrics block of a single FAIL history record (which
    eval_client populates from the verify subprocess's verify_json):
      - correctness_failed_cases: list of failing case indices
      - correctness_total_cases:  total case count at FAIL time
      - correctness_worst_case:   index with the largest max_abs_diff
      - correctness_worst_max_abs: that diff

    Resolves indices via progress.per_shape_descs
    so the agent sees both the index and the actual shape it fouled up.
    Returns "" when none of those fields are present (e.g. compile-error
    failure with no per-case detail, or single-shape op where the failure
    block is redundant).
    """
    if not metrics:
        return ""
    failed = metrics.get("correctness_failed_cases")
    total = metrics.get("correctness_total_cases")
    if not isinstance(failed, list) or not failed:
        return ""
    if not isinstance(total, int) or total <= 1:
        # Single-shape FAIL: the message that says "kernel broken"
        # already conveys this; an extra "1/1 shapes failed" block is noise.
        return ""

    descs = (progress or {}).get("per_shape_descs") or []
    parts = [f"      failed shapes: {len(failed)}/{total}"]
    for idx in failed[:max_listed]:
        if isinstance(idx, int) and 0 <= idx < len(descs):
            parts.append(f"        [{idx}] {descs[idx]}")
        else:
            parts.append(f"        [{idx}] (desc unavailable)")
    if len(failed) > max_listed:
        parts.append(f"        ... ({len(failed)} failures total)")
    worst_idx = metrics.get("correctness_worst_case")
    worst_max = metrics.get("correctness_worst_max_abs")
    if isinstance(worst_idx, int) and isinstance(worst_max, (int, float)):
        parts.append(
            f"      worst: case [{worst_idx}] max_abs={worst_max:.3e}"
        )
    return "\n".join(parts)


def _diagnose_plan_next_step(task_dir: str, *,
                             artifact_path: Optional[str] = None,
                             fallback: bool = False) -> str:
    """Guidance text for the post-DIAGNOSE create_plan step.

    Two callers in get_guidance: action == DIAGNOSE_READY passes the
    artifact path; action == DIAGNOSE_MANUAL_FALLBACK passes nothing
    (artifact_path is unused in fallback mode - the diagnosis context
    is history.jsonl + plan.md).
    """
    if fallback:
        header = "[AR Phase: DIAGNOSE - manual planning fallback]"
        source = "history.jsonl + plan.md (subagent route exhausted)"
    else:
        header = "[AR Phase: DIAGNOSE - diagnosis ready]"
        source = artifact_path or "(diagnosis artifact)"
    return (
        f"{header}\n"
        f"Create a NEW plan with >= 3 diverse items using {source}.\n"
        f"Max 1 parameter-tuning item; the rest must be structural changes "
        f"(algorithmic / fusion / memory layout / data movement).\n\n"
        f"{_create_plan_instruction(task_dir, first_plan=False)}"
        f"\nAfter create_plan.py validates, the hook advances phase to EDIT "
        f"and emits the plan-mirror payload."
    )


def _trace_analysis_block(task_dir: str, backend: str) -> str:
    """Tell the agent about existing --trace captures + how to read them;
    empty when none / non-Ascend. Used by PLAN/REPLAN/DIAGNOSE."""
    if backend != "ascend":
        return ""
    import glob
    dirs = glob.glob(os.path.join(
        task_dir, ".ar_state", "akg_verify", "**",
        "prof_generation_output_case_*"), recursive=True)
    if not dirs:
        return ""
    return (
        f"\n[trace] {len(dirs)} prior --trace capture(s) under akg_verify/<op>/"
        "Iteration*_Step<round>_verify/prof_generation_output_case_<N>/ "
        "(case_<N> = per-shape #N). Read kernel_details.csv / op_statistic.csv "
        "(per-op time) to find the hotspot op / cube-vs-vector bound before "
        "planning; trace_view.json (same dir) is the full timeline — open in "
        "perfetto / chrome://tracing.\n")


def get_guidance(task_dir: str) -> str:
    """Return a context-aware instruction for Claude based on current phase.

    Reads task.yaml to inject dynamic info (editable files, skills path)
    so the .md slash command doesn't need to hardcode anything.
    """
    phase = read_phase(task_dir)
    active = get_active_item(task_dir)
    progress = load_progress(task_dir)
    config = _load_config_safe(task_dir)

    # Extract config fields
    editable = config.editable_files if config else []
    primary_metric = config.primary_metric if config else "score"
    target_dsl, skill_name, backend = _target_dsl_safe()

    if phase == INIT:
        return f"[AR Phase: INIT] Run: export AR_TASK_DIR=\"{task_dir}\""

    if phase == BASELINE:
        return (f"[AR Phase: BASELINE] Run: "
                f"python scripts/engine/baseline.py \"{task_dir}\"")

    if phase == PLAN:
        metric_hint = ""
        if progress:
            baseline = progress.get("baseline_metric")
            if baseline is not None:
                metric_hint = f" Baseline {primary_metric}: {baseline}."
        # PLAN gets a short multi-shape note (count + pointer to the
        # sidecar JSON / ref). The detailed case list stays out-of-band in
        # the actual file - clogging the planning context with 30 lines of
        # `case i: tensor[...]` makes the agent over-engineer for shape
        # generality rather than write good plan items. Returns "" for
        # single-shape ops, so most tasks see no extra prompt content.
        plan_note = _multi_shape_plan_note(progress, task_dir=task_dir)
        plan_note_section = f"\n\n{plan_note}" if plan_note else ""

        # Seed-failure recovery: when the seed kernel failed BASELINE
        # (no timing, or wrong output), the first PLAN must focus on
        # fixing/rewriting the seed kernel — surface the per-shape
        # failure detail and steer the agent toward that goal.
        seed_failed_section = ""
        # editable_files is the per-DSL edit surface. Single-file DSLs
        # usually expose kernel.py; directory-backed DSLs expose wrapper
        # plus project source/build files.
        target_file = _editable_scope_text(editable)
        # SEED FAILED fires for kernel-side baseline failures (kernel_fail);
        # infra_fail never reaches PLAN: the ref-baseline gate parks
        # such tasks at BASELINE with no committed progress.
        outcome = progress.get("baseline_outcome") if progress else None
        seed_failed = bool(progress) and (
            progress.get("seed_metric") is None
            or outcome == "kernel_fail"
        )
        if seed_failed:
            seed_reason = (
                "seed kernel produced no timing (compile/profile failed)"
                if progress.get("seed_metric") is None
                else "seed kernel ran but failed correctness vs reference"
            )
            failed_shapes_block = ""
            if outcome == "kernel_fail":
                fail_metrics = _last_failure_metrics(task_dir)
                block = _failed_shapes_block(fail_metrics, progress)
                if block:
                    failed_shapes_block = (
                        f"\nThe BASELINE correctness failure had per-shape "
                        f"detail (round-0 SEED record):\n{block}\n"
                    )
            seed_failed_section = (
                f"\n\nSEED FAILED: {seed_reason}.\n"
                f"Plan items must focus on FIXING / REWRITING "
                f"the editable seed surface ({target_file}) so the next "
                f"round passes baseline.\n"
                f"Read these editable files to see what failed: "
                f"{', '.join(_editable_paths(task_dir, editable)) or task_dir}. "
                f"baseline.py printed structured failure signals "
                f"(UB overflow / aivec trap / OOM / correctness mismatch) "
                f"above — use those as primary evidence. Each plan item "
                f"is a structural change attempt; incremental fixes "
                f"converge faster than rewrites from scratch."
                f"{failed_shapes_block}"
            )

        # Full schema + worked example only on the very first plan
        # (plan_version 0). A PLAN re-entered after a plan already exists
        # gets the one-line schema reminder — same logic as REPLAN.
        first_plan = (progress or {}).get("plan_version", 0) == 0
        return (f"[AR Phase: PLAN] "
                f"Target DSL: {target_dsl} (skills: {skill_name}, backend: {backend}). "
                f"Read task.yaml, reference.py, and editable files "
                f"({_editable_scope_text(editable)}). Directory-backed DSLs "
                f"may expose multiple editable project files; plan only "
                f"changes inside that editable surface.{_skills_hint()}"
                f"{metric_hint}{plan_note_section}{seed_failed_section}"
                f"{_trace_analysis_block(task_dir, backend)}\n"
                f"\n"
                f"{_create_plan_instruction(task_dir, first_plan=first_plan)}"
                f"\n"
                f"The script writes plan.md in the correct format. Hook validates and advances to EDIT.\n"
                f"(After validation the hook emits the plan-mirror payload; "
                f"mirror it verbatim, do not pre-emptively craft one here.)")

    if phase == EDIT:
        desc = active["description"] if active else "(no active item)"
        item_id = active["id"] if active else "?"
        skill_hint = _active_skill_hint(active.get("skill", "")) if active else ""
        files_hint = f" (files: {', '.join(editable)})" if editable else ""
        # --trace hint only on Ascend (msprof); irrelevant noise on CUDA.
        trace_hint = (
            f"PROFILE (only when evidence is needed): run `python "
            f"scripts/engine/pipeline.py \"{task_dir}\" --trace` to keep "
            f"trace_view.json plus kernel_details / op_statistic CSVs. Later "
            f"planning or diagnosis guidance automatically surfaces them; "
            f"read the CSVs first for hotspot / bound analysis.\n"
            if backend == "ascend" else "")
        # No multi-shape note here. EDIT prompts already carry the active
        # plan item's description and the agent has the file list; piling
        # the full shape spec on top every EDIT phase (10-20+ times per
        # task) bloats context and biases edits toward over-generalisation.
        # If a multi-shape regression surfaces, it lands as a FAIL and the
        # next DIAGNOSE / PLAN retry will surface the offending shapes via
        # _failed_shapes_block.
        return (f"[AR Phase: EDIT] ACTIVE item: **{item_id}** - {desc}\n"
                f"{files_hint}{skill_hint}\n"
                f"CRITICAL: Implement ONLY {item_id}'s idea. Do NOT implement other plan items.\n"
                f"Target DSL: {target_dsl}; edit only task.yaml editable_files. "
                f"For directory-backed DSLs, this may include wrapper and "
                f"project source/build files, not just kernel.py.\n"
                f"The pipeline will settle {item_id} with this round's metric.\n"
                f"Make your edit(s), then: python scripts/engine/pipeline.py \"{task_dir}\"\n"
                f"{trace_hint}"
                f"(The hook delivers the plan-mirror payload after each "
                f"settle / create_plan; mirror it verbatim, do not "
                f"synthesize one from this hint.)")

    if phase == DIAGNOSE:
        # Pre-bake the recent-rounds summary INTO the subagent prompt so the
        # subagent has it without spending a tool call re-reading
        # history.jsonl. The full file stays in the read list for deeper
        # digs (full traces / older rounds).
        # Single pass through history.jsonl: build both the high-level
        # rhythm (last 5 records, any decision) and the FAIL detail block
        # (last 3 FAILs with structured failure_signals, courtesy of
        # keep_or_discard which now persists them per FAIL).
        hpath = history_path(task_dir)
        recent_summary = ""
        fail_details = ""
        if os.path.exists(hpath):
            all_recs = []
            with open(hpath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        all_recs.append(json.loads(line))
                    except Exception:
                        continue
            for rec in all_recs[-5:]:
                _r = rec.get("round")
                _r = "?" if _r is None else _r
                recent_summary += f"  R{_r}: {rec.get('decision','?')} - {rec.get('description','')[:60]}\n"
            last_3_fails = [
                r for r in all_recs
                if r.get("decision") == "FAIL" and r.get("round") is not None
            ][-3:]
            if last_3_fails:
                fail_details = "\n".join(
                    _format_fail_record(r, progress) for r in last_3_fails
                ) + "\n"

        # Compact metric snapshot - saves the subagent from reading
        # history.jsonl just to answer "how big a delta do we need?".
        metric_line = ""
        if progress:
            seed = progress.get("seed_metric")
            base = progress.get("baseline_metric")
            best = progress.get("best_metric")
            if any(v is not None for v in (seed, base, best)):
                metric_line = (
                    f"\nMetrics ({primary_metric}): "
                    f"seed={seed} | ref_baseline={base} | current_best={best}"
                )
        # Single source of plan_version + per-pv attempt counter (also
        # validates the artifact, but the result is unused here - accepting
        # the small extra read so all callers go through the same helper).
        ds = diagnose_state(task_dir, progress=progress) if progress else None
        plan_version = ds.plan_version if ds else 0
        attempts = ds.attempts if ds else 0
        artifact_path = diagnose_artifact_path(task_dir, plan_version)
        if ds and ds.action == DIAGNOSE_READY:
            return _diagnose_plan_next_step(
                task_dir, artifact_path=artifact_path)
        if ds and ds.action == DIAGNOSE_MANUAL_FALLBACK:
            return _diagnose_plan_next_step(task_dir, fallback=True)

        arch = (config.arch if config and config.arch else "<unknown>")
        editable_list = editable or ["<task editable file>"]
        editable_paths = "\n".join(
            f"  - {task_dir}/{name}" for name in editable_list
        )
        # Skills section is conditional on the references dir being present.
        # Without the dir-existence check we'd hand the agent a Glob
        # pattern that returns zero matches, and they'd silently skip
        # the skill-reading step.
        skills_present = os.path.isdir(skills_dir())
        # Resolve to the literal kebab-case DSL dir name so the Glob
        # patterns below target an actual subtree (`triton-ascend/`
        # etc.) rather than the literal `<dsl>` token, which the Glob
        # tool would search for and return zero matches.
        dsl = skill_name
        if skills_present:
            subtree = _skills_subtree(dsl)
            catalog = _skills_catalog(subtree)
            catalog_clause = f" Available: {catalog}." if catalog else ""
            skills_block = (
                f"Read curated DSL-specific skill references for "
                f"`{dsl}` (use them to ground fix directions in "
                f"known-good patterns for this hardware):\n"
                f"  - Glob {subtree}/**/SKILL.md and Read what "
                f"matches the fix direction.{catalog_clause}\n"
                f"  - Cite filename in the rationale of items you "
                f"propose.\n\n"
            )
            scope_constraint = (
                f"  - Glob / Grep ONLY under {subtree}/. "
                f"The listed task files plus that skills subtree are the "
                f"entire scope.\n"
            )
            cite_clause = " Cite reference filenames where relevant."
        else:
            skills_block = ""
            scope_constraint = (
                "  - Do NOT Glob / Grep the wider codebase. The listed "
                "task files are the entire scope.\n"
            )
            cite_clause = ""

        # Artifact contract - the host validates these literals after the
        # Task call returns. See validators.validate_diagnose.
        marker = diagnose_marker(plan_version)

        # Pre-baked subagent prompt. Parent passes this verbatim to the Agent
        # tool so the subagent doesn't improvise (an earlier open-ended brief
        # sent it grepping git log for 100+ tool calls before timing out).
        #
        # Two-section history view: the recent-5 block gives rhythm
        # (KEEP/DISCARD/FAIL pattern), the FAIL-detail block gives the
        # structured signals (UB overflow / aivec trap / OOM / correctness
        # mismatch + hint) that root-cause analysis needs. Before this,
        # the subagent only saw round + 60-char description per FAIL and
        # had to guess which Ascend constraint was violated.
        fail_details_block = (
            f"Last 3 FAILs (use these as the primary evidence):\n"
            f"{fail_details}\n"
            if fail_details
            else "Last 3 FAILs: (none yet - use history.jsonl if needed)\n\n"
        )
        subagent_prompt = (
            f"Diagnose why the current optimization rounds are failing, then "
            f"Write a structured report to a fixed path.\n\n"
            f"Target: dsl={target_dsl}, skill_dsl={skill_name}, "
            f"backend={backend}, arch={arch}{metric_line}\n"
            f"plan_version={plan_version}\n\n"
            f"Recent rounds (last 5 from history.jsonl):\n"
            f"{recent_summary or '  (none settled yet)'}\n"
            f"{fail_details_block}"
            f"Read these task files for context:\n"
            f"  - {task_dir}/reference.py\n"
            f"{editable_paths}\n"
            f"  - {task_dir}/.ar_state/plan.md\n"
            f"  - {task_dir}/.ar_state/history.jsonl (focus on the last "
            f"~10 rounds; older entries are usually stale)\n\n"
            f"{_trace_analysis_block(task_dir, backend)}"
            f"{skills_block}"
            f"Hard constraints:\n"
            f"  - Do NOT search git history (`git log` / `git show` / "
            f"`git grep`) - per-round commits carry no keyword signal and "
            f"burn tool calls.\n"
            f"{scope_constraint}"
            f"  - Stop after at most 12 tool uses.\n"
            f"  - Write tool may ONLY target the artifact path below. Do "
            f"NOT Write editable source files, reference.py, plan.md, or "
            f"anywhere else.\n\n"
            f"REQUIRED OUTPUT - your final action MUST be a Write call to "
            f"this exact path:\n"
            f"  {artifact_path}\n\n"
            f"The file body must contain ALL of:\n"
            f"  - heading section 'Root cause' (one paragraph grounded in "
            f"the FAIL summary / history)\n"
            f"  - heading section 'Fix directions' (≤3 STRUCTURALLY "
            f"different approaches: algorithmic / fusion / memory layout "
            f"/ data movement; NOT parameter tuning.{cite_clause})\n"
            f"  - heading section 'What to avoid' (≤3 patterns to NOT "
            f"repeat)\n"
            f"  - the magic marker line on its own line at the end:\n"
            f"      {marker}\n"
            f"Total ≤ 300 words across the three sections. The host "
            f"validates path + marker + the three section names after "
            f"this Task call returns; missing any element will force a "
            f"retry."
        )
        retry_note = ""
        if attempts > 0:
            retry_note = (
                f"\nThis is DIAGNOSE attempt {attempts + 1}/"
                f"{DIAGNOSE_ATTEMPTS_CAP}. The previous artifact was "
                f"missing or malformed - re-issue Task and ensure the "
                f"subagent ends its work with a Write of the marker line."
            )
        return (f"[AR Phase: DIAGNOSE] consecutive_failures >= 3.\n"
                f"Required action: call the "
                f"Task tool with subagent_type='ar-diagnosis' and this "
                f"EXACT prompt. Do not paraphrase. Do not add or remove "
                f"constraints. Do not Edit, Write, or Bash before this "
                f"Task call.\n"
                f"---BEGIN SUBAGENT PROMPT---\n"
                f"{subagent_prompt}\n"
                f"---END SUBAGENT PROMPT---\n"
                f"Artifact contract: the host gates plan creation on a valid "
                f"{os.path.basename(artifact_path)} (path + marker + 3 "
                f"sections). Up to "
                f"{DIAGNOSE_ATTEMPTS_CAP} Task attempts are allowed; after "
                f"that the gate is relaxed and you must write "
                f"plan_items.xml directly (manual-planning fallback) "
                f"before running create_plan.py - the DIAGNOSE phase "
                f"still requires a new plan, just without subagent help.{retry_note}")

    if phase == REPLAN:
        remaining = "?"
        plan_ver = 0
        if progress:
            remaining = str(progress.get("max_rounds", 0) - progress.get("eval_rounds", 0))
            plan_ver = progress.get("plan_version", 0)
        retry_hint = ""
        if plan_ver >= 2:
            retry_hint = (
                f"\nNote: plan_version is already {plan_ver}. Before "
                "inventing entirely new ideas, scan history.jsonl for "
                "DISCARD items whose metric was close to best (within "
                "~20%) - those ideas may compose differently now that "
                "the kernel's structural baseline has shifted. To revisit "
                "one, just include it as a new item with a fresh pid "
                "(reference the prior pid in <desc> for audit context)."
            )
        return (f"[AR Phase: REPLAN] All items settled. Budget: {remaining} rounds left. "
                f"Read .ar_state/history.jsonl. Analyze what worked/failed.{_skills_hint()}"
                f"{_trace_analysis_block(task_dir, backend)}\n"
                f"\n"
                f"{_create_plan_instruction(task_dir, first_plan=False)}"
                f"{retry_hint}")

    if phase == FINISH:
        best = progress.get("best_metric") if progress else "?"
        baseline = progress.get("baseline_metric") if progress else "?"
        return (f"[AR Phase: FINISH] Done. Best {primary_metric}: {best} "
                f"(baseline: {baseline}). Report auto-generated at "
                f".ar_state/report.md. Summarize for user; do not write any "
                f"files.")

    return f"[AR Phase: {phase}] Unknown phase."
