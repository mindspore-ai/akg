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

"""Phase-specific guidance — what the LLM should do next.

`get_guidance(task_dir)` is the only public API; it reads phase + progress
+ task config + plan, then returns the `[AR Phase: …]` message that hooks
inject into Claude's context after every state-changing event.

The XML schema example for plan creation (`_PLAN_XML_EXAMPLE`) and the
field-rules tail (`_PLAN_FIELD_RULES`) live here — they're prompt content
shared between PLAN, DIAGNOSE, and REPLAN guidance.
"""

# pylint: disable=broad-exception-caught,import-outside-toplevel,missing-function-docstring
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
    _PROJECT_ROOT,
)
from .validators import (
    get_active_item, diagnose_state,
    DIAGNOSE_READY, DIAGNOSE_MANUAL_FALLBACK,
)


def _render_signal_lines(signals: list) -> list:
    """One [kind][params] + optional hint line per signal, capped at two."""
    out: list = []
    for s in signals[:2]:
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
    return out


def _render_legacy_tail_lines(rec: dict) -> list:
    """Fallback rendering when failure_signals is empty: raw error +
    trimmed log tail (last few non-blank lines, where tracebacks and
    ACL runtime errors land)."""
    out: list = []
    err = (rec.get("error") or "").strip()
    if err and "verify failed (kernel broken)" not in err:
        out.append(f"      error: {err[:160]}")
    tail = (rec.get("raw_output_tail") or "").strip()
    if tail:
        keep = [l for l in tail.splitlines()[-4:] if l.strip()]
        if keep:
            out.append("      tail:")
            for line in keep:
                out.append(f"        {line[:160]}")
    return out


def _format_fail_record(rec: dict) -> str:
    """Compact per-FAIL block for the DIAGNOSE subagent prompt.

    Surfaces what kept-or-discard now persists per FAIL:
    `failure_signals` (kind + extracted params + hint) and a trimmed
    `raw_output_tail` when the pattern matchers found nothing. The
    earlier prompt only listed `R<n>: <description>`, so the subagent
    had to Read history.jsonl to learn what actually broke.
    """
    rnd = rec.get("round", "?")
    desc = (rec.get("description") or "")[:80]
    out = [f"  R{rnd}: {desc}"]

    sig = rec.get("failure_signals") or {}
    signals = sig.get("signals") or []
    python_error = sig.get("python_error")

    out.extend(_render_signal_lines(signals))
    if python_error:
        out.append(f"      python_error: {python_error[:200]}")
    if not signals and not python_error:
        out.extend(_render_legacy_tail_lines(rec))
    return "\n".join(out)


# Shared plan-item scaffolding shown in PLAN / DIAGNOSE / REPLAN guidance.
#
# Design notes:
#
# 1. THREE concrete items in the example, not one + "repeat" hint. Agents
#    consistently copy-as-shown — a single-item example produces single-
#    item submissions that fail the ">=3 items" check immediately.
#
# 2. Schema rules live in a plain bullet block ABOVE the example, not as
#    inline <!-- XML comments -->. Comments inside the structure get
#    treated as part of the shape and either leak into the agent's output
#    or train the agent to think the schema is "this with prose".
#
# 3. Wrong-vs-right pairs cover the most common drifts (attributes,
#    snake_case desc, all-parameter-tuning plans). Negative-only rules
#    underperform — pair each "don't" with a concrete "do" alternative.
#
# 4. Example items deliberately avoid every word in create_plan.py's
#    `_PARAM_WORDS` / `_PARAM_PHRASES` so the diversity check passes when
#    the agent generalises the shape to their own task. The three items
#    represent: algorithmic change / memory layout / data alignment —
#    structural changes, not parameter sweeps. Vocabulary stays DSL-neutral
#    (no warps / tiles / blocks / cube) so the same example reads naturally
#    for triton, ascendc, cuda_c, tilelang, etc.
#
# 5. XML stays the required format (tag-delimited beats JSON for LLMs —
#    no commas to forget, no brace balance).
_PLAN_XML_RULES = (
    "Plan item schema (each rule below maps to a create_plan.py check):\n"
    "  • Root <items> has NO attributes.\n"
    "  • At least 3 <item> children. NO attributes on <item> (pid is auto-assigned).\n"
    "  • Each <item> has EXACTLY two children: <desc> and <rationale>.\n"
    "    NO <id>, <pid>, <keywords>, <priority>, or any other tag.\n"
    "  • <desc>: short prose sentence, ≥12 chars, MUST contain spaces.\n"
    "  • <rationale>: 30-400 chars, explains WHY the change should help.\n"
    "  • At most ONE item may be pure parameter tuning (block size / num_warps /\n"
    "    num_stages / autotune sweep). The rest must be structural changes:\n"
    "    algorithmic / fusion / memory layout / data movement.\n"
    "\n"
    "Common drifts (these get rejected):\n"
    "  WRONG: <item id=\"p1\">…</item>          → <item>…</item>     (no attributes)\n"
    "  WRONG: <desc>fuse_swiglu_epilogue</desc> → <desc>Fuse the SwiGLU epilogue</desc>\n"
    "         (snake_case label fails the 'must contain spaces' check)\n"
    "  WRONG: 3 items all named 'tune block size to N' → mix in a fusion or\n"
    "         layout change (diversity check rejects param-only plans)\n"
    "  WRONG: <keywords>fuse,matmul</keywords>  → drop it, _check_diversity\n"
    "         tokenises <desc> directly, no separate keyword tag exists\n"
    "  Escape special chars in text: '&'→'&amp;', '<'→'&lt;', '>'→'&gt;'\n"
    "  (or wrap the field body in <![CDATA[...]]>)."
)
_PLAN_XML_EXAMPLE = (
    '<items>\n'
    '  <item>\n'
    '    <desc>Replace the explicit reduction loop with a tree-style '
    'accumulation</desc>\n'
    '    <rationale>The current per-element reduction serialises the '
    'dependency chain and prevents the hardware from issuing parallel adds; '
    'a tree pattern lets independent partial sums proceed concurrently.'
    '</rationale>\n'
    '  </item>\n'
    '  <item>\n'
    '    <desc>Re-lay out the input so the inner-most axis matches the '
    "hardware's preferred access stride</desc>\n"
    '    <rationale>The current layout forces strided gathers on every step; '
    'switching axes makes the inner access contiguous, which is what the '
    'memory subsystem is designed for.</rationale>\n'
    '  </item>\n'
    '  <item>\n'
    '    <desc>Pad the inner dimension to align with the hardware vector '
    'width</desc>\n'
    '    <rationale>The inner dim is one element short of a vector lane; '
    'padding lets the main loop drop its tail-handling branch and process '
    'every step at full width.</rationale>\n'
    '  </item>\n'
    '</items>'
)
# Kept as a named constant because callers (create_plan.py docstring,
# tests) reference the rules block by attribute. The rules text is the
# primary content; _PLAN_FIELD_RULES is now an alias to keep the public
# name stable.
_PLAN_FIELD_RULES = _PLAN_XML_RULES


def _create_plan_instruction(task_dir: str) -> str:
    """Common 'how to invoke create_plan.py' block used by PLAN, DIAGNOSE,
    and REPLAN guidance. Emits the canonical two-step flow:

      1. Write XML to the FIXED path .ar_state/plan_items.xml.
      2. Run create_plan.py with just <task_dir> — it reads from that path.

    The fixed path eliminates the LLM-drift class where the model wrote
    to one path and then passed a different `@<path>` to create_plan
    (most often a hallucinated /tmp/... or a typoed task subdir).
    """
    xml_path = state_path(task_dir, PLAN_ITEMS_FILE)
    return (
        f"To create the plan, do EXACTLY these two steps:\n"
        f"  1. Use the Write tool to write your <items>...</items> XML to:\n"
        f"       {xml_path}\n"
        "     (Path is fixed — do NOT invent a different path, do NOT use "
        "/tmp/, do NOT pass it as a CLI arg later. The Write tool is the "
        f"only thing that touches this path.)\n"
        f"  2. Run:\n"
        f"       python .autoresearch/scripts/engine/create_plan.py \"{task_dir}\"\n"
        f"     (No second argument. The script reads .ar_state/{PLAN_ITEMS_FILE} "
        "automatically. Adding `@/some/path` reintroduces the drift this "
        f"two-step form exists to prevent.)\n"
        f"\n"
        f"{_PLAN_XML_RULES}\n"
        f"\n"
        f"Canonical example (copy the SHAPE — three items, two children each;\n"
        f"replace the contents with items that fit YOUR task):\n"
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


def _skills_root() -> str:
    """Skills tree root. Override via ``AKG_AGENTS_AR_SKILLS_ROOT`` env var
    (relative to project root, or absolute). Default ``skills``.
    """
    return os.environ.get("AKG_AGENTS_AR_SKILLS_ROOT", "skills")


def _skill_dir_for_dsl(dsl) -> Optional[str]:
    """Resolve the skills subtree path for ``dsl`` (e.g.
    ``skills/triton-ascend``), or None if the subtree doesn't exist.

    DSL strings use underscores (``triton_ascend``); the on-disk tree
    uses dashes (``triton-ascend``). The translation lives here so the
    earlier "agent runs Glob, gets zero matches, silently skips skills"
    trap can't reappear. Returns the path verbatim in the form callers
    embed in prompts — Claude Code's Glob accepts both relative and
    absolute.
    """
    if not dsl:
        return None
    candidate = dsl.lower().replace("_", "-")
    root = _skills_root()
    abs_root = root if os.path.isabs(root) else os.path.join(_PROJECT_ROOT, root)
    if os.path.isdir(os.path.join(abs_root, candidate)):
        return f"{root.rstrip('/').rstrip(os.sep)}/{candidate}"
    return None


def _skills_hint(dsl) -> str:
    """Recommend reading DSL skills when authoring plan items.

    Used by PLAN and REPLAN (parent-voice — the parent agent reads skills
    directly and writes the plan). DIAGNOSE has its own inline skills
    section because the subagent's framing differs: it's diagnosing
    failures, not opening a plan, and the prompt wording reflects that.
    Returns "" when the DSL has no skills directory so callers can
    interpolate unconditionally.
    """
    skill_dir = _skill_dir_for_dsl(dsl)
    if not skill_dir:
        return ""
    return (
        f"\nDSL skills: Glob {skill_dir}/**/*.md, then Read 1-3 "
        "SKILL.md files whose frontmatter description / keywords match "
        "a candidate plan-item direction. Citing the SKILL id in the "
        "rationale is recommended for traceability but not enforced."
    )


def _multi_shape_plan_note(progress: Optional[dict],
                           task_dir: Optional[str] = None) -> str:
    """One-line note for the PLAN phase: say the op is multi-shape and point
    at the actual file(s) holding the shape spec. Deliberately does NOT
    list individual shapes — plan items are coarse-grained decisions, a
    30-line case dump in the planning prompt makes the agent over-engineer
    for shape generality at the expense of writing good plan items.

    NPUKernelBench-style refs read shapes from a sidecar JSON (the ref's
    `get_input_groups()` opens a same-directory `<basename>.json`). When
    that JSON is present in `task_dir`, this note names it explicitly —
    pointing at reference.py alone is not enough because the .py file is
    just a loader; the actual shape list lives in the JSON.

    Returns "" for single-shape ops (progress.num_cases <= 1) and when
    progress.json hasn't been written yet (pre-BASELINE).
    """
    if not progress:
        return ""
    n = progress.get("num_cases")
    if not isinstance(n, int) or n <= 1:
        return ""

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

    if sidecar_names:
        full_paths = [f"{task_dir}/{name}" for name in sidecar_names]
        if len(full_paths) == 1:
            where = f"shape list: {full_paths[0]}"
        else:
            where = "shape lists:\n  - " + "\n  - ".join(full_paths)
    else:
        where = (
            f"shape list: {task_dir}/reference.py "
            "(in the get_input_groups() body)"
        )

    return (
        f"Note: multi-shape op — reference exposes {n} input groups "
        f"via get_input_groups(). {where}\n"
        "Plan items must hold across all shapes; rely on shape-aware "
        "logic (read shape at runtime, dispatch on dtype/rank, adapt "
        "tile size) rather than constants pinned to one shape."
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
    """Render the per-shape failure detail used in DIAGNOSE / PLAN guidance.
    Pulls from the metrics block of a single FAIL history record
    (eval_client populates these from the verify subprocess's verify_json):
      - correctness_failed_cases: list of failing case indices
      - correctness_total_cases:  total case count at FAIL time
      - correctness_worst_case:   index with the largest max_abs_diff
      - correctness_worst_max_abs: that diff

    Resolves indices to describe_case() strings via progress.per_shape_descs
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
    (artifact_path is unused in fallback mode — the diagnosis context
    is history.jsonl + plan.md).
    """
    if fallback:
        header = "[AR Phase: DIAGNOSE — manual planning fallback]"
        source = "history.jsonl + plan.md (subagent route exhausted)"
    else:
        header = "[AR Phase: DIAGNOSE — diagnosis ready]"
        source = artifact_path or "(diagnosis artifact)"
    return (
        f"{header}\n"
        f"Create a NEW plan with >= 3 diverse items using {source}.\n"
        "Max 1 parameter-tuning item; the rest must be structural changes "
        f"(algorithmic / fusion / memory layout / data movement).\n\n"
        f"{_create_plan_instruction(task_dir)}"
        f"\nAfter create_plan.py validates, the hook advances phase to EDIT "
        "and emits the TodoWrite payload."
    )


def _g_init(task_dir: str, **_) -> str:
    return f"[AR Phase: INIT] Run: export AKG_AGENTS_AR_TASK_DIR=\"{task_dir}\""


def _g_baseline(task_dir: str, worker_flag: str, **_) -> str:
    return ("[AR Phase: BASELINE] Run: "
            "python .autoresearch/scripts/engine/baseline.py "
            f"\"{task_dir}\"{worker_flag}")


def _plan_seed_failed_section(task_dir: str, progress: dict,
                              editable: list) -> str:
    """SEED FAILED block, or "" when the seed kernel produced timings.
    Fires only for kernel-side baseline failures (kernel_fail);
    STUCK_BASELINE_OUTCOMES (infra_fail) never reach PLAN."""
    outcome = progress.get("baseline_outcome") if progress else None
    seed_missing = progress.get("seed_metric") is None
    if not (seed_missing or outcome == "kernel_fail"):
        return ""
    target_file = editable[0] if editable else "kernel.py"
    seed_reason = (
        "seed kernel produced no timing (compile/profile failed)"
        if seed_missing
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
    return (
        f"\n\nSEED FAILED: {seed_reason}.\n"
        "Plan items must focus on FIXING / REWRITING "
        f"{target_file} so the next round passes baseline.\n"
        f"Read {task_dir}/{target_file} to see what failed; "
        "baseline.py printed structured failure signals "
        "(UB overflow / aivec trap / OOM / correctness mismatch) "
        "above — use those as primary evidence. Each plan item "
        "is a structural change attempt; incremental fixes "
        "converge faster than rewrites from scratch."
        f"{failed_shapes_block}"
    )


def _g_plan(task_dir: str, progress: dict, dsl, editable: list,
            primary_metric: str, **_) -> str:
    metric_hint = ""
    if progress and progress.get("baseline_metric") is not None:
        metric_hint = (f" Baseline {primary_metric}: "
                       f"{progress.get('baseline_metric')}.")
    plan_note = _multi_shape_plan_note(progress, task_dir=task_dir)
    plan_note_section = f"\n\n{plan_note}" if plan_note else ""
    seed_failed_section = _plan_seed_failed_section(task_dir, progress or {},
                                                    editable)
    return ("[AR Phase: PLAN] "
            f"Read task.yaml, editable files ({editable}), and "
            f"reference.py.{_skills_hint(dsl)}{metric_hint}"
            f"{plan_note_section}{seed_failed_section}\n"
            f"\n"
            f"{_create_plan_instruction(task_dir)}"
            f"\n"
            "The script writes plan.md in the correct format. Hook "
            f"validates and advances to EDIT.\n"
            "(After validation the hook emits a TodoWrite payload — call "
            "it verbatim; do not pre-emptively craft one here.)")


def _g_edit(task_dir: str, active, editable: list, **_) -> str:
    desc = active["description"] if active else "(no active item)"
    item_id = active["id"] if active else "?"
    files_hint = f" (files: {', '.join(editable)})" if editable else ""
    return (f"[AR Phase: EDIT] ACTIVE item: **{item_id}** — {desc}\n"
            f"{files_hint}\n"
            f"CRITICAL: Implement ONLY {item_id}'s idea. Do NOT implement "
            f"other plan items.\n"
            f"The pipeline will settle {item_id} with this round's metric.\n"
            "Make your edit(s), then: "
            f"python .autoresearch/scripts/engine/pipeline.py \"{task_dir}\"\n"
            "(TodoWrite payloads are delivered by the hook after each "
            "settle / create_plan — call them verbatim when emitted; "
            "do not synthesize TodoWrite calls from this hint.)")


def _diagnose_history_blocks(task_dir: str) -> tuple:
    """Return (recent_summary, fail_details_block) from a single
    history.jsonl pass."""
    hpath = history_path(task_dir)
    if not os.path.exists(hpath):
        return "", ""
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
    recent_summary = ""
    for rec in all_recs[-5:]:
        _r = rec.get("round")
        _r = "?" if _r is None else _r
        recent_summary += (f"  R{_r}: {rec.get('decision','?')} — "
                           f"{rec.get('description','')[:60]}\n")
    last_3_fails = [
        r for r in all_recs
        if r.get("decision") == "FAIL" and r.get("round") is not None
    ][-3:]
    fail_details = ""
    if last_3_fails:
        fail_details = "\n".join(_format_fail_record(r)
                                 for r in last_3_fails) + "\n"
    return recent_summary, fail_details


def _diagnose_metric_line(progress: dict, primary_metric: str) -> str:
    """Compact metric snapshot for the subagent. Labels baseline_source
    honestly: "ref" anchors on PyTorch; "seed_fallback" anchors on the
    seed itself (no ref ever measured)."""
    if not progress:
        return ""
    seed = progress.get("seed_metric")
    base = progress.get("baseline_metric")
    best = progress.get("best_metric")
    if all(v is None for v in (seed, base, best)):
        return ""
    src = progress.get("baseline_source")
    if src == "ref":
        base_label = f"ref_baseline={base}"
    elif src == "seed_fallback":
        base_label = f"baseline={base} (seed fallback, no ref measured)"
    else:
        base_label = f"baseline={base}"
    return (f"\nMetrics ({primary_metric}): "
            f"seed={seed} | {base_label} | current_best={best}")


def _diagnose_skills_section(dsl) -> tuple:
    """Return (skills_block, scope_constraint, cite_clause). When the DSL
    has no curated skills tree, the whole skills block is dropped so the
    subagent doesn't get a Glob pattern that returns zero matches."""
    skill_dir = _skill_dir_for_dsl(dsl)
    if skill_dir:
        skills_block = (
            f"Read DSL skills (curated {dsl} knowledge — use it to "
            "ground fix directions in known-good patterns for this "
            f"hardware):\n"
            f"  - Glob {skill_dir}/**/*.md, then Read 1-3 "
            "SKILL.md files whose frontmatter description / keywords "
            f"match a candidate fix direction.\n"
            "  - Cite SKILL ids in the rationale of items you "
            f"propose.\n\n"
        )
        scope_constraint = (
            f"  - Glob / Grep ONLY under {skill_dir}/. The 4 "
            f"task files plus that skills subtree are the entire scope.\n"
        )
        return skills_block, scope_constraint, " Cite SKILL ids where relevant."
    scope_constraint = (
        "  - Do NOT Glob / Grep the wider codebase. The 4 task "
        "files are the entire scope (no curated skills tree exists "
        f"for dsl={dsl}).\n"
    )
    return "", scope_constraint, ""


def _diagnose_subagent_prompt(task_dir: str, *, dsl, backend, arch,
                              metric_line, plan_version, recent_summary,
                              fail_details, editable, editable_paths,
                              skills_block, scope_constraint, cite_clause,
                              artifact_path, marker) -> str:
    fail_details_block = (
        f"Last 3 FAILs (use these as the primary evidence):\n"
        f"{fail_details}\n"
        if fail_details
        else "Last 3 FAILs: (none yet — use history.jsonl if needed)\n\n"
    )
    editable_list = ", ".join(editable or ["kernel.py"])
    return (
        "Diagnose why the current optimization rounds are failing, then "
        f"Write a structured report to a fixed path.\n\n"
        f"Target: dsl={dsl} backend={backend} arch={arch}{metric_line}\n"
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
        f"{skills_block}"
        f"Hard constraints:\n"
        "  - Do NOT search git history (`git log` / `git show` / "
        "`git grep`) — per-round commits carry no keyword signal and "
        f"burn tool calls.\n"
        f"{scope_constraint}"
        f"  - Stop after at most 12 tool uses.\n"
        "  - Write tool may ONLY target the artifact path below. Do "
        f"NOT Write any of the editable files ({editable_list}), "
        "plan.md, or anywhere else.\n\n"
        "REQUIRED OUTPUT — your final action MUST be a Write call to "
        f"this exact path:\n"
        f"  {artifact_path}\n\n"
        f"The file body must contain ALL of:\n"
        "  - heading section 'Root cause' (one paragraph grounded in "
        f"the FAIL summary / history)\n"
        "  - heading section 'Fix directions' (≤3 STRUCTURALLY "
        "different approaches: algorithmic / fusion / memory layout "
        f"/ data movement; NOT parameter tuning.{cite_clause})\n"
        "  - heading section 'What to avoid' (≤3 patterns to NOT "
        f"repeat)\n"
        f"  - the magic marker line on its own line at the end:\n"
        f"      {marker}\n"
        "Total ≤ 300 words across the three sections. The host "
        "validates path + marker + the three section names after "
        "this Task call returns; missing any element will force a "
        "retry."
    )


def _g_diagnose(task_dir: str, progress: dict, config, dsl, editable: list,
                primary_metric: str, **_) -> str:
    ds = diagnose_state(task_dir, progress=progress) if progress else None
    plan_version = ds.plan_version if ds else 0
    attempts = ds.attempts if ds else 0
    artifact_path = diagnose_artifact_path(task_dir, plan_version)
    if ds and ds.action == DIAGNOSE_READY:
        return _diagnose_plan_next_step(task_dir,
                                        artifact_path=artifact_path)
    if ds and ds.action == DIAGNOSE_MANUAL_FALLBACK:
        return _diagnose_plan_next_step(task_dir, fallback=True)

    recent_summary, fail_details = _diagnose_history_blocks(task_dir)
    metric_line = _diagnose_metric_line(progress or {}, primary_metric)
    arch = (config.arch if config and config.arch else "<unknown>")
    backend = (config.backend if config and config.backend else "<unknown>")
    editable_paths = "\n".join(
        f"  - {task_dir}/{name}" for name in (editable or ["kernel.py"])
    )
    skills_block, scope_constraint, cite_clause = _diagnose_skills_section(dsl)
    marker = diagnose_marker(plan_version)

    subagent_prompt = _diagnose_subagent_prompt(
        task_dir, dsl=dsl, backend=backend, arch=arch,
        metric_line=metric_line, plan_version=plan_version,
        recent_summary=recent_summary, fail_details=fail_details,
        editable=editable, editable_paths=editable_paths,
        skills_block=skills_block, scope_constraint=scope_constraint,
        cite_clause=cite_clause,
        artifact_path=artifact_path, marker=marker,
    )
    retry_note = ""
    if attempts > 0:
        retry_note = (
            f"\nThis is DIAGNOSE attempt {attempts + 1}/"
            f"{DIAGNOSE_ATTEMPTS_CAP}. The previous artifact was "
            "missing or malformed — re-issue Task and ensure the "
            "subagent ends its work with a Write of the marker line."
        )
    return (f"[AR Phase: DIAGNOSE] consecutive_failures >= 3.\n"
            "Required action: call the "
            "Task tool with subagent_type='ar-diagnosis' and this "
            "EXACT prompt. Do not paraphrase. Do not add or remove "
            "constraints. Do not Edit, Write, or Bash before this "
            f"Task call.\n"
            f"---BEGIN SUBAGENT PROMPT---\n"
            f"{subagent_prompt}\n"
            f"---END SUBAGENT PROMPT---\n"
            "Artifact contract: the host gates plan creation on a valid "
            f"{os.path.basename(artifact_path)} (path + marker + 3 "
            "sections). Up to "
            f"{DIAGNOSE_ATTEMPTS_CAP} Task attempts are allowed; after "
            "that the gate is relaxed and you must write "
            "plan_items.xml directly (manual-planning fallback) "
            "before running create_plan.py — the DIAGNOSE phase "
            "still requires a new plan, just without subagent help."
            f"{retry_note}")


def _g_replan(task_dir: str, progress: dict, dsl, **_) -> str:
    remaining = "?"
    plan_ver = 0
    if progress:
        remaining = str(progress.get("max_rounds", 0)
                        - progress.get("eval_rounds", 0))
        plan_ver = progress.get("plan_version", 0)
    retry_hint = ""
    if plan_ver >= 2:
        retry_hint = (
            f"\nNote: plan_version is already {plan_ver}. Before "
            "inventing entirely new ideas, scan history.jsonl for "
            "DISCARD items whose metric was close to best (within "
            "~20%) — those ideas may compose differently now that "
            "the kernel's structural baseline has shifted. To revisit "
            "one, just include it as a new item with a fresh pid "
            "(reference the prior pid in <desc> for audit context)."
        )
    return ("[AR Phase: REPLAN] All items settled. Budget: "
            f"{remaining} rounds left. "
            "Read .ar_state/history.jsonl. Analyze what worked/"
            f"failed.{_skills_hint(dsl)}\n"
            f"\n"
            f"{_create_plan_instruction(task_dir)}"
            f"{retry_hint}")


def _g_finish(progress: dict, primary_metric: str, **_) -> str:
    best = progress.get("best_metric") if progress else "?"
    baseline = progress.get("baseline_metric") if progress else "?"
    src = progress.get("baseline_source") if progress else None
    if src == "seed_fallback":
        anchor = (f"seed-fallback baseline: {baseline} "
                  "(no PyTorch ref measured)")
    elif src == "ref":
        anchor = f"ref baseline: {baseline}"
    else:
        anchor = f"baseline: {baseline}"
    return (f"[AR Phase: FINISH] Done. Best {primary_metric}: {best} "
            f"({anchor}). Report auto-generated at "
            ".ar_state/report.md. Summarize for user; do not write any "
            "files.")


_PHASE_HANDLERS = {
    INIT:     _g_init,
    BASELINE: _g_baseline,
    PLAN:     _g_plan,
    EDIT:     _g_edit,
    DIAGNOSE: _g_diagnose,
    REPLAN:   _g_replan,
    FINISH:   _g_finish,
}


def get_guidance(task_dir: str) -> str:
    """Return a context-aware instruction for Claude based on current phase.

    Reads task.yaml to inject dynamic info (DSL, editable files, worker URL,
    skills path) so the .md slash command doesn't need to hardcode anything.
    """
    phase = read_phase(task_dir)
    handler = _PHASE_HANDLERS.get(phase)
    if handler is None:
        return f"[AR Phase: {phase}] Unknown phase."

    config = _load_config_safe(task_dir)
    worker_urls = config.worker_urls if config else []
    return handler(
        task_dir=task_dir,
        active=get_active_item(task_dir),
        progress=load_progress(task_dir),
        config=config,
        dsl=config.dsl if config else None,
        editable=config.editable_files if config else [],
        worker_flag=(f" --worker-url {worker_urls[0]}"
                     if worker_urls else ""),
        primary_metric=config.primary_metric if config else "score",
    )
