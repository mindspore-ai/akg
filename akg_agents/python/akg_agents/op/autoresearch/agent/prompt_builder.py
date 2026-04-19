"""
Prompt assembly for AgentLoop.

This module owns two user-visible surfaces and nothing else:

  - :func:`build_system_prompt` — static system / context prompt.
  - :func:`build_initial_message` — the very first user message of a
    run, derived from task files plus the current ``FeedbackBuilder``
    state.

Skill retrieval (the keyword pipeline) lives in
``skill_pool.py::SkillPool``; ``build_initial_message`` reads from
``feedback.skill_pool`` for rendering and never touches the catalog
directly.
"""

import logging
import os

from .skill_rendering import render_ranked_skill_block, render_skills_markdown


logger = logging.getLogger(__name__)

_PROMPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")


def _load_template(name: str) -> str:
    path = os.path.join(_PROMPTS_DIR, name)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _parse_skill_frontmatter(text: str) -> tuple[dict, str]:
    """Split an SKILL.md into (front-matter dict, body).

    Accepts only top-level scalar keys and a single nested ``metadata:``
    block with scalar children — enough for the skill schema. Values
    are returned as strings (no type coercion). Malformed / missing
    front-matter returns ``({}, text)``.
    """
    if not text.startswith("---"):
        return {}, text
    lines = text.split("\n")
    if len(lines) < 2:
        return {}, text
    end = -1
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end = i
            break
    if end == -1:
        return {}, text
    meta: dict = {}
    in_metadata = False
    for raw in lines[1:end]:
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        if raw.startswith("metadata:"):
            in_metadata = True
            meta["metadata"] = {}
            continue
        if in_metadata and raw.startswith("  ") and ":" in raw:
            k, _, v = raw.strip().partition(":")
            meta["metadata"][k.strip()] = v.strip().strip('"')
            continue
        if raw.startswith(" "):
            continue
        in_metadata = False
        if ":" not in raw:
            continue
        k, _, v = raw.partition(":")
        meta[k.strip()] = v.strip().strip('"')
    body = "\n".join(lines[end + 1:]).lstrip("\n")
    return meta, body


def _scan_task_skills(task_dir: str) -> list[tuple[str, dict, str]]:
    """Enumerate task_dir/skills/<name>/SKILL.md, sorted by name.

    Returns tuples ``(name, frontmatter_dict, body_text)``. Silently
    skips entries that can't be read or parsed — a malformed skill is
    a local issue, not a reason to crash prompt building.
    """
    skills_root = os.path.join(task_dir, "skills")
    if not os.path.isdir(skills_root):
        return []
    out: list[tuple[str, dict, str]] = []
    for name in sorted(os.listdir(skills_root)):
        skill_md = os.path.join(skills_root, name, "SKILL.md")
        if not os.path.isfile(skill_md):
            continue
        try:
            with open(skill_md, "r", encoding="utf-8") as fh:
                raw = fh.read()
        except Exception:
            continue
        meta, body = _parse_skill_frontmatter(raw)
        out.append((name, meta, body))
    return out


def _build_fundamentals_section(task_dir: str, max_chars: int) -> str:
    """Render fundamental-category skills from task_dir/skills/ into a
    single "## DSL Fundamentals" block, greedy-packed within
    ``max_chars``. Returns ``""`` when nothing fits or none exist.

    Skills whose ``category`` front-matter field equals ``fundamental``
    are included in name order. When the budget overflows mid-skill
    the last entry is omitted (we never emit a half-rendered skill)
    and a terminal "... [N skipped: names]" note is appended so the
    drop is visible.
    """
    if max_chars <= 0:
        return ""
    parts: list[str] = ["## DSL Fundamentals (rules you must follow)"]
    used = len(parts[0]) + 1
    skipped: list[str] = []
    overrun = False
    for name, meta, body in _scan_task_skills(task_dir):
        if (meta.get("category") or "").strip() != "fundamental":
            continue
        header = f"\n### {name}\n"
        block = header + body.rstrip() + "\n"
        if overrun or used + len(block) > max_chars:
            overrun = True
            skipped.append(name)
            continue
        parts.append(block)
        used += len(block)
    if len(parts) == 1:
        return ""
    if skipped:
        note = f"\n... [{len(skipped)} fundamental(s) skipped: {', '.join(skipped)}]"
        if used + len(note) <= max_chars:
            parts.append(note)
    return "".join(parts)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------


def build_system_prompt(config, task_dir: str) -> tuple[str, str]:
    """Build the LLM-facing system prompt and the knowledge-only base."""
    cfg = config

    metadata_section = ""
    metadata = dict(cfg.metadata) if cfg.metadata else {}
    try:
        from ..framework.device import get_device_info

        dev_info = get_device_info()
        metadata["device"] = f"{dev_info['type'].upper()} ({dev_info['name']})"
    except Exception:
        pass
    if metadata:
        lines = ["## Task Metadata"]
        for k, v in metadata.items():
            lines.append(f"- {k}: {v}")
        metadata_section = "\n".join(lines) + "\n"

    context_entries = []
    if cfg.program_file:
        context_entries.append(("Agent Instructions", cfg.program_file))
    if cfg.ref_file:
        context_entries.append(("Reference Implementation", cfg.ref_file))
    for fpath in (cfg.context_files or []):
        context_entries.append(("Context", fpath))

    context_blocks = []
    per_file_limit = cfg.agent.system_context_file_truncate
    total_limit = cfg.agent.system_context_total_truncate
    total_chars = 0
    for label, fpath in context_entries:
        abs_path = os.path.join(task_dir, fpath)
        if not os.path.exists(abs_path):
            continue
        try:
            with open(abs_path, "r", encoding="utf-8") as fh:
                content = fh.read()
            if len(content) > per_file_limit:
                content = (
                    content[:per_file_limit]
                    + f"\n... [truncated at {per_file_limit} chars]"
                )
            block = f"## {label}: {fpath}\n```\n{content}\n```"
            separator_cost = 2 if context_blocks else 0
            block_cost = separator_cost + len(block)
            if total_chars + block_cost > total_limit:
                break
            context_blocks.append(block)
            total_chars += block_cost
        except Exception:
            pass

    context_files_section = "\n\n".join(context_blocks) if context_blocks else ""

    constraints_section = ""
    if cfg.constraints:
        lines = [
            "## Hard Constraints",
            "Results violating ANY constraint below are automatically DISCARDED.",
            "",
        ]
        for metric_name, (op_str, threshold) in cfg.constraints.items():
            lines.append(f"- {metric_name} {op_str} {threshold}")
        constraints_section = "\n".join(lines)

    fundamentals_section = _build_fundamentals_section(
        task_dir,
        getattr(cfg.agent, "system_fundamentals_max_chars", 20_000),
    )

    knowledge_prompt = _load_template("system_context.md").format_map(
        {
            "task_name": cfg.name,
            "task_description": cfg.description,
            "metadata_section": metadata_section,
            "editable_files_list": "\n".join(f"- {f}" for f in cfg.editable_files),
            "primary_metric": cfg.primary_metric,
            "metric_direction": (
                "lower is better" if cfg.lower_is_better else "higher is better"
            ),
            "constraints_section": constraints_section,
            "fundamentals_section": fundamentals_section,
            "context_files_section": context_files_section,
        }
    )

    # Inject forbidden edit patterns from config so the agent knows
    # upfront what guardrails will reject (instead of discovering it
    # via a wasted turn).
    banned_args_section = ""
    fp = getattr(cfg, "forbidden_patterns", None) or {}
    guardrail_lines = []
    content_pats = fp.get("content", [])
    if content_pats:
        guardrail_lines.append(
            "Rejected if file contains: "
            + ", ".join(f"`{p}`" for p in content_pats)
        )
    diff_pats = fp.get("diff", [])
    if diff_pats:
        guardrail_lines.append(
            "Rejected if ALL changed lines match: "
            + ", ".join(f"`{p}`" for p in diff_pats)
        )
    diff_any = fp.get("diff_any", [])
    if diff_any:
        guardrail_lines.append(
            "Rejected if ANY new line matches: "
            + ", ".join(f"`{p}`" for p in diff_any)
            + " (removing existing ones is allowed)"
        )
    if guardrail_lines:
        banned_args_section = (
            "\n- Edit guardrails: " + "; ".join(guardrail_lines) + "."
        )

    tool_protocol = _load_template("system_tool_protocol.md").format_map(
        {
            "editable_files": str(list(cfg.editable_files)),
            "primary_metric": cfg.primary_metric,
            "metric_direction": (
                "lower is better" if cfg.lower_is_better else "higher is better"
            ),
            "banned_args_section": banned_args_section,
            "min_items_per_plan": getattr(
                cfg.agent, "min_items_per_plan", 3,
            ),
        }
    )

    full_prompt = knowledge_prompt + "\n" + tool_protocol
    return full_prompt, knowledge_prompt


# ---------------------------------------------------------------------------
# Initial user message
# ---------------------------------------------------------------------------

# Two mutually-exclusive states the initial user message can frame the
# run in. The pre-turn-1 auto-bootstrap was deleted in the SkillPool
# refactor; turn 1 always starts at ``no_plan`` for fresh runs, and
# the only other entry point is ``--resume``. The fresh-vs-resumed
# distinction can't be read off ``feedback`` alone (a resumed session
# in ``replanning`` looks identical to a fresh run that just settled
# every item), so the caller passes ``session_restored`` explicitly.
_STATE_NO_PLAN = "no_plan"
_STATE_RESUMED = "resumed"


def _select_initial_state(phase: str, *, session_restored: bool) -> str:
    if session_restored and phase != "no_plan":
        return _STATE_RESUMED
    return _STATE_NO_PLAN


def _preselected_intro(state: str) -> str:
    if state == _STATE_RESUMED:
        return "Pre-selected skill pool (index). See Plan Status below for the restored plan."
    return (
        "Bindable skills (index). Add `keywords` to a plan item to request "
        "binding; on activation the controller auto-injects the SKILL.md "
        "into your context. Before the first edit on a bound item you must "
        "call `acknowledge_skill(...)` with `valuable_aspects` + "
        "`kernel_application` + `applicability` (apply / unbind). "
        "`search_skills(hint=...)` extends the pool."
    )


def _description_first_line(skill) -> str:
    description = getattr(skill, "description", "") or ""
    return description.strip().splitlines()[0] if description.strip() else ""


def _exploration_hint_lines(state: str, skills: list) -> list[str]:
    if state != _STATE_NO_PLAN:
        return []
    for skill in skills:
        if (getattr(skill, "category", "") or "") != "guide":
            continue
        first_line = _description_first_line(skill)
        if not first_line:
            continue
        return [
            "\n## Exploration Hint",
            "Top-ranked guide to consider before defaulting to parameter tuning:",
            f"- {getattr(skill, 'name', '')}: {first_line}",
        ]
    return []


def _start_guidance(state: str) -> list[str]:
    if state == _STATE_RESUMED:
        return [
            "\n## Start (Resumed session)",
            "Continue the plan. `replanning` → submit a fresh `update_plan(...)`; "
            "otherwise resume editing with the matching `plan_item_id`. Act, don't explain.",
        ]
    return [
        "\n## Start",
        "Submit `update_plan(items=[...])`. Each item needs `text` + `rationale` "
        "(name the bottleneck and the expected effect). Add `keywords` to request "
        "skill matching. Structural changes first. Act, don't explain.",
    ]


def build_initial_message(
    config,
    task_dir: str,
    runner,
    feedback,
    max_rounds: int,
    *,
    session_restored: bool = False,
) -> str:
    """Assemble the first user message sent to the agent."""
    phase = getattr(feedback, "phase", "no_plan") or "no_plan"
    state = _select_initial_state(phase, session_restored=session_restored)
    skill_pool = getattr(feedback, "skill_pool", None)

    lines = [
        f"# Optimization Task: {config.name}",
        f"Primary metric: {config.primary_metric} "
        f"({'lower is better' if config.lower_is_better else 'higher is better'})",
        f"Eval budget: {max_rounds} rounds",
        "",
    ]

    lines.append("## Current Editable Files")
    editable_contents = runner.get_editable_contents()
    for fname, content in editable_contents.items():
        if len(content) > config.agent.editable_file_truncate:
            content = (
                content[: config.agent.editable_file_truncate]
                + f"\n... [truncated at {config.agent.editable_file_truncate} chars]"
            )
        lines.append(f"### {fname}\n```\n{content}\n```")

    if runner.best_result:
        best = runner.best_result
        bv = best.metrics.get(config.primary_metric)
        lines.append(f"\n## Baseline: {config.primary_metric}={bv}")

    # DSL Documentation full-dump is gone: guide/example/case skills
    # live under task_dir/skills/<name>/SKILL.md and the agent pulls
    # them via read_file on demand. fundamentals are already embedded
    # in the system prompt (Layer 0). The pool index below tells the
    # agent what names are available and where to find them.

    if skill_pool and len(skill_pool) > 0:
        # Compact name + description + path index. Full content is
        # never inlined in the initial prompt; if the agent wants
        # SKILL.md body it calls read_file.
        # Post-abandon refactor: every registered skill stays listed
        # (previously-unbound skills are still selectable, just
        # demoted by the binding-tier logic). We only filter
        # fundamentals / reference here — fundamentals already appear
        # verbatim in the system prompt, so a second index entry per
        # fundamental would just be noise.
        _NON_BINDABLE = {"fundamental", "reference"}
        live_skills = [
            s for s in skill_pool
            if (getattr(s, "category", "") or "") not in _NON_BINDABLE
        ]
        if live_skills:
            idx = render_skills_markdown(
                live_skills, mode="index",
                max_chars=config.agent.skill_block_max_chars,
            )
            if idx:
                lines.extend([
                    "\n## Pre-selected Skills",
                    _preselected_intro(state),
                    "Read full content with "
                    "`read_file('skills/<name>/SKILL.md')` when you need it.",
                    "", idx,
                ])
            lines.extend(_exploration_hint_lines(state, live_skills))

    if state == _STATE_RESUMED:
        # ``format_status`` already emits its own ``## Plan Status``
        # header; just prepend a blank line so the section stands clear
        # of the skill block above.
        status = getattr(feedback, "format_status", lambda: "")()
        if status:
            lines.extend(["", status])

    lines.extend(_start_guidance(state))
    return "\n".join(lines)
