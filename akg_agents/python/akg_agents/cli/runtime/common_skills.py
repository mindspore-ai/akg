from __future__ import annotations

import logging
from pathlib import Path

from akg_agents import get_project_root
from akg_agents.core.skills import SkillLoader

logger = logging.getLogger(__name__)

COMMON_SKILLS_DIR = Path(get_project_root()) / "cli" / "resources" / "skills"


def build_common_skills_metadata() -> tuple[str, int]:
    skills = _load_skills(COMMON_SKILLS_DIR)
    if not skills:
        return "", 0
    return _format_skills_metadata(skills, COMMON_SKILLS_DIR), len(skills)


def _load_skills(skills_dir: Path) -> list[dict]:
    try:
        loader = SkillLoader(skills_dir=skills_dir)
        return loader.skills or []
    except Exception as exc:
        logger.warning("Failed to load common skills: %s", exc)
        return []


def _format_skills_metadata(skills: list[dict], skills_dir: Path) -> str:
    lines = _skills_header(skills_dir)
    for skill in skills:
        name = skill.get("name", "")
        desc = skill.get("description", "")
        path = skill.get("path") or str(skills_dir / name / "SKILL.md")
        lines.append(f"- **{name}**: {desc}")
        lines.append(f"  → `read(filePath=\"{path}\")`")
    return "\n".join(lines)


def _skills_header(skills_dir: Path) -> list[str]:
    return [
        "## Skills (on-demand)",
        "",
        f"Skills root: {skills_dir}",
        "Each skill lives in its own folder under the root (e.g., <root>/<skill-name>/).",
        "If a skill references extra files (examples/, scripts/, assets/), read them from that folder.",
        "",
        "When you need detailed guidance, load the full Skill content with the `read` tool:",
        "",
    ]
