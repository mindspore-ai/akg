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
Skill 自进化系统 - organize 模式工具函数

提供 evolved skill 扫描、摘要提取、聚类结果解析、归档等功能。
"""

import json
import logging
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from akg_agents.core_v2.skill.loader import SkillLoader
from akg_agents.core_v2.skill.metadata import SkillMetadata

logger = logging.getLogger(__name__)

SKIP_DIRS_EXACT = {".archive"}
MAX_CLUSTER_SIZE = 5


def scan_evolved_skills(evolved_dir: str) -> List[SkillMetadata]:
    """扫描目录下所有 SKILL.md（排除 .archive/）"""
    evolved_path = Path(evolved_dir)
    if not evolved_path.exists():
        return []

    skills: List[SkillMetadata] = []
    loader = SkillLoader()

    for skill_md in evolved_path.rglob("SKILL.md"):
        rel = skill_md.relative_to(evolved_path)
        if any(part in SKIP_DIRS_EXACT for part in rel.parts):
            continue
        loaded = loader.load_single(skill_md)
        if loaded:
            skills.append(loaded)

    logger.info(f"[organize] 扫描到 {len(skills)} 个 evolved skill")
    return skills


def build_summaries(skills: List[SkillMetadata]) -> List[Dict[str, str]]:
    """从 SkillMetadata 列表中提取 name + description 摘要"""
    return [
        {"name": s.name, "description": s.description or "(无描述)"}
        for s in skills
    ]


def parse_classify_output(llm_output: str) -> List[Dict]:
    """解析 LLM 聚类输出，返回 clusters 列表

    每个 cluster: {"reason": str, "skills": [name, ...]}
    """
    text = llm_output.strip()
    if text.startswith("```"):
        text = re.sub(r"^```\w*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        json_match = re.search(r'\{.*"clusters".*\}', text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
            except json.JSONDecodeError:
                return []
        else:
            return []

    clusters = data.get("clusters", [])
    valid = []
    for c in clusters:
        if "skills" in c and isinstance(c["skills"], list):
            valid.append({
                "reason": c.get("reason", ""),
                "skills": c["skills"],
            })
    return valid


def archive_skills(
    skills: List[SkillMetadata],
    evolved_dir: str,
) -> str:
    """将指定的 skill 目录移入 .archive/{timestamp}/"""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_base = os.path.join(evolved_dir, ".archive", ts)
    os.makedirs(archive_base, exist_ok=True)

    for skill in skills:
        if not skill.skill_path:
            continue
        skill_dir = skill.skill_path.parent
        dest = os.path.join(archive_base, skill_dir.name)
        try:
            shutil.move(str(skill_dir), dest)
            logger.info(f"[organize] 归档: {skill_dir.name} -> .archive/{ts}/")
        except OSError as e:
            logger.warning(f"[organize] 归档失败 {skill_dir.name}: {e}")

    return archive_base


def write_merged_skill(
    name: str,
    description: str,
    body: str,
    dsl: str,
    backend: str,
    evolved_dir: str,
) -> str:
    """写入合并后的 SKILL.md，复用 SkillWriter 构建 frontmatter"""
    from akg_agents.op.tools.skill_evolution.common import SkillWriter

    writer = SkillWriter()
    compressed = {
        "source": "merged",
        "op_name": "",
        "dsl": dsl,
        "backend": backend,
    }

    return writer.write(
        skill_name=name,
        description=description,
        markdown_body=body.strip(),
        compressed=compressed,
        output_dir=evolved_dir,
    )


def split_large_cluster(
    skill_names: List[str],
    max_size: int = MAX_CLUSTER_SIZE,
) -> List[List[str]]:
    """将大簇拆分为子批次，每批最多 max_size 个"""
    if len(skill_names) <= max_size:
        return [skill_names]
    return [
        skill_names[i:i + max_size]
        for i in range(0, len(skill_names), max_size)
    ]


