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
Skill 自进化系统 - SKILL.md 写入器

负责将 LLM 生成的 Markdown 正文与 YAML frontmatter 组装后写入磁盘。

职责：
1. 规范化 skill_name（小写字母+连字符）
2. 生成 YAML frontmatter（元信息：backend、dsl、speedup 等）
3. 拼接 frontmatter + LLM 正文 → 写入 SKILL.md

输出路径：
- 默认: op/resources/skills/{dsl}/cases/{skill_name}/SKILL.md
- 可通过 output_dir 参数自定义
"""

import logging
import os
import re
from pathlib import Path
from typing import Dict, Optional, Union

from .models import CompressedData

logger = logging.getLogger(__name__)


class SkillWriter:
    """SKILL.md 文件写入器

    将 LLM 生成的 Markdown 正文附上 YAML frontmatter 后保存为标准 SKILL.md。
    """

    def write(
        self,
        skill_name: str,
        description: str,
        markdown_body: str,
        compressed: Union[CompressedData, Dict[str, str]],
        output_dir: Optional[str] = None,
    ) -> str:
        """组装并写入 SKILL.md 文件

        Args:
            skill_name: 技能名称（由 LLM 输出）
            description: 一句话描述（由 LLM 输出，用于 KernelGen 检索和筛选）
            markdown_body: LLM 生成的 Markdown 正文内容
            compressed: CompressedData（Mode A）或元数据 dict（Mode B，需含
                        op_name, dsl, backend, source 等键）
            output_dir: 自定义输出目录（可选，默认写入项目 skills 目录）

        Returns:
            写入的 SKILL.md 文件绝对路径
        """
        if isinstance(compressed, dict):
            is_feedback = compressed.get("source") == "user_feedback"
            op_name = compressed.get("op_name", "")
            dsl = compressed.get("dsl", "")
        else:
            is_feedback = False
            op_name = compressed.op_name
            dsl = compressed.dsl

        fallback_prefix = "exp" if is_feedback else "case"

        skill_name = self._sanitize_skill_name(
            skill_name, op_name, dsl,
            fallback_prefix=fallback_prefix,
        )

        if output_dir:
            skill_dir = os.path.join(output_dir, skill_name)
        else:
            skill_dir = self._default_skill_dir(dsl, skill_name)

        os.makedirs(skill_dir, exist_ok=True)
        skill_path = os.path.join(skill_dir, "SKILL.md")

        source = "user_feedback" if is_feedback else "adaptive_search"
        frontmatter = self._build_frontmatter(
            skill_name, description, compressed, source=source,
        )
        content = frontmatter + "\n" + markdown_body + "\n"

        with open(skill_path, "w", encoding="utf-8") as f:
            f.write(content)

        logger.info(f"[SkillEvolution:Writer] SKILL.md 已写入: {skill_path}")
        return skill_path

    def _sanitize_skill_name(
        self, name: str, op_name: str, dsl: str, fallback_prefix: str = "case",
    ) -> str:
        """规范化 skill 名称

        Mode A: {dsl}-case-{category}-{detail}
        Mode B: {dsl}-exp-{category}-{detail}
        """
        dsl_key = dsl.replace("_", "-").lower()
        if not name:
            name = f"{dsl_key}-{fallback_prefix}-{op_name}"

        name = name.lower().strip()
        name = re.sub(r"[^a-z0-9-]", "-", name)
        name = re.sub(r"-+", "-", name)
        name = name.strip("-")
        return name or f"{dsl_key}-{fallback_prefix}-auto"

    def _default_skill_dir(self, dsl: str, skill_name: str) -> str:
        """获取默认 skill 输出目录: op/resources/skills/{dsl}/cases/{skill_name}/"""
        try:
            from akg_agents import get_project_root
            project_root = Path(get_project_root())
        except ImportError:
            project_root = Path(__file__).resolve().parents[3]

        dsl_key = dsl.replace("_", "-").lower()
        return str(
            project_root / "op" / "resources" / "skills"
            / dsl_key / "cases" / skill_name
        )

    def _build_frontmatter(
        self,
        skill_name: str,
        description: str,
        compressed: Union[CompressedData, Dict],
        source: str = "adaptive_search",
    ) -> str:
        """构建 YAML frontmatter 元信息块"""
        if isinstance(compressed, dict):
            op_name = compressed.get("op_name", "")
            backend = compressed.get("backend", "")
            dsl = compressed.get("dsl", "")
        else:
            op_name = compressed.op_name
            backend = compressed.backend
            dsl = compressed.dsl

        default_desc = (
            f"{op_name} 人工调优经验"
            if source == "user_feedback"
            else f"{op_name} 自适应搜索优化经验"
        )
        desc = description or default_desc
        desc = desc.replace('"', '\\"')
        lines = [
            "---",
            f"name: {skill_name}",
            f'description: "{desc}"',
            "category: example",
            'version: "1.0.0"',
            "metadata:",
        ]
        if backend:
            lines.append(f"  backend: {backend}")
        if dsl:
            lines.append(f"  dsl: {dsl}")
        lines.append(f"  source: {source}")
        lines.append("---")
        return "\n".join(lines) + "\n"
