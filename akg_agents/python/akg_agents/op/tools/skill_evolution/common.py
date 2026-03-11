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
Skill 自进化系统 - 公共类型与工具

包含：
- 数据模型（TaskRecord, EvolutionStep, CompressedData）
- 代码 diff 工具函数（各模式共用）
- 输出层：prompt 变量构建 + LLM 输出解析 + SKILL.md 写入
"""

import difflib
import yaml
import re
from dataclasses import dataclass, field
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

DIFF_MAX_LINES = 200


# ==================== 数据模型 ====================


@dataclass
class TaskRecord:
    """一个搜索任务的核心记录"""
    task_id: str
    parent_id: str = ""
    generation: int = 0
    code: str = ""
    speedup: float = 0.0
    gen_time: float = float("inf")


@dataclass
class EvolutionStep:
    """进化链中的一步：父代 → 子代 diff"""
    parent_id: str
    child_id: str
    parent_speedup: float = 0.0
    child_speedup: float = 0.0
    parent_gen_time: float = float("inf")
    child_gen_time: float = float("inf")
    code_diff: str = ""


@dataclass
class CompressedData:
    """搜索日志压缩后的数据，直接注入 LLM prompt"""
    op_name: str = ""
    dsl: str = ""
    backend: str = ""
    arch: str = ""
    task_desc: str = ""
    best_task_id: str = ""
    best_speedup: float = 0.0
    best_gen_time: float = float("inf")
    best_code: str = ""
    evolution_chains: List[EvolutionStep] = field(default_factory=list)
    performance_summary: str = ""
    total_tasks: int = 0
    success_count: int = 0


# ==================== 共享工具函数 ====================


def strip_comments(code: str) -> str:
    """移除 Python 注释和 docstring，用于干净 diff"""
    if not code:
        return ""
    code = re.sub(r'"""[\s\S]*?"""', '', code)
    code = re.sub(r"'''[\s\S]*?'''", '', code)

    cleaned: List[str] = []
    for line in code.splitlines():
        if line.strip().startswith("#"):
            continue
        result: List[str] = []
        in_sq, in_dq = False, False
        i = 0
        while i < len(line):
            ch = line[i]
            if ch == '\\' and i + 1 < len(line):
                result.append(ch)
                result.append(line[i + 1])
                i += 2
                continue
            if ch == '"' and not in_sq:
                in_dq = not in_dq
            elif ch == "'" and not in_dq:
                in_sq = not in_sq
            elif ch == '#' and not in_sq and not in_dq:
                break
            result.append(ch)
            i += 1
        cleaned.append("".join(result).rstrip())

    merged: List[str] = []
    prev_blank = False
    for line in cleaned:
        blank = not line.strip()
        if blank and prev_blank:
            continue
        prev_blank = blank
        merged.append(line)

    return "\n".join(merged).strip() + "\n" if merged else ""


def code_diff(
    base: str, target: str, base_label: str, target_label: str,
    truncate: bool = True,
) -> str:
    """注释剥离后生成 unified diff

    Args:
        truncate: 是否截断超长 diff（默认 True，上限 DIFF_MAX_LINES 行）。
                  error_fix 模式传 False 以保留完整 diff。
    """
    base = strip_comments(base)
    target = strip_comments(target)

    diff_lines = list(difflib.unified_diff(
        base.splitlines(keepends=True),
        target.splitlines(keepends=True),
        fromfile=f"a/{base_label}",
        tofile=f"b/{target_label}",
        n=1,
    ))
    if not diff_lines:
        return "(代码相同)"

    if truncate and len(diff_lines) > DIFF_MAX_LINES:
        total = len(diff_lines)
        diff_lines = diff_lines[:DIFF_MAX_LINES]
        diff_lines.append(f"... (截断，共 {total} 行)\n")

    return "".join(diff_lines).rstrip()


# ==================== LLM 输出解析 ====================


def parse_skill_output(llm_output: str) -> Tuple[str, str, str]:
    """从 LLM 输出中提取 skill_name、description 和正文

    Returns:
        (skill_name, description, markdown_body)
    """
    if not llm_output:
        return "", "", ""

    text = llm_output.strip()
    skill_name = ""
    description = ""

    lines = text.splitlines()
    consumed = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            consumed += 1
            continue
        m_name = re.match(r"skill_name:\s*(.+)", stripped)
        if m_name:
            skill_name = m_name.group(1).strip()
            consumed += 1
            continue
        m_desc = re.match(r"description:\s*(.+)", stripped)
        if m_desc:
            description = m_desc.group(1).strip()
            consumed += 1
            continue
        break

    text = "\n".join(lines[consumed:]).strip()
    if text.startswith("```"):
        text = re.sub(r"^```\w*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)

    return skill_name, description, text.strip()


# ==================== SKILL.md 写入 ====================


class SkillWriter:
    """SKILL.md 文件写入器

    将 LLM 生成的 Markdown 正文附上 YAML frontmatter 后保存为标准 SKILL.md。
    支持 search_log（CompressedData）、expert_tuning（dict）和 error_fix（dict）三种模式。
    """

    _SOURCE_PREFIX_MAP = {
        "expert_tuning": "exp",
        "error_fix": "fix",
    }

    def write(
        self,
        skill_name: str,
        description: str,
        markdown_body: str,
        compressed: Union[CompressedData, Dict[str, str]],
        output_dir: Optional[str] = None,
    ) -> str:
        if isinstance(compressed, dict):
            source = compressed.get("source", "search_log")
            op_name = compressed.get("op_name", "")
            dsl = compressed.get("dsl", "")
        else:
            source = "search_log"
            op_name = compressed.op_name
            dsl = compressed.dsl

        fallback_prefix = self._SOURCE_PREFIX_MAP.get(source, "case")

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
        dsl_key = dsl.replace("_", "-").lower()
        if not name:
            name = f"{dsl_key}-{fallback_prefix}-{op_name}"

        name = name.lower().strip()
        name = re.sub(r"[^a-z0-9-]", "-", name)
        name = re.sub(r"-+", "-", name)
        name = name.strip("-")
        return name or f"{dsl_key}-{fallback_prefix}-auto"

    def _default_skill_dir(self, dsl: str, skill_name: str) -> str:
        try:
            from akg_agents import get_project_root
            project_root = Path(get_project_root())
        except ImportError:
            project_root = Path(__file__).resolve().parents[3]

        dsl_key = dsl.replace("_", "-").lower()
        return str(
            project_root / "op" / "resources" / "skills"
            / dsl_key / "evolved" / skill_name
        )

    ERROR_FIX_SKILL_NAME = "error-fix"
    ERROR_FIX_DESCRIPTION = "常见错误及修复方法，用于代码生成时避免同类问题"

    def get_error_fix_skill_path(
        self,
        metadata: Dict[str, str],
        output_dir: Optional[str] = None,
    ) -> str:
        """返回 error_fix SKILL.md 的目标路径（不创建文件）"""
        dsl = metadata.get("dsl", "")
        if output_dir:
            skill_dir = os.path.join(output_dir, self.ERROR_FIX_SKILL_NAME)
        else:
            skill_dir = self._default_skill_dir(dsl, self.ERROR_FIX_SKILL_NAME)
        return os.path.join(skill_dir, "SKILL.md")

    @staticmethod
    def read_error_fix_body(skill_path: str) -> str:
        """读取已有 error_fix SKILL.md 的正文（去除 YAML frontmatter）"""
        if not os.path.isfile(skill_path):
            return ""
        try:
            with open(skill_path, "r", encoding="utf-8") as f:
                content = f.read()
        except OSError:
            return ""
        if content.startswith("---"):
            end = content.find("---", 3)
            if end != -1:
                return content[end + 3:].strip()
        return content.strip()

    def write_error_fix(
        self,
        markdown_body: str,
        metadata: Dict[str, str],
        output_dir: Optional[str] = None,
    ) -> str:
        """error_fix 专用写入：文件不存在时新建，已存在时追加

        调用方负责在追加前通过 LLM 去重，确保 markdown_body 只包含新增内容。
        """
        dsl = metadata.get("dsl", "")
        backend = metadata.get("backend", "")
        skill_path = self.get_error_fix_skill_path(metadata, output_dir)

        os.makedirs(os.path.dirname(skill_path), exist_ok=True)

        if os.path.isfile(skill_path):
            with open(skill_path, "a", encoding="utf-8") as f:
                f.write("\n\n" + markdown_body.strip() + "\n")
            logger.info(f"[SkillEvolution:Writer] error_fix SKILL.md 已追加: {skill_path}")
        else:
            frontmatter_text = self._make_error_fix_frontmatter(dsl, backend)
            with open(skill_path, "w", encoding="utf-8") as f:
                f.write(frontmatter_text + "\n\n" + markdown_body.strip() + "\n")
            logger.info(f"[SkillEvolution:Writer] error_fix SKILL.md 已新建: {skill_path}")

        return skill_path

    def _make_error_fix_frontmatter(self, dsl: str, backend: str) -> str:
        meta: Dict[str, str] = {"source": "error_fix"}
        if backend:
            meta["backend"] = backend
        if dsl:
            meta["dsl"] = dsl

        frontmatter = {
            "name": self.ERROR_FIX_SKILL_NAME,
            "description": self.ERROR_FIX_DESCRIPTION,
            "category": "example",
            "version": "1.0.0",
            "metadata": meta,
        }
        yaml_str = yaml.dump(
            frontmatter, default_flow_style=False,
            allow_unicode=True, sort_keys=False,
        )
        return f"---\n{yaml_str}---"

    def _build_frontmatter(
        self,
        skill_name: str,
        description: str,
        compressed: Union[CompressedData, Dict],
        source: str = "search_log",
    ) -> str:
        if isinstance(compressed, dict):
            op_name = compressed.get("op_name", "")
            backend = compressed.get("backend", "")
            dsl = compressed.get("dsl", "")
        else:
            op_name = compressed.op_name
            backend = compressed.backend
            dsl = compressed.dsl

        _desc_map = {
            "expert_tuning": f"{op_name} 专家调优经验",
            "error_fix": f"{op_name} 错误修复经验",
        }
        default_desc = _desc_map.get(source, f"{op_name} 搜索日志优化经验")
        desc = description or default_desc

        meta: Dict[str, str] = {"source": source}
        if backend:
            meta["backend"] = backend
        if dsl:
            meta["dsl"] = dsl.replace("_", "-").lower()

        frontmatter = {
            "name": skill_name,
            "description": desc,
            "category": "example",
            "version": "1.0.0",
            "metadata": meta,
        }
        yaml_str = yaml.dump(
            frontmatter, default_flow_style=False,
            allow_unicode=True, sort_keys=False,
        )
        return f"---\n{yaml_str}---\n"
