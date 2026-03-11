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
Evolved Skill 加载与选择模块

对齐 Skill 系统的两阶段筛选机制：
1. SkillLoader 加载 SKILL.md → SkillMetadata
2. OperatorSkillSelector 粗筛（backend, dsl, category）
3. Selector Agent LLM 精筛（基于 description）
4. 全部选中的 skill 转为 handwrite_suggestion 格式导入
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional

from akg_agents.core_v2.skill.loader import SkillLoader
from akg_agents.core_v2.skill.metadata import SkillMetadata
from akg_agents.op.skill.operator_selector import (
    OperatorSkillSelector,
    OperatorSelectionContext,
)

logger = logging.getLogger(__name__)


def _load_and_coarse_filter(
    evolved_dir: str,
    dsl: str,
    backend: str,
) -> List[SkillMetadata]:
    """加载 evolved SKILL.md 并进行粗筛。

    Args:
        evolved_dir: evolved SKILL.md 所在目录
        dsl: DSL 类型（如 triton_ascend）
        backend: 后端（如 ascend）

    Returns:
        粗筛后的 SkillMetadata 列表
    """
    evolved_path = Path(evolved_dir)
    if not evolved_path.exists():
        logger.warning(f"Evolved skill directory does not exist: {evolved_dir}")
        return []

    loader = SkillLoader()
    all_skills = loader.load_from_directory(evolved_path)
    if not all_skills:
        logger.info(f"No valid SKILL.md found in {evolved_dir}")
        return []

    logger.info(f"Loaded {len(all_skills)} evolved skills from {evolved_dir}")

    selector = OperatorSkillSelector()
    context = OperatorSelectionContext(
        dsl=dsl.replace("_", "-"),
        backend=backend,
        include_category_groups=["knowledge", "example"],
    )
    filtered = selector.coarse_filter(all_skills, context)
    logger.info(f"Coarse filter: {len(all_skills)} -> {len(filtered)} evolved skills")

    return filtered


async def _llm_fine_select(
    candidates: List[SkillMetadata],
    op_name: str,
    task_desc: str,
    dsl: str,
    config: dict,
) -> List[str]:
    """使用 Selector Agent 进行 LLM 精筛，返回选中的 skill name 列表。

    复用现有的 Selector Agent，将 skill.description 作为 improvement_doc
    传入，使 select_relevant.j2 模板展示 description 供 LLM 做相关性判断。
    """
    from akg_agents.core.agent.selector import Selector

    selection_candidates = [
        {
            "name": skill.name,
            "improvement_doc": skill.description,
        }
        for skill in candidates
    ]

    selector_agent = Selector(
        op_name=op_name,
        task_desc=task_desc,
        dsl=dsl,
        config=config,
    )
    selected_names = await selector_agent.run(selection_candidates)
    logger.info(
        f"LLM fine select: {len(candidates)} -> {len(selected_names)} evolved skills"
    )
    return selected_names


def _to_handwrite_suggestions(skills: List[SkillMetadata]) -> List[Dict[str, str]]:
    """将 SkillMetadata 列表转换为 handwrite_suggestion 格式。

    每个 skill 的完整 markdown content 作为 improvement_doc，
    附带空的 framework_code/impl_code 以兼容 Designer 模板。
    """
    suggestions = []
    for skill in skills:
        suggestions.append({
            "name": f"evolved/{skill.name}",
            "description": skill.description,
            "improvement_doc": skill.content,
            "framework_code": "",
            "impl_code": "",
        })
    return suggestions


async def select_evolved_skills(
    evolved_dir: str,
    op_name: str,
    task_desc: str,
    dsl: str,
    backend: str,
    config: dict,
) -> List[Dict[str, str]]:
    """加载、粗筛、LLM 精筛 evolved skills，返回全部选中的 handwrite_suggestion 格式条目。

    与 Skill 系统（KernelGen._select_skills）保持一致的两阶段筛选：
    1. SkillLoader 加载 + OperatorSkillSelector 粗筛
    2. Selector Agent LLM 精筛（基于 name + description）
    3. 全部选中的 skill 转为 handwrite_suggestion 格式

    Args:
        evolved_dir: evolved SKILL.md 所在目录
        op_name: 算子名称
        task_desc: 任务描述
        dsl: DSL 类型
        backend: 后端类型
        config: agent 配置字典（需含 agent_model_config）

    Returns:
        全部选中的 evolved skill，handwrite_suggestion 格式
    """
    filtered = _load_and_coarse_filter(evolved_dir, dsl, backend)
    if not filtered:
        return []

    selected_names = await _llm_fine_select(
        filtered, op_name, task_desc, dsl, config
    )
    name_map = {s.name: s for s in filtered}
    selected_skills = [name_map[n] for n in selected_names if n in name_map]
    if not selected_skills:
        logger.info(
            f"LLM fine select returned no relevant skills for {op_name}, "
            f"skipping evolved skill injection"
        )
        return []

    suggestions = _to_handwrite_suggestions(selected_skills)
    skill_names = [s["name"] for s in suggestions]
    logger.info(
        f"Evolved skill selection complete: {len(suggestions)} skills selected: {skill_names}"
    )
    return suggestions
