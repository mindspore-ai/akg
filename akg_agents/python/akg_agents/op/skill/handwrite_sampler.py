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
基于 Skill 系统的手写优化建议加载与采样

替代旧的 HandwriteLoader / HandwriteSampler (基于文件系统/RAG)，
在 Skill 体系下实现相同的两阶段筛选 + 加权随机采样：

1. SkillHandwriteLoader
   - 从 {dsl}/cases/ 目录加载案例 Skills
   - OperatorSkillSelector 粗筛 (dsl/backend)
   - Selector Agent (LLM) 精筛 (全文内容，排序)

2. SkillHandwriteSampler
   - 指数衰减加权随机采样（与旧 HandwriteSampler 算法一致）
   - 每次采样不同子集，保证进化多样性
   - 用完自动重置

用法（在 TaskGenerator / evolution_processors 中）：
    loader = SkillHandwriteLoader(dsl=..., backend=..., op_name=..., task_desc=..., config=...)
    await loader.select_relevant_skills()
    sampler = SkillHandwriteSampler(loader, sample_num=2, decay_rate=2.0)
    suggestions = sampler.sample()  # List[Dict] with {name, improvement_doc}
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np

from akg_agents import get_project_root
from akg_agents.core_v2.skill import SkillLoader
from akg_agents.op.skill.operator_selector import OperatorSkillSelector, OperatorSelectionContext

logger = logging.getLogger(__name__)

project_root = Path(get_project_root())
SKILLS_DIR = project_root / "op" / "resources" / "skills"


class SkillHandwriteLoader:
    """基于 Skill 系统的手写优化建议加载器

    加载 {dsl}/cases/ 下的案例 Skills，通过粗筛 + LLM 精筛
    得到按相关性排序的候选列表。

    与旧 HandwriteLoader 的对应关系：
    - 旧：遍历 handwrite_database 文件系统 → Selector Agent 筛选
    - 新：加载 cases/ Skills → OperatorSkillSelector 粗筛 → Selector Agent 精筛
    """

    def __init__(
        self,
        dsl: str,
        backend: str,
        op_name: str,
        task_desc: str,
        config: dict,
    ):
        self.dsl = dsl
        self.backend = backend
        self.op_name = op_name
        self.task_desc = task_desc
        self.config = config

        self._loader = SkillLoader()
        self._selector = OperatorSkillSelector()
        self._selected_skills: List[Dict[str, str]] = []

    async def select_relevant_skills(self) -> None:
        """加载案例 Skills → 粗筛 → LLM 精筛排序

        调用后通过 get_selected_skills() 获取结果。
        """
        dsl_key = self.dsl.replace("_", "-")
        cases_dir = SKILLS_DIR / dsl_key / "cases"
        if not cases_dir.exists():
            logger.info(f"No cases directory for DSL '{dsl_key}': {cases_dir}")
            return

        all_skills = self._loader.load_from_directory(cases_dir)
        if not all_skills:
            logger.info(f"No case skills found in {cases_dir}")
            return

        # 粗筛：dsl / backend 匹配
        context = OperatorSelectionContext(
            dsl=dsl_key,
            backend=self.backend,
        )
        filtered = self._selector.coarse_filter(all_skills, context)
        if not filtered:
            logger.info("No case skills passed coarse filter")
            return

        # 构建候选列表（full content，供 LLM 精筛）
        candidates = []
        for skill in filtered:
            candidates.append({
                "name": skill.name,
                "improvement_doc": skill.content,
            })

        # LLM 精筛
        try:
            selected_names = await self._llm_select(candidates)
            # 按 LLM 返回的顺序排列（最相关在前）
            name_to_candidate = {c["name"]: c for c in candidates}
            self._selected_skills = [
                name_to_candidate[n] for n in selected_names if n in name_to_candidate
            ]
            logger.info(
                f"SkillHandwriteLoader: selected {len(self._selected_skills)}/{len(candidates)} "
                f"case skills for {self.op_name}"
            )
        except Exception as e:
            logger.warning(f"LLM selection failed: {e}, using all filtered skills")
            self._selected_skills = candidates

    async def _llm_select(self, candidates: List[Dict[str, str]]) -> List[str]:
        """使用 Selector Agent 的 LLM 调用精筛候选"""
        from akg_agents.core.agent.selector import Selector
        from akg_agents.utils.common_utils import ParserFactory

        selector = Selector(
            op_name=self.op_name,
            task_desc=self.task_desc,
            dsl=self.dsl,
            config=self.config,
        )

        # 适配 Selector.run() 所需的 candidate 格式
        adapted = []
        for c in candidates:
            adapted.append({
                "name": c["name"],
                "framework_code": self.task_desc,
                "impl_code": "",
                "improvement_doc": c["improvement_doc"],
            })

        selected_names = await selector.run(adapted)
        return selected_names

    def get_selected_skills(self) -> List[Dict[str, str]]:
        """获取筛选后的案例列表（按相关性排序）"""
        return self._selected_skills.copy()


class SkillHandwriteSampler:
    """基于 Skill 的手写优化建议采样器

    与旧 HandwriteSampler 算法完全一致：
    - 指数衰减权重: weight(i) = exp(-decay_rate * i / total_count)
    - 索引越小（相关性越高）权重越大
    - 不放回采样，用完自动重置
    """

    def __init__(
        self,
        loader: SkillHandwriteLoader,
        sample_num: int = 2,
        decay_rate: float = 2.0,
    ):
        self.loader = loader
        self.sample_num = sample_num
        self.decay_rate = decay_rate

        self._available_skills = self.loader.get_selected_skills()
        self._total_count = len(self._available_skills)
        self._used_indices: set = set()
        self._weights = self._compute_weights()

        if self._total_count == 0:
            logger.warning("No case skills available for sampling")
        else:
            logger.info(
                f"SkillHandwriteSampler initialized with {self._total_count} skills, "
                f"sample_num={sample_num}, decay_rate={decay_rate}"
            )
            if self._total_count > 1:
                weight_ratio = self._weights[0] / self._weights[-1]
                logger.info(f"Weight ratio (first/last): {weight_ratio:.2f}x")

    def _compute_weights(self) -> np.ndarray:
        if self._total_count == 0:
            return np.array([])
        indices = np.arange(self._total_count)
        weights = np.exp(-self.decay_rate * indices / self._total_count)
        return weights / weights.sum()

    def sample(self) -> List[Dict[str, str]]:
        """加权随机采样，返回 [{name, improvement_doc}, ...]"""
        if self._total_count == 0:
            return []

        available_indices = list(set(range(self._total_count)) - self._used_indices)
        if len(available_indices) == 0:
            logger.debug("All skills used, resetting sampler")
            self._used_indices.clear()
            available_indices = list(range(self._total_count))

        actual_sample_num = min(self.sample_num, len(available_indices))

        available_weights = self._weights[available_indices]
        available_probs = available_weights / available_weights.sum()

        sampled_indices = np.random.choice(
            available_indices,
            size=actual_sample_num,
            replace=False,
            p=available_probs,
        )

        self._used_indices.update(sampled_indices)

        suggestions = []
        imported_names = []
        for idx in sampled_indices:
            skill = self._available_skills[idx]
            suggestions.append(skill)
            imported_names.append(skill["name"])

        if imported_names:
            logger.info(
                f"SkillHandwriteSampler: sampled {len(imported_names)} case skills"
            )
            for i, name in enumerate(imported_names, 1):
                logger.info(f"  [{i}] {name}")

        return suggestions

    def reset(self):
        """重置采样器，清空已使用记录"""
        self._used_indices.clear()
        logger.debug("SkillHandwriteSampler reset")
