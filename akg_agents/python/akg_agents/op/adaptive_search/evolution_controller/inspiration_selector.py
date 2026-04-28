# Copyright 2025 Huawei Technologies Co., Ltd
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
Solver 2: Lineage Boltzmann Inspiration Selector

谱系内 Boltzmann 采样 + 概率迁移灵感选择器。

设计原则：
  - 灵感以本谱系为主，保持软岛屿的进化方向一致性
  - 以 p_migrate 概率从其他谱系精英中选取，实现跨谱系基因迁移
  - Boltzmann 采样替代固定分层 (3:4:3)，自适应数据分布

理论基础：
  - Boltzmann/Gibbs Selection (Goldberg 1990, de la Maza & Tidor 1993)
  - Island Model Migration (Cantú-Paz 2001: 最优迁移率 5%-20%)
  - 温度与 Solver 1 退火策略共享，减少独立参数
"""

import logging
import math
import random
from typing import Dict, List, Optional, Set

from akg_agents.op.adaptive_search.success_db import SuccessDB, SuccessRecord
from akg_agents.op.adaptive_search.evolution_controller.lineage_tree import LineageTree
from akg_agents.op.adaptive_search.evolution_controller.config import (
    InspirationSelectionConfig,
    StrategySignal,
)

logger = logging.getLogger(__name__)


class LineageBoltzmannInspirationSelector:
    """谱系内 Boltzmann 采样 + 概率迁移灵感选择器"""

    def __init__(self, config: InspirationSelectionConfig, lineage_tree: LineageTree):
        self.sample_num = config.inspiration_sample_num
        self.p_migrate = config.p_migrate
        self.foreign_elite_num = config.foreign_elite_num
        self.T_start = config.T_start
        self.T_end = config.T_end
        self.lineage_tree = lineage_tree

    def select(
        self,
        parent: SuccessRecord,
        db: SuccessDB,
        progress: float,
        signal: Optional[StrategySignal] = None,
    ) -> List[Dict]:
        """
        选择灵感集。

        Args:
            parent: 父代记录
            db: 成功记录数据库
            progress: 搜索进度 ∈ [0, 1]
            signal: 来自 Solver 3 的策略调整信号

        Returns:
            灵感列表，第一个元素为父代（标记 is_parent=True）
        """
        # 构造父代灵感
        parent_insp = parent.to_inspiration()
        parent_insp['is_parent'] = True
        result = [parent_insp]

        all_records = db.get_all()
        if len(all_records) <= 1:
            return result

        # 计算 Boltzmann 温度（与 Solver 1 退火策略联动）
        T = self._anneal(self.T_start, self.T_end, progress)

        # 应用 Solver 3 策略信号
        p_migrate = self.p_migrate
        if signal:
            p_migrate = min(p_migrate * signal.p_migrate_multiplier, 0.8)
            T *= signal.T_insp_multiplier

        # 构建候选池
        parent_lineage = self.lineage_tree.get_lineage_root(parent.id)

        local_pool = [
            r for r in all_records
            if r.id != parent.id
            and self.lineage_tree.get_lineage_root(r.id) == parent_lineage
        ]
        foreign_elites = self._build_foreign_elites(
            all_records, parent.id, parent_lineage
        )

        # 逐个采样
        selected: List[SuccessRecord] = []
        selected_ids: Set[str] = set()

        for _ in range(self.sample_num):
            insp = None

            # 迁移判定
            if (random.random() < p_migrate
                    and foreign_elites
                    and any(e.id not in selected_ids for e in foreign_elites)):
                available = [e for e in foreign_elites if e.id not in selected_ids]
                insp = random.choice(available)
            else:
                # 本谱系 Boltzmann 采样
                available = [r for r in local_pool if r.id not in selected_ids]
                if available:
                    insp = self._boltzmann_sample(available, T)
                else:
                    # 本谱系候选耗尽，fallback 到全局
                    fallback = [
                        r for r in all_records
                        if r.id != parent.id and r.id not in selected_ids
                    ]
                    if fallback:
                        insp = self._boltzmann_sample(fallback, T)

            if insp:
                selected.append(insp)
                selected_ids.add(insp.id)

        # 排序：本谱系靠前（按质量），迁移灵感靠后
        local_selected = [
            s for s in selected
            if self.lineage_tree.get_lineage_root(s.id) == parent_lineage
        ]
        foreign_selected = [
            s for s in selected
            if self.lineage_tree.get_lineage_root(s.id) != parent_lineage
        ]
        local_selected.sort(key=lambda r: r.gen_time)
        foreign_selected.sort(key=lambda r: r.gen_time)

        # 转换为灵感格式
        for s in local_selected + foreign_selected:
            insp = s.to_inspiration()
            insp['is_parent'] = False
            insp['is_migration'] = (
                self.lineage_tree.get_lineage_root(s.id) != parent_lineage
            )
            result.append(insp)

        logger.debug(
            f"Selected {len(selected)} inspirations "
            f"(local={len(local_selected)}, foreign={len(foreign_selected)}, "
            f"T={T:.3f}, p_migrate={p_migrate:.2f})"
        )

        return result

    def _build_foreign_elites(
        self,
        all_records: List[SuccessRecord],
        parent_id: str,
        parent_lineage: str,
    ) -> List[SuccessRecord]:
        """构建其他谱系的精英池（每谱系 top-m）"""
        foreign_by_lineage: Dict[str, List[SuccessRecord]] = {}
        for r in all_records:
            if r.id == parent_id:
                continue
            root = self.lineage_tree.get_lineage_root(r.id)
            if root != parent_lineage:
                foreign_by_lineage.setdefault(root, []).append(r)

        elites = []
        for lineage_records in foreign_by_lineage.values():
            sorted_records = sorted(lineage_records, key=lambda r: r.gen_time)
            elites.extend(sorted_records[:self.foreign_elite_num])

        return elites

    @staticmethod
    def _boltzmann_sample(
        candidates: List[SuccessRecord], T: float
    ) -> SuccessRecord:
        """Boltzmann (Gibbs) 概率采样"""
        if len(candidates) == 1:
            return candidates[0]

        max_speedup = max(r.speedup for r in candidates)
        qualities = [r.speedup / max(max_speedup, 1e-9) for r in candidates]

        # 数值稳定的 softmax
        max_q = max(qualities)
        T = max(T, 1e-9)
        exp_values = [math.exp((q - max_q) / T) for q in qualities]
        total = sum(exp_values)
        if total < 1e-15:
            return random.choice(candidates)
        probs = [e / total for e in exp_values]

        # 按概率采样
        r = random.random()
        cumsum = 0.0
        for i, p in enumerate(probs):
            cumsum += p
            if cumsum >= r:
                return candidates[i]
        return candidates[-1]

    @staticmethod
    def _anneal(start: float, end: float, progress: float) -> float:
        """指数退火"""
        progress = min(max(progress, 0.0), 1.0)
        if start <= 0 or end <= 0:
            return end
        return start * (end / start) ** progress
