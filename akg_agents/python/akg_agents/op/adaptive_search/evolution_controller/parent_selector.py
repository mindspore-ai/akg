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
Solver 1: Lineage-Aware Parent Selector

谱系感知的软岛屿父代选择器。

两阶段选择：
  Phase 1 (层间): 基于探索赤字选择谱系
  Phase 2 (层内): 在选定谱系内使用局部 UCB 选择具体记录

理论基础：
  - 软岛屿策略：好谱系获得更多机会，但不能无限膨胀
  - 退火调度：搜索进度驱动的探索-利用平衡 (Auer et al., 2002)
  - Softmax 应得份额分配：避免直接比例分配的极端倾斜
"""

import logging
import math
import random
from typing import Dict, Optional

from akg_agents.op.adaptive_search.success_db import SuccessDB, SuccessRecord
from akg_agents.op.adaptive_search.evolution_controller.lineage_tree import LineageTree
from akg_agents.op.adaptive_search.evolution_controller.config import (
    ParentSelectionConfig,
    StrategySignal,
)

logger = logging.getLogger(__name__)


class LineageAwareParentSelector:
    """谱系感知的软岛屿父代选择器"""

    def __init__(self, config: ParentSelectionConfig, lineage_tree: LineageTree):
        self.tau_start = config.tau_start
        self.tau_end = config.tau_end
        self.c_local_start = config.c_local_start
        self.c_local_end = config.c_local_end
        self.p_exploit = config.p_exploit
        self.quality_bonus_strength = config.quality_bonus_strength
        self.lineage_tree = lineage_tree

    def select(
        self,
        db: SuccessDB,
        progress: float,
        signal: Optional[StrategySignal] = None,
    ) -> Optional[SuccessRecord]:
        """
        选择父代记录。

        Args:
            db: 成功记录数据库
            progress: 搜索进度 ∈ [0, 1]
            signal: 来自 Solver 3 的策略调整信号

        Returns:
            选中的 SuccessRecord，或 None（DB 为空时）
        """
        if db.is_empty():
            return None

        # 重建谱系树
        self.lineage_tree.rebuild(db)

        records = db.get_all()
        if len(records) == 1:
            record = records[0]
            db.increment_selection(record.id)
            return record

        # 计算当前退火参数
        tau = self._anneal(self.tau_start, self.tau_end, progress)
        c_local = self._anneal(self.c_local_start, self.c_local_end, progress)

        # 应用 Solver 3 的策略信号
        p_exploit = self.p_exploit
        if signal:
            tau *= signal.tau_multiplier
            if signal.p_exploit_override is not None:
                p_exploit = signal.p_exploit_override

        # Phase 1: 选择谱系
        lineage_id = self._select_lineage(tau, p_exploit)

        if lineage_id is None:
            # fallback: 用原始 UCB 选全局最优
            logger.warning("Lineage selection returned None, falling back to global best")
            best = db.get_best_record()
            if best:
                db.increment_selection(best.id)
            return best

        # Phase 2: 谱系内 UCB 选择
        record = self._select_within_lineage(db, lineage_id, c_local)

        if record:
            db.increment_selection(record.id)
        return record

    def _select_lineage(self, tau: float, p_exploit: float) -> Optional[str]:
        """
        Phase 1: 基于探索赤字选择谱系。

        计算各谱系的应得份额 d_i（softmax 加权潜力）和实际份额 a_i，
        然后通过 exploit（argmax 赤字×质量加成）或 explore（轮盘赌赤字）选择谱系。
        """
        all_stats = self.lineage_tree.get_all_lineage_stats()
        if not all_stats:
            return None

        K = len(all_stats)
        if K == 1:
            return list(all_stats.keys())[0]

        # 计算谱系潜力（归一化 speedup）
        total_speedup = sum(s.best_speedup for s in all_stats.values())
        if total_speedup < 1e-9:
            return random.choice(list(all_stats.keys()))

        potentials = {
            root_id: stats.best_speedup / total_speedup
            for root_id, stats in all_stats.items()
        }

        # 计算应得份额（softmax with temperature）
        deserved = self._softmax(potentials, tau)

        # 计算探索赤字
        N = sum(s.size for s in all_stats.values())
        epsilon = 1.0 / max(N, 1)
        deficits = {}
        for root_id, stats in all_stats.items():
            actual = stats.share
            deficits[root_id] = deserved[root_id] / (actual + epsilon)

        # 选择：exploit 模式或 explore 模式
        if random.random() < p_exploit:
            # 加入 quality_bonus 以区分赤字相近的谱系
            scores = {}
            speedups = [s.best_speedup for s in all_stats.values()]
            s_min, s_max = min(speedups), max(speedups)
            for root_id, stats in all_stats.items():
                norm_q = (stats.best_speedup - s_min) / (s_max - s_min + 1e-9)
                quality_bonus = 1.0 + self.quality_bonus_strength * norm_q
                scores[root_id] = deficits[root_id] * quality_bonus
            return max(scores, key=scores.get)
        else:
            return self._roulette(deficits)

    def _select_within_lineage(
        self, db: SuccessDB, lineage_id: str, c_local: float
    ) -> Optional[SuccessRecord]:
        """
        Phase 2: 谱系内局部 UCB 选择。

        使用全局排名质量 + 谱系内探索奖励。
        """
        all_records = db.get_all()
        lineage_records = [
            r for r in all_records
            if self.lineage_tree.get_lineage_root(r.id) == lineage_id
        ]
        if not lineage_records:
            return None

        # 全局排名质量（与原 UCB 一致）
        sorted_all = sorted(all_records, key=lambda r: r.gen_time)
        rank_map = {r.id: i + 1 for i, r in enumerate(sorted_all)}
        n_total = len(sorted_all)

        # 谱系内总选择次数
        S_lineage = sum(r.selection_count for r in lineage_records)

        best_score = -float('inf')
        best_record = None
        for r in lineage_records:
            rank = rank_map[r.id]
            quality = (n_total - rank) / max(n_total - 1, 1)
            exploration = c_local * math.sqrt(
                math.log(S_lineage + 1) / (r.selection_count + 1)
            )
            score = quality + exploration
            if score > best_score:
                best_score = score
                best_record = r

        return best_record

    @staticmethod
    def _anneal(start: float, end: float, progress: float) -> float:
        """指数退火：start * (end/start)^progress"""
        progress = min(max(progress, 0.0), 1.0)
        if start <= 0 or end <= 0:
            return end
        return start * (end / start) ** progress

    @staticmethod
    def _softmax(values: Dict[str, float], temperature: float) -> Dict[str, float]:
        """带温度的 softmax"""
        max_v = max(values.values())
        temperature = max(temperature, 1e-9)
        exp_values = {
            k: math.exp((v - max_v) / temperature)
            for k, v in values.items()
        }
        total = sum(exp_values.values())
        if total < 1e-15:
            n = len(values)
            return {k: 1.0 / n for k in values}
        return {k: v / total for k, v in exp_values.items()}

    @staticmethod
    def _roulette(weights: Dict[str, float]) -> str:
        """轮盘赌选择"""
        total = sum(weights.values())
        if total < 1e-15:
            return random.choice(list(weights.keys()))
        r = random.random() * total
        cumsum = 0.0
        for k, w in weights.items():
            cumsum += w
            if cumsum >= r:
                return k
        return list(weights.keys())[-1]
