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
UCB Parent Selector for Adaptive Search

基于 UCB (Upper Confidence Bound) 策略选择父代任务。
"""

import math
import random
import logging
from typing import Optional, List, Dict, Any

from ai_kernel_generator.core.adaptive_search.success_db import SuccessDB, SuccessRecord

logger = logging.getLogger(__name__)


class UCBParentSelector:
    """
    UCB 父代选择器
    
    使用 UCB1 算法选择父代，平衡利用（选择性能好的）和探索（选择被选次数少的）。
    
    UCB 公式:
        UCB(s) = Q(s) + c * sqrt(ln(N_total) / (N(s) + 1))
    
    其中:
        - Q(s): 质量得分，基于性能计算
        - N(s): 该记录被选择的次数
        - N_total: 全局总选择次数
        - c: 探索系数
    """
    
    def __init__(self,
                 db: SuccessDB,
                 exploration_coef: float = 1.414,
                 random_factor: float = 0.1,
                 use_softmax: bool = False,
                 softmax_temperature: float = 1.0):
        """
        初始化 UCB 选择器
        
        Args:
            db: 成功任务数据库
            exploration_coef: 探索系数 c（默认 sqrt(2)）
            random_factor: 随机扰动因子（0~1）
            use_softmax: 是否使用 softmax 概率采样
            softmax_temperature: softmax 温度参数
        """
        self.db = db
        self.exploration_coef = exploration_coef
        self.random_factor = random_factor
        self.use_softmax = use_softmax
        self.softmax_temperature = softmax_temperature
        
        logger.info(f"UCBParentSelector initialized: c={exploration_coef}, random={random_factor}, softmax={use_softmax}")
    
    def _compute_quality(self, record: SuccessRecord, baseline_gen_time: float) -> float:
        """
        计算质量得分 Q(s)
        
        性能越好（gen_time 越小），得分越高。
        
        Args:
            record: 成功记录
            baseline_gen_time: 基准 gen_time（用于归一化）
            
        Returns:
            float: 质量得分 (0~1)
        """
        gen_time = record.gen_time
        
        if gen_time == float('inf') or gen_time <= 0:
            return 0.0
        
        if baseline_gen_time == float('inf') or baseline_gen_time <= 0:
            baseline_gen_time = gen_time
        
        # 归一化质量得分：baseline / (baseline + gen_time)
        # gen_time 越小，得分越高
        quality = baseline_gen_time / (baseline_gen_time + gen_time)
        
        return quality
    
    def _compute_exploration(self, 
                             selection_count: int, 
                             total_selections: int) -> float:
        """
        计算探索项
        
        Args:
            selection_count: 该记录被选择的次数
            total_selections: 全局总选择次数
            
        Returns:
            float: 探索得分
        """
        if total_selections == 0:
            return float('inf')  # 首次选择，给最大探索奖励
        
        if selection_count == 0:
            return float('inf')  # 从未被选过，给最大探索奖励
        
        # UCB1 探索项: c * sqrt(ln(N) / n)
        exploration = self.exploration_coef * math.sqrt(
            math.log(total_selections + 1) / (selection_count + 1)
        )
        
        return exploration
    
    def _compute_ucb_score(self, 
                           record: SuccessRecord,
                           baseline_gen_time: float,
                           total_selections: int) -> float:
        """
        计算 UCB 得分
        
        Args:
            record: 成功记录
            baseline_gen_time: 基准 gen_time
            total_selections: 全局总选择次数
            
        Returns:
            float: UCB 得分
        """
        quality = self._compute_quality(record, baseline_gen_time)
        exploration = self._compute_exploration(record.selection_count, total_selections)
        
        # 添加随机扰动
        random_term = random.uniform(-self.random_factor, self.random_factor)
        
        ucb_score = quality + exploration + random_term
        
        logger.debug(
            f"Record {record.id[:8]}: Q={quality:.4f}, E={exploration:.4f}, "
            f"R={random_term:.4f}, UCB={ucb_score:.4f} "
            f"(gen_time={record.gen_time:.4f}, count={record.selection_count})"
        )
        
        return ucb_score
    
    def select(self) -> Optional[SuccessRecord]:
        """
        选择一个父代记录
        
        Returns:
            SuccessRecord: 被选中的记录，DB 为空时返回 None
        """
        if self.db.is_empty():
            logger.warning("SuccessDB is empty, cannot select parent")
            return None
        
        records = self.db.get_all()
        total_selections = self.db.get_total_selections()
        baseline_gen_time = self.db.get_best_gen_time()
        
        if len(records) == 1:
            # 只有一条记录，直接返回
            selected = records[0]
            self.db.increment_selection(selected.id)
            logger.info(f"Only one record, selecting {selected.id}")
            return selected
        
        # 计算所有记录的 UCB 得分
        scores: List[tuple] = []
        for record in records:
            score = self._compute_ucb_score(record, baseline_gen_time, total_selections)
            scores.append((record, score))
        
        if self.use_softmax:
            selected = self._softmax_select(scores)
        else:
            selected = self._greedy_select(scores)
        
        if selected:
            self.db.increment_selection(selected.id)
            logger.info(
                f"Selected parent {selected.id[:16]} "
                f"(gen_time={selected.gen_time:.4f}, count={selected.selection_count})"
            )
        
        return selected
    
    def _greedy_select(self, scores: List[tuple]) -> Optional[SuccessRecord]:
        """贪婪选择（选 UCB 最高的）"""
        if not scores:
            return None
        
        # 按 UCB 得分降序排序
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[0][0]
    
    def _softmax_select(self, scores: List[tuple]) -> Optional[SuccessRecord]:
        """Softmax 概率采样"""
        if not scores:
            return None
        
        # 计算 softmax 概率
        max_score = max(s[1] for s in scores)
        exp_scores = []
        for record, score in scores:
            # 防止溢出
            adjusted_score = (score - max_score) / self.softmax_temperature
            exp_scores.append((record, math.exp(adjusted_score)))
        
        total_exp = sum(e[1] for e in exp_scores)
        probabilities = [(r, e / total_exp) for r, e in exp_scores]
        
        # 按概率采样
        rand = random.random()
        cumulative = 0.0
        for record, prob in probabilities:
            cumulative += prob
            if rand <= cumulative:
                return record
        
        # 兜底返回最后一个
        return probabilities[-1][0]
    
    def get_selection_stats(self) -> Dict[str, Any]:
        """获取选择统计信息"""
        if self.db.is_empty():
            return {"total_records": 0, "total_selections": 0}
        
        records = self.db.get_all()
        selection_counts = [r.selection_count for r in records]
        
        return {
            "total_records": len(records),
            "total_selections": self.db.get_total_selections(),
            "max_selection_count": max(selection_counts),
            "min_selection_count": min(selection_counts),
            "avg_selection_count": sum(selection_counts) / len(selection_counts),
            "unselected_count": sum(1 for c in selection_counts if c == 0)
        }

