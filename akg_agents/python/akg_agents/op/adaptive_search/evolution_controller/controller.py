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
Evolution Controller

进化控制器门面类，协调三个求解器：
  Solver 1: 谱系感知父代选择 (LineageAwareParentSelector)
  Solver 2: Boltzmann + 迁移灵感选择 (LineageBoltzmannInspirationSelector)
  Solver 3: 多维收敛检测 (MultiDimensionalConvergenceDetector)

作为 AKG AdaptiveSearchController 的外挂插件，通过薄代理钩子接入，
不修改原始搜索逻辑。
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from akg_agents.op.adaptive_search.success_db import SuccessDB, SuccessRecord
from akg_agents.op.adaptive_search.evolution_controller.config import (
    EvolutionControllerConfig,
    StrategySignal,
)
from akg_agents.op.adaptive_search.evolution_controller.lineage_tree import LineageTree
from akg_agents.op.adaptive_search.evolution_controller.parent_selector import (
    LineageAwareParentSelector,
)
from akg_agents.op.adaptive_search.evolution_controller.inspiration_selector import (
    LineageBoltzmannInspirationSelector,
)
from akg_agents.op.adaptive_search.evolution_controller.convergence_detector import (
    MultiDimensionalConvergenceDetector,
)

logger = logging.getLogger(__name__)


class EvolutionController:
    """
    进化控制器：协调谱系感知选择、智能灵感采样和自适应收敛检测。

    使用方式：
        config = EvolutionControllerConfig()
        evo = EvolutionController(config)
        evo.on_search_start()

        # 在搜索循环中：
        parent = evo.select_parent(db, progress)
        inspirations = evo.select_inspirations(parent, db, progress)
        should_stop, reason = evo.should_stop(submitted, success)
        evo.on_result(db, success=True)
    """

    def __init__(self, config: Optional[EvolutionControllerConfig] = None):
        config = config or EvolutionControllerConfig()

        self.lineage_tree = LineageTree()
        self.parent_selector = LineageAwareParentSelector(
            config.parent_selection, self.lineage_tree
        )
        self.inspiration_selector = LineageBoltzmannInspirationSelector(
            config.inspiration, self.lineage_tree
        )
        self.convergence_detector = MultiDimensionalConvergenceDetector(
            config.convergence
        )
        self._search_start_time: Optional[float] = None
        self._config = config

    def on_search_start(self) -> None:
        """搜索开始时调用，初始化计时器"""
        self._search_start_time = time.time()
        logger.info("EvolutionController initialized")

    def on_result(self, db: SuccessDB, success: bool) -> None:
        """
        任务完成后调用，更新内部状态。

        Args:
            db: 成功记录数据库
            success: 本次任务是否成功
        """
        if success and not db.is_empty():
            self.lineage_tree.rebuild(db)
            self.convergence_detector.update(db, self.lineage_tree)

    def select_parent(
        self, db: SuccessDB, progress: float
    ) -> Optional[SuccessRecord]:
        """
        选择父代（Solver 1 + Solver 3 策略信号）。

        Args:
            db: 成功记录数据库
            progress: 搜索进度 ∈ [0, 1]

        Returns:
            选中的 SuccessRecord，或 None
        """
        signal = self.convergence_detector.get_strategy_signal()
        return self.parent_selector.select(db, progress, signal=signal)

    def select_inspirations(
        self, parent: SuccessRecord, db: SuccessDB, progress: float
    ) -> List[Dict]:
        """
        选择灵感集（Solver 2 + Solver 3 策略信号）。

        Args:
            parent: 父代记录
            db: 成功记录数据库
            progress: 搜索进度 ∈ [0, 1]

        Returns:
            灵感列表，第一个元素为父代
        """
        signal = self.convergence_detector.get_strategy_signal()
        return self.inspiration_selector.select(
            parent, db, progress, signal=signal
        )

    def should_stop(
        self, total_submitted: int, total_success: int
    ) -> Tuple[bool, str]:
        """
        判断是否应该停止（Solver 3）。

        Args:
            total_submitted: 已提交的总任务数
            total_success: 成功的任务数

        Returns:
            (should_stop, reason_string)
        """
        elapsed = (
            time.time() - self._search_start_time
            if self._search_start_time else 0.0
        )
        return self.convergence_detector.should_stop(
            total_submitted, total_success, elapsed
        )

    def get_diagnostics(self) -> Dict[str, Any]:
        """
        获取诊断信息，用于日志和 UI 展示。

        Returns:
            包含谱系统计、多样性指数、收敛状态等信息的字典
        """
        lineage_info = {}
        for root_id, stats in self.lineage_tree.get_all_lineage_stats().items():
            lineage_info[root_id] = {
                "size": stats.size,
                "best_speedup": round(stats.best_speedup, 4),
                "share": round(stats.share, 4),
                "max_depth": stats.max_depth,
                "total_selections": stats.total_selections,
            }

        return {
            "lineage_stats": lineage_info,
            "num_lineages": self.lineage_tree.num_lineages,
            "diversity_index": round(self.lineage_tree.get_diversity_index(), 4),
            "convergence": self.convergence_detector.get_diagnostics(),
            "strategy_signal": self.convergence_detector.get_strategy_signal().__dict__,
        }
