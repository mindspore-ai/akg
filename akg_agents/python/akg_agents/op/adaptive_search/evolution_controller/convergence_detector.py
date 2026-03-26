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
Solver 3: Multi-Dimensional Convergence Detector

多维收敛检测器。

四个独立的收敛信号：
  S1: 性能停滞度 (Performance Plateau)
  S2: 多样性趋势 (Diversity Trend)
  S3: 谱系活跃度 (Lineage Activity)
  S4: 边际收益 (Marginal Utility)

三状态机：
  EXPLORING → WATCHING → STOPPED
"""

import logging
import math
from typing import List, Optional, Tuple

from akg_agents.op.adaptive_search.success_db import SuccessDB
from akg_agents.op.adaptive_search.evolution_controller.lineage_tree import LineageTree
from akg_agents.op.adaptive_search.evolution_controller.config import (
    ConvergenceConfig,
    ConvergenceSignals,
    SearchState,
    StrategySignal,
)

logger = logging.getLogger(__name__)


class MultiDimensionalConvergenceDetector:
    """多维收敛检测器"""

    def __init__(self, config: ConvergenceConfig):
        self.window = config.convergence_window
        self.perf_threshold = config.perf_improvement_threshold
        self.diversity_threshold = config.diversity_change_threshold
        self.activity_threshold = config.activity_threshold
        self.patience = config.patience
        self.max_total_tasks = config.max_total_tasks
        self.max_time_seconds = config.max_time_seconds

        # 内部状态
        self._state = SearchState.EXPLORING
        self._plateau_count = 0
        self._best_history: List[float] = []
        self._diversity_history: List[float] = []
        self._stop_reason: str = ""
        self._last_checked_history_len: int = 0  # 防止无新数据时重复计算
        self._frozen_diagnostics: Optional[dict] = None  # 收敛触发瞬间的诊断快照

    @property
    def state(self) -> SearchState:
        return self._state

    @property
    def stop_reason(self) -> str:
        return self._stop_reason

    def update(self, db: SuccessDB, lineage_tree: LineageTree) -> None:
        """
        当有新的成功结果时调用，更新历史数据。

        Args:
            db: 成功记录数据库
            lineage_tree: 谱系树
        """
        # 触发停止后冻结收敛历史，避免“触发时刻”与“最终收尾”诊断不一致
        if self._state == SearchState.STOPPED:
            return

        best_record = db.get_best_record()
        if best_record is None:
            return

        self._best_history.append(best_record.gen_time)
        self._diversity_history.append(lineage_tree.get_diversity_index())

    def should_stop(
        self,
        total_submitted: int,
        total_success: int,
        elapsed_seconds: float,
    ) -> Tuple[bool, str]:
        """
        判断是否应该停止搜索。

        Args:
            total_submitted: 已提交的总任务数
            total_success: 成功的任务数
            elapsed_seconds: 已经过的时间（秒）

        Returns:
            (should_stop, reason_string)
        """
        if self._state == SearchState.STOPPED:
            return True, self._stop_reason

        # 安全阀：硬性限制
        if total_submitted >= self.max_total_tasks:
            self._state = SearchState.STOPPED
            self._stop_reason = f"Reached max_total_tasks ({self.max_total_tasks})"
            self._freeze_diagnostics()
            return True, self._stop_reason

        if self.max_time_seconds and elapsed_seconds >= self.max_time_seconds:
            self._state = SearchState.STOPPED
            self._stop_reason = f"Time budget exhausted ({self.max_time_seconds}s)"
            self._freeze_diagnostics()
            return True, self._stop_reason

        # 最少任务保证（防止过早停止）
        min_tasks = max(self.window, 5)
        if total_success < min_tasks:
            self._state = SearchState.EXPLORING
            return False, ""

        # 无新数据时跳过重复计算（防止 plateau_count 虚高）
        current_len = len(self._best_history)
        if current_len == self._last_checked_history_len:
            return self._state == SearchState.STOPPED, self._stop_reason
        self._last_checked_history_len = current_len

        # 计算收敛信号
        signals, signal_window = self._compute_signals(total_success)

        # 状态转移
        new_state, reason = self._transition(signals)
        old_state = self._state
        self._state = new_state

        if old_state != new_state:
            logger.info(
                f"Convergence state: {old_state.value} -> {new_state.value}"
                f" (reason: {reason or 'normal'})"
            )

        if new_state == SearchState.STOPPED:
            self._stop_reason = reason
            self._freeze_diagnostics(
                window=signal_window,
                plateau_count=signals.s1_count,
                s1_plateau=signals.s1_plateau,
                s2_trend=signals.s2_trend,
                s3_activity=signals.s3_activity,
            )
            return True, reason

        return False, ""

    def get_strategy_signal(self) -> StrategySignal:
        """获取当前策略调整信号，传递给 Solver 1 和 Solver 2"""
        if self._state == SearchState.EXPLORING:
            return StrategySignal(reason="normal")

        if self._state == SearchState.WATCHING:
            return StrategySignal(
                tau_multiplier=1.5,
                p_migrate_multiplier=2.0,
                T_insp_multiplier=1.3,
                reason="watching_boost_exploration",
            )

        return StrategySignal(reason="stopped")

    def _compute_signals(self, total_success: int) -> Tuple[ConvergenceSignals, int]:
        """计算三维收敛信号（S1 性能停滞、S2 多样性趋势、S3 谱系活跃度）"""
        w = min(self.window, max(total_success // 2, 1))

        # S1: 性能停滞度
        s1_plateau = False
        if len(self._best_history) > w:
            old_best = self._best_history[-(w + 1)]
            new_best = self._best_history[-1]
            perf_improvement = (old_best - new_best) / (old_best + 1e-9)
            s1_plateau = perf_improvement < self.perf_threshold

        if s1_plateau:
            self._plateau_count += 1
        else:
            self._plateau_count = 0

        # S2: 多样性趋势
        s2_trend = "stable"
        if len(self._diversity_history) > w:
            d_change = (
                self._diversity_history[-1]
                - self._diversity_history[-(w + 1)]
            )
            if d_change < -self.diversity_threshold:
                s2_trend = "declining"
            elif d_change > self.diversity_threshold:
                s2_trend = "increasing"

        # S3: 谱系活跃度（用多样性指数作为近似）
        s3_activity = (
            self._diversity_history[-1]
            if self._diversity_history else 0.0
        )

        signals = ConvergenceSignals(
            s1_plateau=s1_plateau,
            s1_count=self._plateau_count,
            s2_trend=s2_trend,
            s3_activity=s3_activity,
        )
        return signals, w

    def _transition(self, signals: ConvergenceSignals) -> Tuple[SearchState, str]:
        """
        状态转移逻辑。

        EXPLORING: 搜索仍在改善
        WATCHING: 性能停滞但可能还有探索空间
        STOPPED: 确认收敛或达到极限
        """
        # 真正收敛：停滞 + 多样性下降 + 谱系充分探索
        if (signals.s1_plateau
                and signals.s1_count >= self.patience
                and signals.s2_trend == "declining"
                and signals.s3_activity >= self.activity_threshold):
            return SearchState.STOPPED, "converged"

        # 仍在改善
        if not signals.s1_plateau:
            return SearchState.EXPLORING, ""

        # 停滞但仍有探索空间
        return SearchState.WATCHING, ""

    def get_diagnostics(self) -> dict:
        """获取诊断信息"""
        if self._frozen_diagnostics is not None:
            return dict(self._frozen_diagnostics)

        return self._build_diagnostics()

    def _freeze_diagnostics(
        self,
        *,
        window: Optional[int] = None,
        plateau_count: Optional[int] = None,
        s1_plateau: Optional[bool] = None,
        s2_trend: Optional[str] = None,
        s3_activity: Optional[float] = None,
    ) -> None:
        """冻结收敛触发时刻的诊断快照。"""
        if self._frozen_diagnostics is not None:
            return
        self._frozen_diagnostics = self._build_diagnostics(
            window=window,
            plateau_count=plateau_count,
            s1_plateau=s1_plateau,
            s2_trend=s2_trend,
            s3_activity=s3_activity,
        )

    def _build_diagnostics(
        self,
        *,
        window: Optional[int] = None,
        plateau_count: Optional[int] = None,
        s1_plateau: Optional[bool] = None,
        s2_trend: Optional[str] = None,
        s3_activity: Optional[float] = None,
    ) -> dict:
        """构建当前诊断信息（可带覆盖项，用于冻结触发快照）。"""
        w = window if window is not None else min(self.window, max(len(self._best_history) // 2, 1))

        # 重新计算当前信号值（用于诊断展示）
        perf_improvement = None
        if len(self._best_history) > w:
            old_best = self._best_history[-(w + 1)]
            new_best = self._best_history[-1]
            perf_improvement = (old_best - new_best) / (old_best + 1e-9)

        diversity_change = None
        if len(self._diversity_history) > w:
            diversity_change = (
                self._diversity_history[-1]
                - self._diversity_history[-(w + 1)]
            )

        computed_s1_plateau = perf_improvement is not None and perf_improvement < self.perf_threshold
        computed_s2_trend = (
            "declining" if diversity_change is not None and diversity_change < -self.diversity_threshold
            else "increasing" if diversity_change is not None and diversity_change > self.diversity_threshold
            else "stable"
        )
        computed_s3_activity = self._diversity_history[-1] if self._diversity_history else None

        final_s1_plateau = s1_plateau if s1_plateau is not None else computed_s1_plateau
        final_s2_trend = s2_trend if s2_trend is not None else computed_s2_trend
        final_s3_activity = s3_activity if s3_activity is not None else computed_s3_activity
        final_plateau_count = plateau_count if plateau_count is not None else self._plateau_count

        return {
            "state": self._state.value,
            "stop_reason": self._stop_reason,
            "plateau_count": final_plateau_count,
            "patience": self.patience,
            "window": w,
            # S1 相关
            "s1_perf_improvement": round(perf_improvement, 6) if perf_improvement is not None else None,
            "s1_perf_threshold": self.perf_threshold,
            "s1_plateau": final_s1_plateau,
            # S2 相关
            "s2_diversity_change": round(diversity_change, 6) if diversity_change is not None else None,
            "s2_diversity_threshold": self.diversity_threshold,
            "s2_trend": final_s2_trend,
            # S3 相关
            "s3_activity": round(final_s3_activity, 4) if final_s3_activity is not None else None,
            "s3_activity_threshold": self.activity_threshold,
            # 历史
            "current_best_gen_time": self._best_history[-1] if self._best_history else None,
            "current_diversity": round(self._diversity_history[-1], 4) if self._diversity_history else None,
            "total_success_tracked": len(self._best_history),
        }
