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
Evolution Controller Configuration

进化控制器的所有配置数据类。
"""

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class SearchState(Enum):
    """搜索状态"""
    EXPLORING = "exploring"
    WATCHING = "watching"
    STOPPED = "stopped"


@dataclass
class StrategySignal:
    """收敛检测器向其他求解器发出的策略调整信号"""
    tau_multiplier: float = 1.0
    p_exploit_override: Optional[float] = None
    p_migrate_multiplier: float = 1.0
    T_insp_multiplier: float = 1.0
    reason: str = ""


@dataclass
class ConvergenceSignals:
    """三维收敛信号"""
    s1_plateau: bool = False
    s1_count: int = 0
    s2_trend: str = "stable"  # 'declining' | 'stable' | 'increasing'
    s3_activity: float = 0.0


@dataclass
class ParentSelectionConfig:
    """Solver 1: 谱系感知父代选择配置"""
    tau_start: float = 1.5
    tau_end: float = 0.5
    c_local_start: float = 1.5
    c_local_end: float = 0.5
    p_exploit: float = 0.7
    quality_bonus_strength: float = 0.3


@dataclass
class InspirationSelectionConfig:
    """Solver 2: Boltzmann + 迁移灵感选择配置"""
    inspiration_sample_num: int = 3
    p_migrate: float = 0.2
    foreign_elite_num: int = 2
    T_start: float = 1.5
    T_end: float = 0.5


@dataclass
class ConvergenceConfig:
    """Solver 3: 多维收敛检测配置"""
    convergence_window: int = 5
    perf_improvement_threshold: float = 0.02
    diversity_change_threshold: float = 0.05
    activity_threshold: float = 0.8
    patience: int = 2
    max_total_tasks: int = 100
    max_time_seconds: Optional[float] = None


@dataclass
class EvolutionControllerConfig:
    """进化控制器总配置"""
    parent_selection: ParentSelectionConfig = field(default_factory=ParentSelectionConfig)
    inspiration: InspirationSelectionConfig = field(default_factory=InspirationSelectionConfig)
    convergence: ConvergenceConfig = field(default_factory=ConvergenceConfig)


def _load_yaml(path: str) -> Dict[str, Any]:
    """加载 YAML 文件"""
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        logger.warning("PyYAML not installed, using defaults for evolution controller config")
        return {}
    except FileNotFoundError:
        logger.info(f"Evolution controller config not found at {path}, using defaults")
        return {}


def load_evolution_controller_config(
    config_path: Optional[str] = None,
    max_total_tasks: int = 100,
) -> EvolutionControllerConfig:
    """
    从 YAML 文件加载进化控制器配置。

    Args:
        config_path: 配置文件路径，为空时使用默认路径（同目录下的 evolution_controller_config.yaml）
        max_total_tasks: 从搜索配置继承的最大任务数

    Returns:
        EvolutionControllerConfig 实例
    """
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(__file__), "evolution_controller_config.yaml"
        )

    data = _load_yaml(config_path)

    ps = data.get("parent_selection", {})
    insp = data.get("inspiration", {})
    conv = data.get("convergence", {})

    return EvolutionControllerConfig(
        parent_selection=ParentSelectionConfig(
            tau_start=ps.get("tau_start", 1.5),
            tau_end=ps.get("tau_end", 0.5),
            c_local_start=ps.get("c_local_start", 1.5),
            c_local_end=ps.get("c_local_end", 0.5),
            p_exploit=ps.get("p_exploit", 0.7),
            quality_bonus_strength=ps.get("quality_bonus_strength", 0.3),
        ),
        inspiration=InspirationSelectionConfig(
            inspiration_sample_num=insp.get("sample_num", 3),
            p_migrate=insp.get("p_migrate", 0.2),
            foreign_elite_num=insp.get("foreign_elite_num", 2),
            T_start=insp.get("T_start", 1.5),
            T_end=insp.get("T_end", 0.5),
        ),
        convergence=ConvergenceConfig(
            convergence_window=conv.get("convergence_window", 5),
            perf_improvement_threshold=conv.get("perf_improvement_threshold", 0.02),
            diversity_change_threshold=conv.get("diversity_change_threshold", 0.05),
            activity_threshold=conv.get("activity_threshold", 0.8),
            patience=conv.get("patience", 2),
            max_total_tasks=max_total_tasks,
            max_time_seconds=conv.get("max_time_seconds", None),
        ),
    )
