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
Evolution Controller Module

进化控制器：外挂式的自适应搜索增强模块。
通过谱系感知的父代选择、Boltzmann 灵感采样和多维收敛检测，
提升 AKG 自适应搜索的性能。
"""

from akg_agents.op.adaptive_search.evolution_controller.controller import (
    EvolutionController,
)
from akg_agents.op.adaptive_search.evolution_controller.config import (
    EvolutionControllerConfig,
    ParentSelectionConfig,
    InspirationSelectionConfig,
    ConvergenceConfig,
    SearchState,
    StrategySignal,
)
from akg_agents.op.adaptive_search.evolution_controller.lineage_tree import (
    LineageTree,
    LineageStats,
)

__all__ = [
    "EvolutionController",
    "EvolutionControllerConfig",
    "ParentSelectionConfig",
    "InspirationSelectionConfig",
    "ConvergenceConfig",
    "SearchState",
    "StrategySignal",
    "LineageTree",
    "LineageStats",
]
