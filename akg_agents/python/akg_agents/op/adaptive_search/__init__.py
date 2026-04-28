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
Adaptive Search Module

基于 UCB 选择策略的自适应搜索框架，用于替代岛屿/精英进化算法。
"""

from akg_agents.op.adaptive_search.adaptive_search import (
    adaptive_search,
    adaptive_search_from_config,
    load_adaptive_search_config
)
from akg_agents.op.adaptive_search.success_db import SuccessDB, SuccessRecord
from akg_agents.op.adaptive_search.ucb_selector import UCBParentSelector
from akg_agents.op.adaptive_search.task_pool import (
    AsyncTaskPool, 
    PendingTask, 
    TaskResult,
    TaskWrapper
)
from akg_agents.op.adaptive_search.task_generator import (
    TaskGenerator,
    TaskGeneratorConfig
)
from akg_agents.op.adaptive_search.controller import (
    AdaptiveSearchController,
    SearchConfig
)

__all__ = [
    # 主入口
    'adaptive_search',
    'adaptive_search_from_config',
    'load_adaptive_search_config',
    
    # 数据库
    'SuccessDB',
    'SuccessRecord', 
    
    # 选择器
    'UCBParentSelector',
    
    # 任务池
    'AsyncTaskPool',
    'PendingTask',
    'TaskResult',
    'TaskWrapper',
    
    # 生成器
    'TaskGenerator',
    'TaskGeneratorConfig',
    
    # 控制器
    'AdaptiveSearchController',
    'SearchConfig',

    # 进化控制器
    'EvolutionController',
    'EvolutionControllerConfig',
]


# 进化控制器（延迟导入，避免循环依赖）
def __getattr__(name):
    if name in ('EvolutionController', 'EvolutionControllerConfig'):
        from akg_agents.op.adaptive_search.evolution_controller import (
            EvolutionController,
            EvolutionControllerConfig,
        )
        return {'EvolutionController': EvolutionController,
                'EvolutionControllerConfig': EvolutionControllerConfig}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

