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
Evolve模块 - 进化式算子生成

子模块：
- evolution_core: 核心功能（存储、采样、岛屿模型）
- evolution_utils: 辅助工具（提示加载、通用工具）
- evolution_processors: 处理器类（初始化、任务创建、结果处理）
- runner_manager: Runner管理（配置、CLI、批量执行）
- result_collector: 实时结果收集器
"""

__all__ = [
    'evolution_core',
    'evolution_utils',
    'evolution_processors',
    'runner_manager',
    'result_collector'
]

