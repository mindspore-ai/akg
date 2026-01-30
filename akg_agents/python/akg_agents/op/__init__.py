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
op/ - 算子场景专用层

包含算子生成相关的所有专用逻辑：
- op/config/: 算子配置文件
- op/langgraph/: 算子 LangGraph 状态、节点、路由、任务
- op/workflows/: 算子工作流定义
- op/evolve.py: 进化式算子生成
- op/adaptive_search/: 自适应搜索框架
- op/utils/: 算子专用工具
- op/tools/: 算子专用工具函数
- op/resources/: 算子专用资源文件

基于 core_v2/ 通用框架构建，对应 CLI 命令: akg_cli op
"""

# Note: Imports are intentionally lazy to avoid circular import issues
# Use explicit imports when needed:
#   from akg_agents.op.evolve import evolve
#   from akg_agents.op.config import ConfigValidator, load_config
#   from akg_agents.op.adaptive_search import adaptive_search

__all__ = []

