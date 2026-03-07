# Copyright 2026 Huawei Technologies Co., Ltd
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
Skill 自进化系统

模块结构：
  common.py               - 公共类型、工具函数、LLM 输出解析、SKILL.md 写入
  search_log_utils.py     - search_log 模式：从搜索日志收集 + 压缩进化链
  expert_tuning_utils.py  - expert_tuning 模式：从对话目录提取专家调优经验
"""

from .common import TaskRecord, EvolutionStep, CompressedData, SkillWriter
from .search_log_utils import collect as search_log_collect, compress as search_log_compress

__all__ = [
    "TaskRecord",
    "EvolutionStep",
    "CompressedData",
    "search_log_collect",
    "search_log_compress",
    "SkillWriter",
]
