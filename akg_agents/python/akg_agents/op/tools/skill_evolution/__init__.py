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
  error_fix_utils.py      - error_fix 模式：从错误修复记录提取调试经验
  merge_utils.py          - merge_skills 模式：将 evolved skills 按主题合并去重

A/B 测试相关工具已迁移至 examples/kernel_related/skill_evolution/：
  ab_test_utils.py        - A/B 测试工具：运行管理、日志解析、结果收集、Tracking 更新
  run_ab_test.py          - A/B 测试批量运行器
  tracking.md             - 实验结果跟踪文档
"""

from .common import TaskRecord, EvolutionStep, CompressedData, SkillWriter, parse_skill_output, get_default_evolved_dir
from .search_log_utils import (
    collect as search_log_collect,
    compress as search_log_compress,
    to_prompt_vars as search_log_to_prompt_vars,
)
from .expert_tuning_utils import (
    collect as expert_tuning_collect,
    build_timeline,
    to_prompt_vars as expert_tuning_to_prompt_vars,
)
from .error_fix_utils import (
    SuccessfulFixRecord,
    collect as error_fix_collect,
    to_prompt_vars as error_fix_to_prompt_vars,
)

__all__ = [
    "TaskRecord",
    "EvolutionStep",
    "CompressedData",
    "SkillWriter",
    "parse_skill_output",
    "get_default_evolved_dir",
    "search_log_collect",
    "search_log_compress",
    "search_log_to_prompt_vars",
    "expert_tuning_collect",
    "build_timeline",
    "expert_tuning_to_prompt_vars",
    "SuccessfulFixRecord",
    "error_fix_collect",
    "error_fix_to_prompt_vars",
]
