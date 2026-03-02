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

"""Skill 自进化系统：从 adaptive_search 日志中提取优化经验，生成 SKILL.md"""

from .models import TaskRecord, EvolutionStep, CompressedData
from .collector import collect
from .compressor import compress
from .writer import SkillWriter

__all__ = [
    "TaskRecord",
    "EvolutionStep",
    "CompressedData",
    "collect",
    "compress",
    "SkillWriter",
]
