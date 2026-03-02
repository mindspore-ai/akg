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

"""Skill 自进化系统 - 数据模型"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class TaskRecord:
    """一个搜索任务的核心记录"""
    task_id: str
    parent_id: str = ""
    generation: int = 0
    code: str = ""
    speedup: float = 0.0
    gen_time: float = float("inf")


@dataclass
class EvolutionStep:
    """进化链中的一步：父代 → 子代 diff"""
    parent_id: str
    child_id: str
    parent_speedup: float = 0.0
    child_speedup: float = 0.0
    parent_gen_time: float = float("inf")
    child_gen_time: float = float("inf")
    code_diff: str = ""


@dataclass
class CompressedData:
    """压缩后的数据，直接注入 LLM prompt"""
    op_name: str = ""
    dsl: str = ""
    backend: str = ""
    arch: str = ""
    best_task_id: str = ""
    best_speedup: float = 0.0
    best_gen_time: float = float("inf")
    best_code: str = ""
    evolution_chains: List[EvolutionStep] = field(default_factory=list)
    performance_summary: str = ""
    total_tasks: int = 0
    success_count: int = 0
