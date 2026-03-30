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
Skill 选择 ST — 需要 LLM，验证真实选择准确性

覆盖：
- LLM 对不同算子类型选择正确的 guide（看护核心匹配准确性）
- initial 阶段不含 case，debug 阶段含 fundamental
- exclude_skill_names / force_skill_names 运行时生效
- 选择耗时 baseline
"""

import time
import pytest
import logging

logger = logging.getLogger(__name__)

RELU_TASK = """\
import torch
import torch.nn as nn
class Model(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x)
def get_inputs():
    return [torch.randn(16, 16384)]
def get_init_inputs():
    return []
"""

MATMUL_TASK = """\
import torch
import torch.nn as nn
class Model(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, a, b):
        return torch.matmul(a, b)
def get_inputs():
    return [torch.randn(1024, 1024), torch.randn(1024, 1024)]
def get_init_inputs():
    return []
"""


@pytest.fixture(scope="module")
def kernel_gen():
    from akg_agents.op.agents.kernel_gen import KernelGen
    return KernelGen()


MUST_MATCH = [
    ("relu", RELU_TASK, "elementwise"),
    ("matmul", MATMUL_TASK, "matmul"),
]


@pytest.mark.level2
@pytest.mark.use_model
@pytest.mark.asyncio
@pytest.mark.parametrize("op_name,task_desc,expected_keyword", MUST_MATCH,
                         ids=[c[0] for c in MUST_MATCH])
async def test_guide_selection_accuracy(kernel_gen, op_name, task_desc, expected_keyword):
    """看护：典型算子必须选中对应 guide"""
    t0 = time.time()
    skills = await kernel_gen._select_skills_by_stage(
        stage="initial", op_name=op_name, task_desc=task_desc,
        verifier_error="", dsl="triton_ascend", backend="ascend", framework="torch",
    )
    elapsed = time.time() - t0
    logger.info(f"[{op_name}] {elapsed:.1f}s, {len(skills)} skills")

    guide_names = [s.name for s in skills if getattr(s, "category", "") == "guide"]
    assert any(expected_keyword in n for n in guide_names), \
        f"Expected guide containing '{expected_keyword}', got: {guide_names}"


@pytest.mark.level2
@pytest.mark.use_model
@pytest.mark.asyncio
async def test_initial_stage_no_case(kernel_gen):
    """initial 阶段不应包含 case"""
    skills = await kernel_gen._select_skills_by_stage(
        stage="initial", op_name="relu", task_desc=RELU_TASK,
        verifier_error="", dsl="triton_ascend", backend="ascend", framework="torch",
    )
    categories = {getattr(s, "category", "") for s in skills}
    assert "case" not in categories


@pytest.mark.level2
@pytest.mark.use_model
@pytest.mark.asyncio
async def test_debug_stage_has_fundamentals(kernel_gen):
    """debug 阶段仍含 fundamental"""
    skills = await kernel_gen._select_skills_by_stage(
        stage="debug", op_name="relu", task_desc=RELU_TASK,
        verifier_error="RuntimeError: shape mismatch",
        dsl="triton_ascend", backend="ascend", framework="torch",
    )
    categories = {getattr(s, "category", "") for s in skills}
    assert "fundamental" in categories or "reference" in categories


@pytest.mark.level2
@pytest.mark.use_model
@pytest.mark.asyncio
async def test_exclude_skill_names():
    """exclude_skill_names 排除指定 skill"""
    from akg_agents.op.agents.kernel_gen import KernelGen
    kg = KernelGen()
    kg.exclude_skill_names = ["triton-ascend-optimization"]
    skills = await kg._select_skills_by_stage(
        stage="initial", op_name="relu", task_desc=RELU_TASK,
        verifier_error="", dsl="triton_ascend", backend="ascend", framework="torch",
    )
    assert "triton-ascend-optimization" not in [s.name for s in skills]


@pytest.mark.level2
@pytest.mark.use_model
@pytest.mark.asyncio
async def test_force_skill_names():
    """force_skill_names 强制导入指定 skill"""
    from akg_agents.op.agents.kernel_gen import KernelGen
    kg = KernelGen()
    kg.force_skill_names = ["triton-ascend-case-reduction-amax-medium"]
    skills = await kg._select_skills_by_stage(
        stage="initial", op_name="relu", task_desc=RELU_TASK,
        verifier_error="", dsl="triton_ascend", backend="ascend", framework="torch",
    )
    assert "triton-ascend-case-reduction-amax-medium" in [s.name for s in skills]


@pytest.mark.level2
@pytest.mark.use_model
@pytest.mark.asyncio
async def test_selection_latency_baseline(kernel_gen):
    """记录选择耗时 baseline"""
    t0 = time.time()
    await kernel_gen._select_skills_by_stage(
        stage="initial", op_name="gelu", task_desc=RELU_TASK.replace("relu", "gelu"),
        verifier_error="", dsl="triton_ascend", backend="ascend", framework="torch",
    )
    elapsed = time.time() - t0
    logger.info(f"Skill selection latency: {elapsed:.2f}s")
    if elapsed > 30:
        logger.warning(f"Selection took {elapsed:.1f}s (>30s)")
