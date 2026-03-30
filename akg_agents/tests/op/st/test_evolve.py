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

"""Evolve 端到端系统测试 — ReLU (Triton Ascend)"""

import pytest
from akg_agents.op.evolve import evolve
from akg_agents.core.async_pool.task_pool import TaskPool
from akg_agents.op.config.config_validator import load_config
from akg_agents.core.worker.manager import register_local_worker
from akg_agents.utils.task_label import resolve_task_label
from ..utils import get_device_id

RELU_TASK_DESC = """\
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x)

batch_size = 16
dim = 16384

def get_inputs():
    x = torch.rand(batch_size, dim, device='npu')
    return [x]

def get_init_inputs():
    return []
"""


@pytest.mark.level0
@pytest.mark.torch
@pytest.mark.triton
@pytest.mark.ascend
@pytest.mark.ascend910b4
@pytest.mark.asyncio
async def test_evolve_relu_ascend910b4():
    """端到端测试: evolve 生成 ReLU (triton_ascend, ascend910b4)"""
    op_name = "akg_agents_relu"
    dsl = "triton_ascend"
    framework = "torch"
    backend = "ascend"
    arch = "ascend910b4"

    device_id = get_device_id()
    await register_local_worker([device_id], backend=backend, arch=arch)

    config = load_config(dsl=dsl, backend=backend)
    config["task_label"] = resolve_task_label(op_name=op_name, parallel_index=1)
    config["max_step"] = 5

    task_pool = TaskPool(max_concurrency=2)

    result = await evolve(
        op_name=op_name,
        task_desc=RELU_TASK_DESC,
        dsl=dsl,
        framework=framework,
        backend=backend,
        arch=arch,
        config=config,
        task_pool=task_pool,
        max_rounds=2,
        parallel_num=2,
    )

    assert isinstance(result, dict)
    assert "successful_tasks" in result
    assert "total_tasks" in result
    assert result["total_tasks"] >= 1

    if result["successful_tasks"] > 0:
        best = result.get("best_implementations", [])
        assert len(best) > 0
        assert "impl_code" in best[0]
