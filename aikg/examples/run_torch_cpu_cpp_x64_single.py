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

from ai_kernel_generator.config.config_validator import load_config
from ai_kernel_generator.core.async_pool.task_pool import TaskPool
from ai_kernel_generator.core.task import Task
from ai_kernel_generator.core.worker.manager import register_local_worker
from ai_kernel_generator.utils.environment_check import check_env_for_task
import asyncio
import os
os.environ['AIKG_STREAM_OUTPUT'] = 'on'


def get_op_name():
    return 'relu_cpu_cpp_x64'


def get_task_desc():
    return '''
import torch
import torch.nn as nn


class Model(nn.Module):
    """
    ReLU 激活函数模型（CPU）
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算 ReLU 激活函数
        Args:
            x: 输入张量
        Returns:
            ReLU 激活后的张量
        """
        return torch.relu(x)


batch_size = 16
dim = 16384


def get_inputs():
    x = torch.randn(batch_size, dim, dtype=torch.float32, device='cpu')
    return [x]


def get_init_inputs():
    return []  # No special initialization inputs needed
'''


async def run_torch_cpu_cpp_x64_single():
    op_name = get_op_name()
    task_desc = get_task_desc()

    task_pool = TaskPool()
    
    # 注册 LocalWorker
    await register_local_worker([0], backend="cpu", arch="x86_64")
    
    # C++ CPU 配置
    config = load_config(config_path="./python/ai_kernel_generator/config/default_cpp_config.yaml")

    check_env_for_task("torch", "cpu", "cpp", config)

    task = Task(
        op_name=op_name,
        task_desc=task_desc,
        task_id="0",
        dsl="cpp",
        backend="cpu",
        arch="x86_64",
        config=config,
        framework="torch",
        workflow="coder_only_workflow"
    )

    task_pool.create_task(task.run)
    results = await task_pool.wait_all()
    for op_name, result, _ in results:
        if result:
            print(f"Task {op_name} passed")
        else:
            print(f"Task {op_name} failed")


if __name__ == "__main__":
    asyncio.run(run_torch_cpu_cpp_x64_single())
