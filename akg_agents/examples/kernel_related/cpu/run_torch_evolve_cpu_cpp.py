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
进化搜索示例 - CPU C++ (x86_64)

小 shape ReLU 算子的进化搜索，用于 CPU x86_64 + C++ DSL。

用法:
  cd akg/aikg && source env.sh
  python examples/kernel_related/run_torch_evolve_cpu_cpp.py
"""

import os
import asyncio
from akg_agents.op.evolve import evolve
from akg_agents.core.async_pool.task_pool import TaskPool
from akg_agents.op.config.config_validator import load_config
from akg_agents.utils.environment_check import check_env_for_task
from akg_agents.op.utils.evolve.runner_manager import RunnerConfig, print_evolve_config, print_evolution_result
from akg_agents.core.worker.manager import register_local_worker

# os.environ['AKG_AGENTS_STREAM_OUTPUT'] = 'on'

def get_op_name():
    return 'akg_agents_relu'


def get_task_desc():
    return '''
import torch
import torch.nn as nn


class Model(nn.Module):
    """
    ReLU 激活函数模型（CPU 小 shape）
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


async def run_torch_evolve_cpu_cpp():
    """运行 CPU C++ 进化示例"""

    config = RunnerConfig()

    # 基础配置
    config.dsl = "cpp"
    config.framework = "torch"
    config.backend = "cpu"
    config.arch = "x86_64"

    # 进化参数（小规模测试）
    config.max_rounds = 2
    config.parallel_num = 2

    # 岛屿模型参数（单岛屿，简单模式）
    config.num_islands = 1
    config.migration_interval = 0
    config.elite_size = 0
    config.parent_selection_prob = 0.5

    # 设备配置（CPU 只用一个逻辑设备）
    config.device_list = [0]

    # 任务配置
    config.op_name = get_op_name()
    config.task_desc = get_task_desc()

    # 打印配置信息
    print_evolve_config(config.op_name, config)

    # 初始化
    task_pool = TaskPool(max_concurrency=config.parallel_num)
    await register_local_worker(config.device_list, backend=config.backend, arch=config.arch)

    # 加载配置
    loaded_config = load_config("cpp")
    check_env_for_task(config.framework, config.backend, config.dsl, loaded_config)

    # 运行进化
    print("开始进化过程...")
    evolution_result = await evolve(
        op_name=config.op_name,
        task_desc=config.task_desc,
        dsl=config.dsl,
        framework=config.framework,
        backend=config.backend,
        arch=config.arch,
        config=loaded_config,
        task_pool=task_pool,
        max_rounds=config.max_rounds,
        parallel_num=config.parallel_num,
        num_islands=config.num_islands,
        migration_interval=config.migration_interval,
        elite_size=config.elite_size,
        parent_selection_prob=config.parent_selection_prob
    )

    # 打印结果
    print_evolution_result(evolution_result, config)


if __name__ == "__main__":
    asyncio.run(run_torch_evolve_cpu_cpp())
