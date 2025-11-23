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

import asyncio
# 导入evolve函数和必要的模块
from ai_kernel_generator.core.evolve import evolve
from ai_kernel_generator.core.async_pool.task_pool import TaskPool
from ai_kernel_generator.config.config_validator import load_config
from ai_kernel_generator.utils.environment_check import check_env_for_task
from ai_kernel_generator.tools.single_evolve_runner import EvolveConfig, print_evolve_config, print_evolution_result
from ai_kernel_generator import get_project_root
from pathlib import Path
from ai_kernel_generator.core.worker.manager import register_local_worker


def get_op_name():
    return 'aikg_relu'


def get_task_desc():
    return '''
import torch
import torch.nn as nn
import torch_npu

class Model(nn.Module):
    """
    Simple model that performs a ReLU activation.
    """
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies ReLU activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with ReLU applied, same shape as input.
        """
        return torch.relu(x)

batch_size = 16
dim = 16384

def get_inputs():
    x = torch.rand(batch_size, dim, device='npu')
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed
'''


async def run_torch_evolve_triton_ascend():
    """运行Triton Ascend进化示例 (使用 WorkerManager 新架构)"""

    # 创建配置对象并设置硬编码参数
    config = EvolveConfig()

    # 基础配置
    config.dsl = "triton_ascend"
    config.framework = "torch"
    config.backend = "ascend"
    config.arch = "ascend910b4"

    # 进化参数
    config.max_rounds = 3
    config.parallel_num = 4

    # 岛屿模型参数
    config.num_islands = 2
    config.migration_interval = 2
    config.elite_size = 5
    config.parent_selection_prob = 0.5

    # 设备配置
    config.device_list = [0, 1, 2, 3]

    # 配置文件路径
    config.config_path = str(Path(get_project_root()) / "config" / "vllm_triton_ascend_evolve_config.yaml")

    # 选择要运行的任务
    config.op_name = get_op_name()
    config.task_desc = get_task_desc()

    # 打印配置信息
    print_evolve_config(config.op_name, config)

    # 初始化资源池
    task_pool = TaskPool(max_concurrency=config.parallel_num)
    
    # 新写法：一行代码注册 LocalWorker
    await register_local_worker(config.device_list, backend=config.backend, arch=config.arch)

    # 加载配置并检查环境
    # 注意：如果是 triton_ascend，需要确保 config/vllm_triton_ascend_evolve_config.yaml 存在
    # 如果不存在，可以使用默认加载方式
    try:
        loaded_config = load_config(config_path=config.config_path)
    except ValueError:
        print(f"Config file {config.config_path} not found, using default.")
        loaded_config = load_config(dsl=config.dsl, backend=config.backend)

    check_env_for_task(config.framework, config.backend, config.dsl, loaded_config)

    # 调用evolve函数
    print("开始进化过程...")
    evolution_result = await evolve(
        op_name=config.op_name,
        task_desc=config.task_desc,
        dsl=config.dsl,
        framework=config.framework,
        backend=config.backend,
        arch=config.arch,
        config=loaded_config,
        # device_pool=device_pool, # 新写法：不传 device_pool，内部会自动通过 WorkerManager 获取资源
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
    asyncio.run(run_torch_evolve_triton_ascend())

