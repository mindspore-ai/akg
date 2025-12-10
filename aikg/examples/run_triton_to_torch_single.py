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
Triton → PyTorch 转换示例

将 Triton Kernel 代码转换为等价的 PyTorch 原生实现。

使用方法:
    python examples/run_triton_to_torch_single.py
"""

from ai_kernel_generator.config.config_validator import load_config
from ai_kernel_generator.core.async_pool.task_pool import TaskPool
from ai_kernel_generator.core.task import Task
from ai_kernel_generator.core.worker.manager import register_local_worker
from ai_kernel_generator.utils.environment_check import check_env_for_task
import asyncio
import os
os.environ['AIKG_STREAM_OUTPUT'] = 'on'


def get_op_name():
    return 'triton_relu_to_torch'


def get_task_desc():
    """Triton ReLU Kernel (需要转换为 PyTorch)"""
    return '''
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def relu_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    output = tl.maximum(x, 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = torch.empty_like(x)
        n_elements = x.numel()
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        relu_kernel[grid](x, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
        return output


batch_size = 16
dim = 16384


def get_inputs():
    x = torch.randn(batch_size, dim, dtype=torch.float32, device='cuda')
    return [x]


def get_init_inputs():
    return []
'''


async def run_triton_to_torch_single():
    op_name = get_op_name()
    task_desc = get_task_desc()

    task_pool = TaskPool()
    
    # 注册 LocalWorker
    await register_local_worker([0], backend="cuda", arch="a100")
    
    # 加载 torch DSL 配置
    config = load_config(config_path="./python/ai_kernel_generator/config/default_torch_config.yaml")

    # 环境检查
    check_env_for_task("torch", "cuda", "torch", config)

    # 创建任务
    task = Task(
        op_name=op_name,
        task_desc=task_desc,
        task_id="0",
        dsl="torch",           # 目标 DSL: PyTorch
        backend="cuda",
        arch="a100",
        config=config,
        framework="torch",     # 输入框架: torch (含 Triton)
        workflow="coder_only_workflow"
    )

    task_pool.create_task(task.run)
    results = await task_pool.wait_all()
    
    for op_name, result, task_info in results:
        if result:
            print(f"Task {op_name} passed")
            print("\nGenerated PyTorch code:")
            print("-" * 50)
            print(task_info.get('coder_code', ''))
        else:
            print(f"Task {op_name} failed")


if __name__ == "__main__":
    asyncio.run(run_triton_to_torch_single())
