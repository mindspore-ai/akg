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
CUDA C → PyTorch 转换示例

将 CUDA C Kernel 代码转换为等价的 PyTorch 原生实现。

使用方法:
    python examples/run_cudac_to_torch_single.py
"""

from akg_agents.op.config.config_validator import load_config
from akg_agents.core.async_pool.task_pool import TaskPool
from akg_agents.op.langgraph_op.task import LangGraphTask
from akg_agents.core.worker.manager import register_local_worker
from akg_agents.utils.environment_check import check_env_for_task
import asyncio
import os
os.environ['AKG_AGENTS_STREAM_OUTPUT'] = 'on'


def get_op_name():
    return 'cudac_relu_to_torch'


def get_task_desc():
    """CUDA C ReLU Kernel (需要转换为 PyTorch)"""
    return '''
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA C kernel 代码
cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void relu_kernel(const float* input, float* output, int n_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elements) {
        output[idx] = fmaxf(input[idx], 0.0f);
    }
}

torch::Tensor relu_cuda(torch::Tensor input) {
    auto output = torch::zeros_like(input);
    int n_elements = input.numel();
    
    const int block_size = 256;
    const int num_blocks = (n_elements + block_size - 1) / block_size;
    
    relu_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        n_elements
    );
    
    return output;
}
"""

cpp_source = "torch::Tensor relu_cuda(torch::Tensor input);"

# JIT 编译 CUDA 扩展
relu_module = load_inline(
    name="relu_cuda",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["relu_cuda"],
    verbose=False,
)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return relu_module.relu_cuda(x)


batch_size = 16
dim = 16384


def get_inputs():
    x = torch.randn(batch_size, dim, dtype=torch.float32, device='cuda')
    return [x]


def get_init_inputs():
    return []
'''


async def run_cudac_to_torch_single():
    op_name = get_op_name()
    task_desc = get_task_desc()

    task_pool = TaskPool()
    
    # 注册 LocalWorker
    await register_local_worker([0], backend="cuda", arch="a100")
    
    # 加载 torch DSL 配置
    config = load_config(config_path="./python/akg_agents/config/default_torch_config.yaml")

    # 环境检查
    check_env_for_task("torch", "cuda", "torch", config)

    # 创建任务
    task = LangGraphTask(
        op_name=op_name,
        task_desc=task_desc,
        task_id="0",
        dsl="torch",           # 目标 DSL: PyTorch
        backend="cuda",
        arch="a100",
        config=config,
        framework="torch",     # 输入框架: torch (含 CUDA C 扩展)
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
    asyncio.run(run_cudac_to_torch_single())

