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

from akg_agents.op.config.config_validator import load_config
from akg_agents.core.async_pool.task_pool import TaskPool
from akg_agents.op.langgraph_op.task import LangGraphTask
from akg_agents.core.worker.manager import register_local_worker
from akg_agents.utils.environment_check import check_env_for_task
import asyncio
import os
# 注释掉流式输出，因为需要 session_id（TUI 模式才使用）
os.environ['AKG_AGENTS_STREAM_OUTPUT'] = 'on'


def get_op_name():
    return 'akg_custom'


def get_task_desc():
    return '''
import torch
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, sm_scale):
        super().__init__()
        self.sm_scale = sm_scale

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """
        Scaled Dot-Product Attention implementation using PyTorch's optimized function.
        
        This function computes: Attention(Q, K, V) = softmax(Q @ K^T * scale) @ V
        
        Input tensor layout (required by torch.nn.functional.scaled_dot_product_attention):
            Shape: (B, H, L, D) where:
            - B (Batch): Number of sequences processed in parallel
            - H (Heads): Number of attention heads
            - L (Length): Sequence length (number of tokens)
            - D (Dimension): Embedding dimension per head
            
        Note: The sequence length MUST be at dimension -2 (second to last),
              and embedding dimension MUST be at dimension -1 (last).
        
        Args:
            query: Query tensor of shape (B, H, L, D)
            key: Key tensor of shape (B, H, S, D), where S can differ from L
            value: Value tensor of shape (B, H, S, D)
            
        Returns:
            Attention output of shape (B, H, L, D)
        """
        return torch.nn.functional.scaled_dot_product_attention(
            query, key, value,
            scale=self.sm_scale
        )


def get_inputs():
    """
    Generate input tensors for scaled dot-product attention.
    
    Tensor shape: (B, H, L, D)
        B = 4     : Batch size (number of independent sequences)
        H = 32    : Number of attention heads (multi-head attention)
        L = 1024  : Sequence length (number of tokens in each sequence)
        D = 64    : Head dimension (embedding size per attention head)
    
    Total model dimension = H * D = 32 * 64 = 2048
    
    Note: For inference, we don't need gradient computation, so .requires_grad_()
          is removed. Use torch.no_grad() context or set requires_grad=False.
    """
    B, H, L, D = 4, 32, 1024, 64
    dtype = torch.float32
    
    q = torch.empty((B, H, L, D), dtype=dtype).normal_(mean=0.0, std=0.5)
    k = torch.empty((B, H, L, D), dtype=dtype).normal_(mean=0.0, std=0.5)
    v = torch.empty((B, H, L, D), dtype=dtype).normal_(mean=0.0, std=0.5)
    
    return [q, k, v]


def get_init_inputs():
    """
    Initialize the scaling factor for attention.
    
    Standard scaling factor is 1/sqrt(D) = 1/sqrt(64) ≈ 0.125
    Here we use 0.5 for demonstration purposes.
    """
    sm_scale = 0.5
    return [sm_scale]
'''


async def run_torch_cpu_cpp_single():
    op_name = get_op_name()
    task_desc = get_task_desc()

    task_pool = TaskPool()

    # 注册 LocalWorker（CPU x86_64）
    await register_local_worker([0], backend="cpu", arch="x86_64")

    config = load_config("cpp")  # 使用默认 cpp 配置
    # 也可以指定配置文件路径:
    # config = load_config(config_path="./python/akg_agents/op/config/cpp_coderonly_config.yaml")

    check_env_for_task("torch", "cpu", "cpp", config)

    task = LangGraphTask(
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
    asyncio.run(run_torch_cpu_cpp_single())
