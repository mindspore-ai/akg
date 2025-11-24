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
手写优化建议利用分析工具

功能：分析AI生成代码是否正确利用了优化建议，对比性能

使用方法：
python examples/handwrite_optimization_analyzer.py
"""

import os
import sys
import asyncio
from ai_kernel_generator.config.config_validator import load_config
from ai_kernel_generator.core.task import Task
from ai_kernel_generator.core.async_pool.device_pool import DevicePool
from ai_kernel_generator.core.verifier.kernel_verifier import KernelVerifier


def get_task_desc():
    """获取实际任务的torch代码"""
    task_desc = '''
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input_tensor):
        # Mean along dimension 1
        return torch.mean(input_tensor, 1)


def get_inputs():
    # Batch size: 2000
    # Hidden dimension: 4096
    input_tensor = torch.randn(2000, 4096, dtype=torch.float32)
    return [input_tensor]


def get_init_inputs():
    # No parameters required
    return []
'''
    return task_desc


def get_handwrite_optimization_example():
    """获取手写优化示例"""
    example_name = "sum_fused_elemwise"

    torch_code = '''
import torch
import torch_npu

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, bias):
        t1 = x * 2.0
        t2 = t1 + bias
        t3 = torch.sigmoid(t2)
        t4 = torch.sum(t3, dim=-1, keepdim=True)
        
        return t4

def get_init_inputs():
    return []

def get_inputs():
    x = torch.randn(1000, 8192)
    bias = torch.randn(8192)
    return [x, bias]
'''

    triton_code = '''
import torch
import triton
import triton.language as tl

@triton.autotune(configs=[
        triton.Config({'BLOCK_SIZE_M': 50, 'SUB_BLOCK_SIZE_M': 25, 'BLOCK_SIZE_N': 256}),
        triton.Config({'BLOCK_SIZE_M': 25, 'SUB_BLOCK_SIZE_M': 4, 'BLOCK_SIZE_N': 2048}),
        triton.Config({'BLOCK_SIZE_M': 25, 'SUB_BLOCK_SIZE_M': 25, 'BLOCK_SIZE_N': 256}),
        triton.Config({'BLOCK_SIZE_M': 20, 'SUB_BLOCK_SIZE_M': 20, 'BLOCK_SIZE_N': 256}),
    ],
    key=['M', 'N']
)
@triton.jit
def sum_fused_elemwise_kernel(
    x_ptr, bias_ptr, output_ptr,
    M, N,
    stride_x_m, stride_x_n,
    stride_out_m,
    BLOCK_SIZE_M: tl.constexpr, 
    SUB_BLOCK_SIZE_M: tl.constexpr, 
    BLOCK_SIZE_N: tl.constexpr 
):
    pid = tl.program_id(0)

    for m_start in range(0, BLOCK_SIZE_M, SUB_BLOCK_SIZE_M):
        m_offsets = pid * BLOCK_SIZE_M + m_start + tl.arange(0, SUB_BLOCK_SIZE_M)
        mmask = m_offsets < M

        acc = tl.zeros([SUB_BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)
        for n_start in range(0, N, BLOCK_SIZE_N):
            n_offsets = n_start + tl.arange(0, BLOCK_SIZE_N)
            nmask = n_offsets < N

            mask = (mmask[:, None]) & (nmask[None, :])
            offsets = m_offsets[:,None] * stride_x_m + n_offsets[None,:] * stride_x_n

            x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
            bias = tl.load(bias_ptr + n_offsets, mask=nmask, other=0.0)
            
            t1 = x * 2.0
            t2 = t1 + bias
            t3 = tl.sigmoid(t2)
            
            acc += tl.where(mask, t3, 0.0)

        total_sum = tl.sum(acc, axis=1)

        output_ptrs = output_ptr + m_offsets * stride_out_m
        tl.store(output_ptrs, total_sum, mask=mmask)


def sum_fused_elemwise_triton_torch(x, bias):
    assert x.dim() == 2
    assert bias.dim() == 1
    assert x.shape[1] == bias.shape[0]
    
    M, N = x.shape
    
    output = torch.empty((M, 1), device=x.device, dtype=x.dtype)
    
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']), )
    
    sum_fused_elemwise_kernel[grid](
        x, bias, output,
        M, N,
        x.stride(0), x.stride(1),
        output.stride(0)
    )
    
    return output
'''

    improvement_doc = '''
# 任务特征
**操作类型**：reduction+elementwise融合，reduce轴为最后一根轴；2D Tensor输入，1D Tensor输出
**数据尺寸**：(1000, 8192), (8192,) -> 算子规格较大
**数据类型**：输入输出均为float32类型
**任务特点**：逐元素与归约计算融合，先进行向量化逐元素操作，再沿列方向求和归约；通过行列双向分块优化内存访问，利用for循环处理行列分块；采用Auto-Tuning机制动态选择分块配置。

# 关键优化策略

## 优化1: 调整行切分策略
```python
pid = tl.program_id(0)
for m_start in range(0, BLOCK_SIZE_M, SUB_BLOCK_SIZE_M):
    m_offsets = pid * BLOCK_SIZE_M + m_start + tl.arange(0, SUB_BLOCK_SIZE_M)
```
- 每个kernel计算多行（BLOCK_SIZE_M），减少总线程块数量
- kernel内二次切分（SUB_BLOCK_SIZE_M），避免超过硬件缓存

## 优化2: AutoTune配置优化
- grid等于核数，SUB切分不含尾块时性能最优
- 避免尾块计算，优先采用不含尾块的切分方案

## 优化3: 延迟归约操作
```python
acc = tl.zeros([SUB_BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)
for n_start in range(0, N, BLOCK_SIZE_N):
    acc += tl.where(mask, t3, 0.0)
total_sum = tl.sum(acc, axis=1)
```
- 引入累加器矩阵，将循环内多次小归约合并为循环后一次批量归约
'''
    
    return example_name, torch_code, triton_code, improvement_doc


async def run_analysis(
    op_name="custom_op",
    task_desc=None,
    example_name=None,
    torch_code=None,
    triton_reference_code=None,
    improvement_doc=None,
    device_id=0,
    run_times=50,
    warmup_times=5
):
    """运行优化建议利用分析
    
    Args:
        op_name: 算子名称
        task_desc: 实际任务的torch代码（用于Task执行）
        example_name: 优化示例名称
        torch_code: 优化示例的torch代码（用于handwrite_suggestions）
        triton_reference_code: 手写的triton参考代码
        improvement_doc: 优化建议文档
        device_id: 设备ID
        run_times: 性能测试运行次数
        warmup_times: 性能测试预热次数
    """
    print("="*80)
    print("手写优化建议利用分析工具")
    print("="*80)
    print(f"\n📋 算子: {op_name}")
    print(f"🎯 目标: 对比有无优化建议的性能差异\n")
    
    # 1. 加载配置
    config = load_config(config_path="./python/ai_kernel_generator/config/vllm_triton_ascend_evolve_config.yaml")
    device_pool = DevicePool(device_list=[device_id])
    
    # 2. 不带优化建议的测试
    print("🚀 步骤1: 不带优化建议的代码生成")
    print("-" * 80)
    
    task_no_hint = Task(
        op_name=op_name,
        task_desc=task_desc,
        task_id="no_hint_001",
        backend="ascend",
        arch="ascend910b4",
        dsl="triton_ascend",
        config=config,
        device_pool=device_pool,
        framework="torch",
        task_type="profile",
        workflow="default_workflow",
        handwrite_suggestions=[]
    )
    
    _, success_no_hint, task_info_no_hint = await task_no_hint.run()
    
    if not success_no_hint:
        print("❌ 不带优化建议的Task执行失败")
        return None
    
    print("✅ 不带优化建议的Task执行完成\n")
    
    no_hint_code = task_info_no_hint.get('coder_code', '')
    no_hint_gen_time = task_info_no_hint.get('profile_res', {}).get('gen_time', float('inf'))
    no_hint_base_time = task_info_no_hint.get('profile_res', {}).get('base_time', 0.0)
    no_hint_speedup = task_info_no_hint.get('profile_res', {}).get('speedup', 0.0)
    
    # 3. 带优化建议的测试
    print("🚀 步骤2: 带优化建议的代码生成")
    print("-" * 80)
    
    handwrite_suggestions = [{
        'name': example_name,
        'torch_code': torch_code,
        'triton_code': triton_reference_code,
        'improvement': improvement_doc
    }]
    
    task_with_hint = Task(
        op_name=op_name,
        task_desc=task_desc,
        task_id="with_hint_001",
        backend="ascend",
        arch="ascend910b4",
        dsl="triton_ascend",
        config=config,
        device_pool=device_pool,
        framework="torch",
        task_type="profile",
        workflow="default_workflow",
        handwrite_suggestions=handwrite_suggestions
    )
    
    _, success_with_hint, task_info_with_hint = await task_with_hint.run()
    
    if not success_with_hint:
        print("❌ 带优化建议的Task执行失败")
        return None
    
    print("✅ 带优化建议的Task执行完成\n")
    
    with_hint_code = task_info_with_hint.get('coder_code', '')
    with_hint_gen_time = task_info_with_hint.get('profile_res', {}).get('gen_time', float('inf'))
    with_hint_base_time = task_info_with_hint.get('profile_res', {}).get('base_time', 0.0)
    with_hint_speedup = task_info_with_hint.get('profile_res', {}).get('speedup', 0.0)
    
    # 4. 打印对比结果
    print("="*80)
    print("性能对比结果")
    print("="*80)
    print(f"\n{'指标':<20} {'Torch基准':<15} {'无优化建议':<15} {'有优化建议':<15}")
    print("-" * 80)
    print(f"{'时间(us)':<20} {no_hint_base_time:>14.2f} {no_hint_gen_time:>14.2f} {with_hint_gen_time:>14.2f}")
    print(f"{'加速比':<20} {'1.00x':>14} {no_hint_speedup:>14.2f}x {with_hint_speedup:>14.2f}x")
    
    if no_hint_gen_time > 0:
        improvement_ratio = (no_hint_gen_time - with_hint_gen_time) / no_hint_gen_time * 100
        print(f"{'性能提升':<20} {'-':>14} {'基准':>14} {improvement_ratio:>13.1f}%")
    
    print("="*80)
    
    print("\n不带优化建议的代码:")
    print("-" * 80)
    print(no_hint_code[:500] + "..." if len(no_hint_code) > 500 else no_hint_code)
    
    print("\n带优化建议的代码:")
    print("-" * 80)
    print(with_hint_code[:500] + "..." if len(with_hint_code) > 500 else with_hint_code)
    print("="*80)
    
    return {
        'success': True,
        'no_hint': {
            'base_time': no_hint_base_time,
            'gen_time': no_hint_gen_time,
            'speedup': no_hint_speedup,
            'code': no_hint_code
        },
        'with_hint': {
            'base_time': with_hint_base_time,
            'gen_time': with_hint_gen_time,
            'speedup': with_hint_speedup,
            'code': with_hint_code
        },
        'improvement_ratio': improvement_ratio if no_hint_gen_time > 0 else 0.0
    }


async def main():
    """主函数"""
    device_id = int(os.getenv("DEVICE_ID", "0"))
    
    # 获取实际任务代码
    task_desc = get_task_desc()
    
    # 获取优化示例
    example_name, torch_code, triton_code, improvement_doc = get_handwrite_optimization_example()
    
    result = await run_analysis(
        op_name="custom_op",
        task_desc=task_desc,
        example_name=example_name,
        torch_code=torch_code,
        triton_reference_code=triton_code,
        improvement_doc=improvement_doc,
        device_id=device_id,
        run_times=50,
        warmup_times=5
    )
    
    if result and result['success']:
        print("✅ 分析完成!\n")
    else:
        print("❌ 分析失败")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
