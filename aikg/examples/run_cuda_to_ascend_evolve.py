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
CUDA-to-Ascend 跨平台 Evolve 优化示例

本示例演示如何使用服务化架构实现 Triton-CUDA 到 Triton-Ascend 的自动转换，
并通过 Evolve 流程进行算子优化，最终输出跨平台性能对比。

核心流程：
1. GPU Worker：执行 Triton-CUDA 代码
   - 生成参考数据（.pt 文件，用于正确性验证）
   - 测量 GPU kernel 执行时间（复用 KernelVerifier.profile_single_task）
2. Ascend Worker：使用 Evolve 生成优化代码
   - 验证正确性（使用 GPU 参考数据）
   - 测量 NPU kernel 执行时间（KernelVerifier.profile_single_task）
3. 输出跨平台性能对比：
   - GPU kernel 时间 vs NPU kernel 时间


使用方式：

初始设置：启动多后端Workers

# GPU 机器上启动 CUDA Worker
./scripts/server_related/start_worker_service.sh cuda a100 0,1,2,3,4,5,6,7 9001

# NPU 机器上启动 Ascend Worker
./scripts/server_related/start_worker_service.sh ascend ascend910b4 0,1,2,3,4,5,6,7 9001


运行跨平台 Evolve 优化：
    export CUDA_WORKER_URL=http://cuda-server:9001
    export ASCEND_WORKER_URL=http://ascend-server:9001
    python examples/run_cuda_to_ascend_evolve.py


可选参数：
    --max-rounds N        Evolve 最大轮数（默认 2）
    --parallel-num N      每轮并行任务数（默认 4）
    --num-islands N       岛屿数量（默认 2）
    --elite-size N        精英池大小（默认 3）
    --gpu-profile-runs N  GPU 性能测量运行次数（默认 50）

示例：
    # 基础运行（2轮，4并发）
    python examples/run_cuda_to_ascend_evolve.py
    
    # 更多轮数和并发
    python examples/run_cuda_to_ascend_evolve.py --max-rounds 5 --parallel-num 8
    
    # 禁用岛屿模型（简单模式）
    python examples/run_cuda_to_ascend_evolve.py --num-islands 1 --elite-size 0

"""

import asyncio
import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, Any, List

os.environ['AIKG_STREAM_OUTPUT'] = 'on'


def get_op_name():
    return 'relu_add_dual_output'


def get_task_desc():
    """Triton-CUDA 代码示例"""
    return '''
import torch
import torch.nn as nn
import triton
import triton.language as tl
from typing import Tuple


@triton.jit
def relu_add_kernel(
    x_ptr,
    y_ptr,
    out1_ptr,
    out2_ptr,
    TILE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    offsets = row_idx * TILE + tl.arange(0, TILE)
    
    x = tl.load(x_ptr + offsets)
    y = tl.load(y_ptr + offsets)
    
    out1 = tl.maximum(x, 0.0)
    out2 = x + y
    
    tl.store(out1_ptr + offsets, out1)
    tl.store(out2_ptr + offsets, out2)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        N, TILE = x.shape
        out1 = torch.empty_like(x)
        out2 = torch.empty_like(x)
        
        grid = (N,)
        relu_add_kernel[grid](x, y, out1, out2, TILE=TILE)
        
        return out1, out2


N = 537
TILE = 32


def get_inputs():
    x = torch.randn(N, TILE, dtype=torch.float16)
    y = torch.randn(N, TILE, dtype=torch.float16)
    return [x, y]


def get_init_inputs():
    return []
'''


@dataclass
class CrossPlatformEvolveConfig:
    """跨平台 Evolve 配置"""
    op_name: str = ""
    task_desc: str = ""
    
    # 源后端（CUDA）
    source_backend: str = "cuda"
    source_arch: str = "a100"
    source_dsl: str = "triton_cuda"
    
    # 目标后端（Ascend）
    target_backend: str = "ascend"
    target_arch: str = "ascend910b4"
    target_dsl: str = "triton_ascend"
    
    framework: str = "torch"
    
    # Evolve 参数
    max_rounds: int = 2
    parallel_num: int = 4
    num_islands: int = 2
    migration_interval: int = 2
    elite_size: int = 3
    parent_selection_prob: float = 0.5
    
    # 性能测量配置
    warmup_runs: int = 5
    profile_runs: int = 50


@dataclass
class CrossPlatformResult:
    """跨平台 Evolve 结果"""
    success: bool = False
    
    gpu_kernel_time_us: float = 0.0
    npu_kernel_time_us: float = float('inf')
    cross_platform_speedup: float = 0.0  # GPU/NPU
    npu_faster: bool = False
    
    total_rounds: int = 0
    total_tasks: int = 0
    successful_tasks: int = 0
    success_rate: float = 0.0
    
    best_implementations: List[Dict[str, Any]] = field(default_factory=list)
    error_message: str = ""


async def generate_gpu_reference_and_profile(
    config: CrossPlatformEvolveConfig,
    cuda_worker_url: str
):
    """在 GPU 上生成参考数据并测量 kernel 时间"""
    from ai_kernel_generator.core.worker.manager import register_remote_worker, get_worker_manager
    from ai_kernel_generator.utils.cross_platform import generate_reference_with_profile
    
    print("=" * 60)
    print("[Phase 1] GPU 参考数据生成 & Kernel 性能测量")
    print("=" * 60)
    print(f"  算子: {config.op_name}")
    print(f"  性能测量: warmup={config.warmup_runs}, runs={config.profile_runs}")
    print()
    
    # 注册 CUDA Worker
    print("[Step 1.1] 注册 CUDA Worker...")
    try:
        await register_remote_worker(
            backend=config.source_backend,
            arch=config.source_arch,
            worker_url=cuda_worker_url
        )
        print(f"  ✓ CUDA Worker 注册成功")
    except Exception as e:
        return False, b'', 0.0, f"CUDA Worker 注册失败: {e}"
    
    worker_manager = get_worker_manager()
    cuda_worker = await worker_manager.select(backend=config.source_backend, arch=config.source_arch)
    if not cuda_worker:
        return False, b'', 0.0, "无法获取 CUDA Worker"
    
    try:
        print()
        print("[Step 1.2] 生成参考数据并测量 GPU kernel 时间...")
        print("  (复用 KernelVerifier.generate_reference_data + profile_single_task)")
        
        result = await generate_reference_with_profile(
            op_name=config.op_name,
            task_desc=config.task_desc,
            worker=cuda_worker,
            dsl=config.source_dsl,
            backend=config.source_backend,
            arch=config.source_arch,
            framework=config.framework,
            task_id="gpu_ref",
            warmup_times=config.warmup_runs,
            run_times=config.profile_runs,
        )
        
        if not result.success:
            return False, b'', 0.0, result.log
        
        print(f"  ✓ 参考数据生成成功 ({len(result.reference_bytes)} bytes)")
        print(f"    输出数量: {result.output_count}")
        print()
        print(f"  ✓ GPU Kernel 性能测量完成")
        print(f"    GPU Kernel 时间: {result.kernel_time_us:.2f} us")
        
        return True, result.reference_bytes, result.kernel_time_us, "成功"
        
    finally:
        await worker_manager.release(cuda_worker)


async def run_cross_platform_evolve(
    config: CrossPlatformEvolveConfig,
    ascend_worker_url: str,
    ref_bytes: bytes,
    gpu_kernel_time_us: float
) -> CrossPlatformResult:
    """在 Ascend 上运行 Evolve 优化"""
    from ai_kernel_generator.config.config_validator import load_config
    from ai_kernel_generator.core.worker.manager import register_remote_worker
    from ai_kernel_generator.core.evolve import evolve
    from ai_kernel_generator.core.async_pool.task_pool import TaskPool
    from ai_kernel_generator.utils.cross_platform import (
        create_cross_platform_config,
        extract_best_npu_kernel_time,
        calculate_cross_platform_speedup
    )
    from ai_kernel_generator import get_project_root
    from pathlib import Path
    
    result = CrossPlatformResult()
    result.gpu_kernel_time_us = gpu_kernel_time_us
    
    print()
    print("=" * 60)
    print("[Phase 2] Ascend Evolve 优化")
    print("=" * 60)
    print(f"  算子: {config.op_name}")
    print(f"  GPU Kernel Baseline: {gpu_kernel_time_us:.2f} us")
    print()
    
    # 注册 Ascend Worker
    print("[Step 2.1] 注册 Ascend Worker...")
    try:
        await register_remote_worker(
            backend=config.target_backend,
            arch=config.target_arch,
            worker_url=ascend_worker_url
        )
        print(f"  ✓ Ascend Worker 注册成功")
    except Exception as e:
        result.error_message = f"Ascend Worker 注册失败: {e}"
        return result
    
    # 加载配置
    print()
    print("[Step 2.2] 加载配置...")
    try:
        config_path = str(Path(get_project_root()) / "config" / "vllm_triton_ascend_evolve_config.yaml")
        base_config = load_config(config_path=config_path)
    except Exception:
        base_config = load_config(dsl=config.target_dsl, backend=config.target_backend)
    
    # 注入跨平台配置
    ascend_config = create_cross_platform_config(
        base_config=base_config,
        reference_bytes=ref_bytes,
        gpu_kernel_time_us=gpu_kernel_time_us,
        warmup_times=config.warmup_runs,
        run_times=config.profile_runs
    )
    
    print(f"  ✓ 配置加载完成")
    print(f"    参考数据大小: {len(ref_bytes)} bytes")
    print(f"    GPU Kernel Baseline: {gpu_kernel_time_us:.2f} us")
    
    # 运行 Evolve
    print()
    print("[Step 2.3] 运行 Evolve 优化...")
    print(f"    轮数: {config.max_rounds}, 并发: {config.parallel_num}")
    print(f"    岛屿: {config.num_islands}, 精英池: {config.elite_size}")
    print()
    
    task_pool = TaskPool(max_concurrency=config.parallel_num)
    
    try:
        evolve_result = await evolve(
            op_name=config.op_name,
            task_desc=config.task_desc,
            dsl=config.target_dsl,
            framework=config.framework,
            backend=config.target_backend,
            arch=config.target_arch,
            config=ascend_config,
            task_pool=task_pool,
            max_rounds=config.max_rounds,
            parallel_num=config.parallel_num,
            num_islands=config.num_islands,
            migration_interval=config.migration_interval,
            elite_size=config.elite_size,
            parent_selection_prob=config.parent_selection_prob
        )
        
        # 处理结果
        result.total_rounds = evolve_result.get('total_rounds', 0)
        result.total_tasks = evolve_result.get('total_tasks', 0)
        result.successful_tasks = evolve_result.get('successful_tasks', 0)
        result.success_rate = evolve_result.get('final_success_rate', 0.0)
        result.best_implementations = evolve_result.get('best_implementations', [])
        
        if result.best_implementations:
            result.success = True
            
            # 提取最佳 NPU kernel 时间
            result.npu_kernel_time_us = extract_best_npu_kernel_time(evolve_result)
            
            # 计算跨平台性能比
            speedup = calculate_cross_platform_speedup(
                gpu_kernel_time_us, result.npu_kernel_time_us
            )
            
            if speedup['valid']:
                result.cross_platform_speedup = speedup['speedup']
                result.npu_faster = speedup['npu_faster']
        else:
            result.error_message = "Evolve 未能生成有效的实现"
            
    except Exception as e:
        result.error_message = f"Evolve 执行失败: {str(e)}"
        import traceback
        traceback.print_exc()
    
    return result


def print_result(result: CrossPlatformResult, config: CrossPlatformEvolveConfig):
    """打印结果"""
    from ai_kernel_generator.utils.cross_platform import print_cross_platform_performance
    
    print()
    print("=" * 70)
    print("跨平台 Evolve 优化结果")
    print("=" * 70)
    print()
    
    if result.success:
        print(" Evolve 统计:")
        print(f"  总轮数: {result.total_rounds}")
        print(f"  总任务数: {result.total_tasks}")
        print(f"  成功任务数: {result.successful_tasks}")
        print(f"  成功率: {result.success_rate:.2%}")
        
        # 打印跨平台性能对比
        print_cross_platform_performance(
            result.gpu_kernel_time_us,
            result.npu_kernel_time_us,
            config.op_name
        )
        
        # 显示最佳实现（使用 GPU/NPU 作为 speedup）
        if result.best_implementations:
            print()
            print(" 最佳实现:")
            for i, impl in enumerate(result.best_implementations[:3]):
                profile = impl.get('profile', {})
                gen_time = profile.get('gen_time', float('inf'))
                task_id = impl.get('task_id', 'unknown')
                
                if result.gpu_kernel_time_us > 0 and gen_time != float('inf'):
                    # speedup = gpu / npu（和原来 base/gen 的逻辑一致）
                    speedup = result.gpu_kernel_time_us / gen_time
                    print(f"    #{i+1}: task_id={task_id}")
                    print(f"        GPU: {result.gpu_kernel_time_us:.2f}us, NPU: {gen_time:.2f}us")
                    print(f"        Speedup (GPU/NPU): {speedup:.2f}x")
                else:
                    print(f"    #{i+1}: task_id={task_id}, NPU: {gen_time:.2f}us")
        
    else:
        print("   Evolve 优化失败")
        print(f"  错误: {result.error_message}")
    
    print()
    print("=" * 70)


async def main(args):
    """主函数"""
    cuda_worker_url = os.environ.get("CUDA_WORKER_URL", "http://localhost:9001")
    ascend_worker_url = os.environ.get("ASCEND_WORKER_URL", "http://localhost:9001")
    
    print()
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║     CUDA-to-Ascend 跨平台 Evolve 优化                          ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    print()
    print(f"CUDA Worker:   {cuda_worker_url}")
    print(f"Ascend Worker: {ascend_worker_url}")
    print()
    
    config = CrossPlatformEvolveConfig(
        op_name=get_op_name(),
        task_desc=get_task_desc(),
        max_rounds=args.max_rounds,
        parallel_num=args.parallel_num,
        num_islands=args.num_islands,
        elite_size=args.elite_size,
        warmup_runs=args.warmup_runs,
        profile_runs=args.profile_runs,
    )
    
    start_time = time.time()
    
    # Phase 1: GPU
    success, ref_bytes, gpu_kernel_time, log = await generate_gpu_reference_and_profile(
        config, cuda_worker_url
    )
    
    if not success:
        print(f"\nPhase 1 失败: {log}")
        return 1
    
    # Phase 2: NPU Evolve
    result = await run_cross_platform_evolve(
        config, ascend_worker_url, ref_bytes, gpu_kernel_time
    )
    
    total_time = time.time() - start_time
    
    print_result(result, config)
    print(f"总耗时: {total_time:.1f} 秒")
    print()
    
    return 0 if result.success else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CUDA-to-Ascend 跨平台 Evolve 优化")
    parser.add_argument("--max-rounds", type=int, default=2)
    parser.add_argument("--parallel-num", type=int, default=4)
    parser.add_argument("--num-islands", type=int, default=2)
    parser.add_argument("--elite-size", type=int, default=3)
    parser.add_argument("--profile-runs", type=int, default=50)
    parser.add_argument("--warmup-runs", type=int, default=5)
    
    args = parser.parse_args()
    
    if not os.environ.get("CUDA_WORKER_URL") or not os.environ.get("ASCEND_WORKER_URL"):
        print("警告: CUDA_WORKER_URL 或 ASCEND_WORKER_URL 未设置")
        print()
    
    sys.exit(asyncio.run(main(args)))
