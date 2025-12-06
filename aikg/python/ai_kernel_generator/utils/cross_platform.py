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
跨平台 CUDA-to-Ascend 转换工具模块

提供跨平台场景下的参考数据生成、性能测量等功能。

核心功能：
1. generate_reference_with_profile: 在 GPU 上生成参考数据 + 测量 kernel 时间
   - 在生成参考数据的脚本中同时测量 kernel 执行时间
   - 只需一次远程调用，避免复杂的 profile 流程
   
2. create_cross_platform_config: 创建跨平台 Evolve 配置
   - 注入参考数据和 GPU kernel 时间
"""

import os
import logging
from typing import Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CrossPlatformReferenceResult:
    """跨平台参考数据生成结果"""
    success: bool = False
    reference_bytes: bytes = b''
    kernel_time_us: float = 0.0  # GPU kernel 执行时间（微秒）
    log: str = ""
    output_count: int = 0
    output_shapes: list = None


async def generate_reference_with_profile(
    op_name: str,
    task_desc: str,
    worker,
    dsl: str = "triton_cuda",
    backend: str = "cuda",
    arch: str = "a100",
    framework: str = "torch",
    log_dir: str = "~/aikg_logs",
    task_id: str = "0",
    warmup_times: int = 5,
    run_times: int = 50,
    timeout: int = 180
) -> CrossPlatformReferenceResult:
    """
    在 GPU 上执行 task_desc 并生成参考数据，同时测量 kernel 执行时间
    
    复用 KernelVerifier 的现有工具：
    - generate_reference_data(): 生成参考数据
    - profile_single_task(): 测量 kernel 时间
    
    Args:
        op_name: 算子名称
        task_desc: Triton-CUDA 代码
        worker: Worker 实例
        warmup_times: 预热次数
        run_times: 性能测量运行次数
        timeout: 超时时间
        
    Returns:
        CrossPlatformReferenceResult: 包含参考数据和 kernel 时间
    """
    from ai_kernel_generator.core.verifier.kernel_verifier import KernelVerifier
    import tempfile
    import torch
    
    result = CrossPlatformReferenceResult()
    
    # 创建 KernelVerifier 实例
    config = {'log_dir': log_dir}
    verifier = KernelVerifier(
        op_name=op_name,
        framework_code=task_desc,
        task_id=task_id,
        framework=framework,
        dsl=dsl,
        backend=backend,
        arch=arch,
        config=config,
        worker=worker
    )
    
    try:
        # Step 1: 生成参考数据
        logger.info(f"[{op_name}] Generating reference data...")
        success, log, ref_bytes = await verifier.generate_reference_data(task_desc, timeout=timeout)
        
        if not success:
            result.log = f"Reference generation failed: {log}"
            return result
        
        logger.info(f"[{op_name}] Reference data generated: {len(ref_bytes)} bytes")
        
        # 解析参考数据获取输出信息
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            f.write(ref_bytes)
            temp_path = f.name
        
        try:
            ref_data = torch.load(temp_path, map_location='cpu')
            result.output_count = len(ref_data.get('outputs', []))
            result.output_shapes = ref_data.get('output_shapes', [])
        finally:
            os.unlink(temp_path)
        
        # Step 2: 测量 kernel 时间（使用 profile_single_task）
        logger.info(f"[{op_name}] Measuring GPU kernel time (profile_single_task)...")
        profile_result = await verifier.profile_single_task(
            task_desc,
            warmup_times=warmup_times,
            run_times=run_times,
            timeout=timeout
        )
        
        if profile_result.get('success', False):
            result.kernel_time_us = profile_result.get('time_us', 0.0)
            logger.info(f"[{op_name}] GPU kernel time: {result.kernel_time_us:.2f} us")
        else:
            # 性能测量失败但参考数据生成成功，继续但警告
            result.kernel_time_us = 0.0
            logger.warning(f"[{op_name}] GPU kernel time measurement failed: {profile_result.get('log', 'Unknown')}")
        
        result.reference_bytes = ref_bytes
        result.success = True
        result.log = "Success"
        return result
        
    except Exception as e:
        result.log = f"Exception: {str(e)}"
        logger.error(f"[{op_name}] generate_reference_with_profile exception: {e}", exc_info=True)
        return result


def create_cross_platform_config(
    base_config: Dict[str, Any],
    reference_bytes: bytes,
    gpu_kernel_time_us: float,
    warmup_times: int = 5,
    run_times: int = 50
) -> Dict[str, Any]:
    """
    创建跨平台 Evolve 的配置
    
    注入参考数据和 GPU kernel 时间到配置中。
    """
    config = base_config.copy()
    
    config['use_reference_data'] = True
    config['reference_data'] = reference_bytes
    config['gpu_kernel_time_us'] = gpu_kernel_time_us
    
    # 设置 profile_settings，包含 override_base_time_us
    # 这样 NPU 端的 profile 会使用 GPU kernel 时间作为 base_time
    # speedup = gpu_time / npu_gen_time 才有意义
    config['profile_settings'] = {
        'run_times': run_times,
        'warmup_times': warmup_times,
        'override_base_time_us': gpu_kernel_time_us,  # 跨平台关键：用 GPU 时间替换 base_time
    }
    
    return config


def calculate_cross_platform_speedup(
    gpu_kernel_time_us: float,
    npu_kernel_time_us: float
) -> Dict[str, Any]:
    """
    计算跨平台性能比
    
    speedup = gpu_time / npu_time（和原来 base/gen 的逻辑一致）
    - speedup > 1.0: NPU 更快
    - speedup < 1.0: GPU 更快
    """
    if gpu_kernel_time_us <= 0 or npu_kernel_time_us <= 0:
        return {
            'valid': False,
            'gpu_time_us': gpu_kernel_time_us,
            'npu_time_us': npu_kernel_time_us,
            'speedup': 0.0,
            'npu_faster': False,
        }
    
    if npu_kernel_time_us == float('inf'):
        return {
            'valid': False,
            'gpu_time_us': gpu_kernel_time_us,
            'npu_time_us': npu_kernel_time_us,
            'speedup': 0.0,
            'npu_faster': False,
        }
    
    # speedup = gpu / npu（和原来 base/gen 的逻辑一致）
    speedup = gpu_kernel_time_us / npu_kernel_time_us
    npu_faster = speedup > 1.0
    
    return {
        'valid': True,
        'gpu_time_us': gpu_kernel_time_us,
        'npu_time_us': npu_kernel_time_us,
        'speedup': speedup,
        'npu_faster': npu_faster,
    }


def extract_best_npu_kernel_time(evolve_result: Dict[str, Any]) -> float:
    """从 evolve 结果中提取最佳 NPU kernel 时间"""
    best_impls = evolve_result.get('best_implementations', [])
    if not best_impls:
        return float('inf')
    
    best_impl = best_impls[0]
    profile = best_impl.get('profile', {})
    return profile.get('gen_time', float('inf'))


def print_cross_platform_performance(
    gpu_kernel_time_us: float,
    npu_kernel_time_us: float,
    op_name: str = ""
):
    """
    打印跨平台性能对比
    
    speedup = gpu / npu（和原来 base/gen 的逻辑一致）
    - speedup > 1.0: NPU 更快
    - speedup < 1.0: GPU 更快
    """
    print()
    print("=" * 60)
    if op_name:
        print(f"  跨平台性能对比 - {op_name}")
    else:
        print("  跨平台性能对比")
    print("=" * 60)
    
    result = calculate_cross_platform_speedup(gpu_kernel_time_us, npu_kernel_time_us)
    
    print(f"  GPU Kernel (Triton-CUDA):    {gpu_kernel_time_us:>10.2f} us")
    print(f"  NPU Kernel (Triton-Ascend):  {npu_kernel_time_us:>10.2f} us")
    print("-" * 60)
    
    if result['valid']:
        speedup = result['speedup']
        # speedup = gpu / npu
        if speedup > 1.0:
            print(f"  Speedup (GPU/NPU): {speedup:.2f}x  (NPU 更快)")
        elif speedup < 1.0:
            print(f"  Speedup (GPU/NPU): {speedup:.2f}x  (GPU 更快)")
        else:
            print(f"  Speedup (GPU/NPU): {speedup:.2f}x  (性能相当)")
    else:
        print("  无法计算性能比（数据无效）")
    
    print("=" * 60)
