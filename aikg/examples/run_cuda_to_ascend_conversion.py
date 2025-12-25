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
CUDA-to-Ascend 转换示例（多输出）

本示例演示如何使用服务化架构实现 Triton-CUDA 到 Triton-Ascend 的自动转换：
1. 注册两个 Remote Worker：CUDA (a100) 和 Ascend (ascend910b4)
2. 提交任务时指定 source_backend=cuda, backend=ascend
3. Server 自动在 CUDA Worker 上生成参考数据
4. NPU Worker 使用参考数据验证转换后的代码

示例算子：ReLU + Add 双输出（Triton-CUDA 实现）
- 输入: x, y 两个张量，shape=(537, 32)
- 输出1: relu(x)
- 输出2: x + y
- grid=537, TILE=32

使用方式：


初始设置：启动多后端Workers

# GPU 机器上启动 CUDA Worker
./scripts/server_related/start_worker_service.sh cuda a100 0,1,2,3,4,5,6,7 9001

# NPU 机器上启动 Ascend Worker
./scripts/server_related/start_worker_service.sh ascend ascend910b4 0,1,2,3,4,5,6,7 9001

# [可选] IPv6 环境：通过环境变量设置监听地址
#   export AIKG_WORKER_HOST=::   # 监听所有 IPv6 接口（双栈模式）
#   ./scripts/server_related/start_worker_service.sh cuda a100 0,1,2,3 9001


方式1: 快速验证模式（仅测试参考数据生成和传输，不调用 LLM，包含性能测试）
    export CUDA_WORKER_URL=http://cuda-server:9001
    export ASCEND_WORKER_URL=http://ascend-server:9001
    python examples/run_cuda_to_ascend_conversion.py --verify
    
    # IPv6 场景（注意地址需要用方括号包围）：
    # export CUDA_WORKER_URL=http://[2001:db8::1]:9001
    # export ASCEND_WORKER_URL=http://[2001:db8::2]:9001


方式2: 直接使用 Remote Workers
    export CUDA_WORKER_URL=http://cuda-server:9001
    export ASCEND_WORKER_URL=http://ascend-server:9001
    python examples/run_cuda_to_ascend_conversion.py --direct --num-concurrent 2
    
    # 批量验证目录下所有 Python 文件
    python examples/run_cuda_to_ascend_conversion.py --direct --task-desc-dir /path/to/tasks

"""

import asyncio
import os
import sys
import glob

os.environ['AIKG_STREAM_OUTPUT'] = 'on'


def get_op_name():
    return 'relu_add_dual_output'


def get_task_desc():
    """
    task_desc（Triton-CUDA 代码，用于生成参考数据和转换目标）
    
    本示例演示多输出场景：
    - 输入: x, y 两个张量，shape=(537, 32)
    - 输出1: relu(x)
    - 输出2: x + y
    - grid=537, TILE=32
    """
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
    """
    Triton-CUDA kernel: ReLU + Add 双输出
    每个 program 处理一行（TILE 个元素）
    """
    row_idx = tl.program_id(0)
    
    # 计算偏移
    offsets = row_idx * TILE + tl.arange(0, TILE)
    
    # 加载数据
    x = tl.load(x_ptr + offsets)
    y = tl.load(y_ptr + offsets)
    
    # 计算 ReLU 和 Add
    out1 = tl.maximum(x, 0.0)
    out2 = x + y
    
    # 存储结果
    tl.store(out1_ptr + offsets, out1)
    tl.store(out2_ptr + offsets, out2)


class Model(nn.Module):
    """
    ReLU + Add 双输出模型（Triton-CUDA 实现）
    
    演示多输出场景的参考数据生成和验证
    grid=537, TILE=32
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        使用 Triton-CUDA kernel 计算 ReLU 和 Add 操作
        
        Args:
            x: 输入张量1, shape=(537, 32)
            y: 输入张量2, shape=(537, 32)
        Returns:
            out1: ReLU激活后的张量 relu(x)
            out2: 相加后的张量 x + y
        """
        N, TILE = x.shape
        out1 = torch.empty_like(x)
        out2 = torch.empty_like(x)
        
        grid = (N,)
        relu_add_kernel[grid](
            x, y, out1, out2,
            TILE=TILE,
        )
        
        return out1, out2


N = 537
TILE = 32


def get_inputs():
    x = torch.randn(N, TILE, dtype=torch.float16)
    y = torch.randn(N, TILE, dtype=torch.float16)
    return [x, y]


def get_init_inputs():
    return []  # No special initialization inputs needed
'''


def get_npu_impl_code():
    """
    NPU 实现 用于 与 CUDA 实现 进行验证
    """
    return '''
import torch
import torch.nn as nn
import torch_npu
from typing import Tuple

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out1 = torch.relu(x)
        out2 = x + y
        return out1, out2
'''


def get_task_files_from_dir(task_desc_dir: str):
    """
    从目录中获取所有 Python 文件
    
    Returns:
        list of (op_name, task_desc) tuples
    """
    tasks = []
    py_files = glob.glob(os.path.join(task_desc_dir, "*.py"))
    
    for py_file in sorted(py_files):
        op_name = os.path.splitext(os.path.basename(py_file))[0]
        with open(py_file, 'r', encoding='utf-8') as f:
            task_desc = f.read()
        tasks.append((op_name, task_desc))
    
    return tasks


async def run_direct_with_workers(num_concurrent: int = 4, task_desc_dir: str = None):
    """
    方式2: 直接使用 Remote Workers（不通过 Server）
    
    手动处理参考数据生成和传递
    
    Args:
        num_concurrent: 同一 task 的并发数量，默认为 4，用于更快找到正确解
        task_desc_dir: 任务描述目录，如果指定则读取目录下所有 Python 文件进行批量验证
    """
    from ai_kernel_generator.config.config_validator import load_config
    from ai_kernel_generator.core.worker.manager import register_remote_worker, get_worker_manager
    from ai_kernel_generator.core.verifier.kernel_verifier import KernelVerifier
    from ai_kernel_generator.core.task import Task
    from ai_kernel_generator.core.async_pool.task_pool import TaskPool
    from tests.utils import process_task_results, generate_beautiful_test_report
    
    # 获取任务列表
    if task_desc_dir:
        tasks = get_task_files_from_dir(task_desc_dir)
        if not tasks:
            print(f"错误: 目录 {task_desc_dir} 中没有找到 Python 文件")
            return
        print(f"从目录 {task_desc_dir} 加载了 {len(tasks)} 个任务")
    else:
        # 单个默认任务
        tasks = [(get_op_name(), get_task_desc())]
    
    # 从环境变量获取 Worker URL
    cuda_worker_url = os.environ.get("CUDA_WORKER_URL", "http://localhost:9001")
    ascend_worker_url = os.environ.get("ASCEND_WORKER_URL", "http://localhost:9001")
    
    print("=" * 60)
    print("CUDA-to-Ascend 转换示例 (Direct Worker 模式)")
    print("=" * 60)
    print(f"CUDA Worker URL: {cuda_worker_url}")
    print(f"Ascend Worker URL: {ascend_worker_url}")
    print(f"任务数量: {len(tasks)}")
    print(f"每任务并发数: {num_concurrent}")
    print()
    
    # ========== 1. 注册 Remote Workers ==========
    print("[Step 1] 注册 Workers...")
    
    try:
        await register_remote_worker(
            backend="cuda",
            arch="a100",
            worker_url=cuda_worker_url
        )
        print(f"  [OK] CUDA Worker 注册成功")
    except Exception as e:
        print(f"  [FAIL] CUDA Worker 注册失败: {e}")
        return
    
    try:
        await register_remote_worker(
            backend="ascend",
            arch="ascend910b4",
            worker_url=ascend_worker_url
        )
        print(f"  [OK] Ascend Worker 注册成功")
    except Exception as e:
        print(f"  [FAIL] Ascend Worker 注册失败: {e}")
        return
    
    worker_manager = get_worker_manager()
    print()
    
    # 收集所有任务的结果
    all_results = []
    
    for task_idx, (op_name, task_desc) in enumerate(tasks):
        print(f"\n{'=' * 60}")
        print(f"[任务 {task_idx + 1}/{len(tasks)}] {op_name}")
        print("=" * 60)
        
        # ========== 2. 在 CUDA Worker 上生成参考数据 ==========
        print("[Step 2] 在 CUDA Worker 上生成参考数据...")
        
        config = load_config("triton_cuda", backend="cuda")
        
        cuda_worker = await worker_manager.select(backend="cuda", arch="a100")
        if not cuda_worker:
            print("  [FAIL] 无法获取 CUDA Worker")
            all_results.append((op_name, False, {"error": "无法获取 CUDA Worker"}))
            continue
        
        try:
            verifier = KernelVerifier(
                op_name=op_name,
                framework_code=task_desc,
                task_id=f"gen_ref_{task_idx:03d}",
                framework="torch",
                dsl="triton_cuda",
                backend="cuda",
                arch="a100",
                config=config,
                worker=cuda_worker
            )
            
            success, log, ref_bytes = await verifier.generate_reference_data(task_desc, timeout=120)
            
            if not success:
                print(f"  [FAIL] 参考数据生成失败:")
                print(log)
                all_results.append((op_name, False, {"error": log}))
                continue
            
            print(f"  [OK] 参考数据生成成功 ({len(ref_bytes)} bytes)")
        finally:
            await worker_manager.release(cuda_worker)
        
        print()
        
        # ========== 3. 在 Ascend Worker 上执行转换 ==========
        print(f"[Step 3] 在 Ascend Worker 上执行转换（{num_concurrent} 个并发）...")
        
        ascend_config = load_config("triton_ascend", backend="ascend")
        
        # 注入参考数据
        ascend_config['use_reference_data'] = True
        ascend_config['reference_data'] = ref_bytes
        
        task_pool = TaskPool(max_concurrency=num_concurrent)
        
        # 创建多个相同的 Task 并发运行，更快找到正确解
        for i in range(num_concurrent):
            task = Task(
                op_name=op_name,
                task_desc=task_desc,
                task_id=f"convert_{task_idx:03d}_{i:03d}",
                dsl="triton_ascend",
                backend="ascend",
                arch="ascend910b4",
                config=ascend_config,
                framework="torch",
                workflow="coder_only_workflow"
            )
            task_pool.create_task(task.run)
        
        results = await task_pool.wait_all()
        all_results.extend(results)
        
        # 打印当前任务结果
        print()
        print(f"[{op_name} 结果]")
        success_count = sum(1 for _, s, _ in results if s)
        print(f"  成功: {success_count}/{len(results)}")
        
        # 打印成功的代码
        for result_op_name, task_success, task_info in results:
            if task_success and task_info.get("coder_code"):
                print(f"\n[生成的 Triton-Ascend 代码 - {result_op_name}]")
                print("-" * 40)
                code = task_info.get("coder_code", "")
                print(code)
                break  # 只打印第一个成功的代码
    
    # ========== 4. 汇总结果 ==========
    print()
    print("=" * 60)
    print("总体结果汇总")
    print("=" * 60)
    
    if task_desc_dir:
        # 批量模式：使用美观的报告
        ascend_config = load_config("triton_ascend", backend="ascend")
        generate_beautiful_test_report(
            all_results, 
            ascend_config, 
            framework="torch", 
            dsl="triton_ascend", 
            backend="ascend", 
            arch="ascend910b4"
        )
    else:
        # 单任务模式：使用简单统计
        process_task_results(all_results, print_summary=True)


async def run_quick_verify():
    """
    快速验证模式：测试参考数据生成、传输和性能测试
    
    流程：
    1. CUDA Worker 生成参考数据 (.pt) 并进行性能测试（复用 cross_platform.generate_reference_with_profile）
    2. 传输 .pt 到 Ascend Worker
    3. Ascend Worker 执行验证（使用 NPU 实现代码）并进行性能测试
    """
    from ai_kernel_generator.config.config_validator import load_config
    from ai_kernel_generator.core.worker.manager import register_remote_worker, get_worker_manager
    from ai_kernel_generator.core.verifier.kernel_verifier import KernelVerifier
    from ai_kernel_generator.utils.cross_platform import generate_reference_with_profile
    
    op_name = get_op_name()
    task_desc = get_task_desc()
    
    # 从环境变量获取 Worker URL
    cuda_worker_url = os.environ.get("CUDA_WORKER_URL", "http://localhost:9001")
    ascend_worker_url = os.environ.get("ASCEND_WORKER_URL", "http://localhost:9001")
    
    print("=" * 60)
    print("快速验证模式：测试参考数据生成、传输和性能测试")
    print("=" * 60)
    print(f"CUDA Worker URL: {cuda_worker_url}")
    print(f"Ascend Worker URL: {ascend_worker_url}")
    print()
    
    # ========== 1. 注册 Workers ==========
    print("[Step 1] 注册 Workers...")
    
    try:
        await register_remote_worker(
            backend="cuda",
            arch="a100",
            worker_url=cuda_worker_url
        )
        print(f"  [OK] CUDA Worker 注册成功")
    except Exception as e:
        print(f"  [FAIL] CUDA Worker 注册失败: {e}")
        return False
    
    try:
        await register_remote_worker(
            backend="ascend",
            arch="ascend910b4",
            worker_url=ascend_worker_url
        )
        print(f"  [OK] Ascend Worker 注册成功")
    except Exception as e:
        print(f"  [FAIL] Ascend Worker 注册失败: {e}")
        return False
    
    worker_manager = get_worker_manager()
    print()
    
    # ========== 2. CUDA Worker 生成参考数据 + 性能测试 ==========
    print("[Step 2] CUDA Worker 生成参考数据并测量性能...")
    print("  (复用 cross_platform.generate_reference_with_profile)")
    
    cuda_worker = await worker_manager.select(backend="cuda", arch="a100")
    if not cuda_worker:
        print("  [FAIL] 无法获取 CUDA Worker")
        return False
    
    cuda_time = None
    try:
        # 复用 cross_platform 模块的函数
        result = await generate_reference_with_profile(
            op_name=op_name,
            task_desc=task_desc,
            worker=cuda_worker,
            dsl="triton_cuda",
            backend="cuda",
            arch="a100",
            framework="torch",
            task_id="quick_verify_001",
            warmup_times=5,
            run_times=50,
            timeout=180
        )
        
        if not result.success:
            print(f"  [FAIL] 参考数据生成失败: {result.log}")
            return False
        
        ref_bytes = result.reference_bytes
        cuda_time = result.kernel_time_us
        
        print(f"  [OK] 参考数据生成成功 ({len(ref_bytes)} bytes)")
        print(f"    输出数量: {result.output_count}")
        if cuda_time > 0:
            print(f"  [OK] GPU Kernel 性能测量完成")
            print(f"    执行时间: {cuda_time:.2f} us")
        else:
            print(f"  [WARN] GPU Kernel 性能测量失败")
            
    finally:
        await worker_manager.release(cuda_worker)
    
    print()
    
    # ========== 3. Ascend Worker 验证 ==========
    print("[Step 3] Ascend Worker 使用参考数据验证...")
    
    ascend_config = load_config("triton_ascend", backend="ascend")
    
    # 注入参考数据
    ascend_config['use_reference_data'] = True
    ascend_config['reference_data'] = ref_bytes
    
    ascend_worker = await worker_manager.select(backend="ascend", arch="ascend910b4")
    if not ascend_worker:
        print("  [FAIL] 无法获取 Ascend Worker")
        return False
    
    ascend_time = None
    try:
        # 创建一个简单的验证：使用 PyTorch 原始代码作为 impl
        verifier = KernelVerifier(
            op_name=op_name,
            framework_code=task_desc,
            task_id="quick_verify_002",
            framework="torch",
            dsl="triton_ascend",
            backend="ascend",
            arch="ascend910b4",
            config=ascend_config,
            worker=ascend_worker
        )
        
        # 构造 task_info，使用 PyTorch 代码作为 impl（只是为了验证流程）
        simple_impl_code = get_npu_impl_code()
        task_info = {'coder_code': simple_impl_code}
        
        # 运行验证
        verify_result, verify_log = await verifier.run(task_info, current_step=0)
        
        if verify_result:
            print("  [OK] Ascend Worker 验证成功")
            print("    参考数据传输和加载正常")
        else:
            print("  [FAIL] Ascend Worker 验证失败:")
            print(verify_log)
            return False
        
        # ========== 3.1 Ascend Worker 性能测试 ==========
        print()
        print("[Step 3.1] Ascend Worker 性能测试...")
        
        try:
            ascend_profile_result = await verifier.run_profile(
                task_info,
                current_step=0,
                profile_settings={'warmup_times': 5, 'run_times': 50}
            )
            
            gen_time = ascend_profile_result.get('gen_time')
            if gen_time is not None and gen_time != float('inf'):
                ascend_time = gen_time
                print(f"  [OK] Ascend 性能测试完成")
                print(f"    执行时间: {ascend_time:.2f} us")
            else:
                print(f"  [WARN] Ascend 性能测试失败")
        except Exception as e:
            print(f"  [WARN] Ascend 性能测试异常: {e}")
            
    finally:
        await worker_manager.release(ascend_worker)
    
    print()
    print("=" * 60)
    print("快速验证完成！参考数据生成和传输流程正常")
    print("=" * 60)
    
    # ========== 4. 性能对比汇总 ==========
    print()
    print("=" * 60)
    print("性能测试汇总")
    print("=" * 60)
    
    if cuda_time is not None and cuda_time > 0:
        print(f"[GPU - CUDA a100]")
        print(f"  执行时间: {cuda_time:.2f} us")
    else:
        print(f"[GPU - CUDA a100] 未获取到性能数据")
    
    print()
    
    if ascend_time is not None and ascend_time > 0:
        print(f"[NPU - Ascend 910b4]")
        print(f"  执行时间: {ascend_time:.2f} us")
    else:
        print(f"[NPU - Ascend 910b4] 未获取到性能数据")
    
    # 计算性能对比
    if cuda_time is not None and cuda_time > 0 and ascend_time is not None and ascend_time > 0:
        print()
        print("-" * 40)
        ratio = cuda_time / ascend_time
        if ratio > 1:
            print(f"NPU 比 GPU 快 {ratio:.2f}x")
        else:
            print(f"GPU 比 NPU 快 {1/ratio:.2f}x")
    
    print("=" * 60)
    
    return True


def print_usage():
    print("""
用法:
  python run_cuda_to_ascend_conversion.py --verify
      快速验证模式：测试参考数据生成、传输和性能测试，不调用 LLM
      
  python run_cuda_to_ascend_conversion.py --direct [--num-concurrent N] [--task-desc-dir DIR]
      直接使用 Remote Workers（需设置 CUDA_WORKER_URL 和 ASCEND_WORKER_URL 环境变量）
      --num-concurrent N: 指定同一 task 的并发数量（默认 4），用于更快找到正确解
      --task-desc-dir DIR: 指定任务描述目录，批量读取目录下所有 Python 文件进行验证

示例:
  # 快速验证模式（测试参考数据生成、传输和性能测试）
  export CUDA_WORKER_URL=http://gpu-server:9001
  export ASCEND_WORKER_URL=http://npu-server:9001
  python run_cuda_to_ascend_conversion.py --verify
  
  # Direct Worker 模式（需先启动 Worker Service）
  # GPU: ./scripts/server_related/start_worker_service.sh cuda a100 0 9001
  # NPU: ./scripts/server_related/start_worker_service.sh ascend ascend910b4 0 9001
  export CUDA_WORKER_URL=http://gpu-server:9001
  export ASCEND_WORKER_URL=http://npu-server:9001
  python run_cuda_to_ascend_conversion.py --direct
  
  # 指定8个并发运行（更快找到正确解）
  python run_cuda_to_ascend_conversion.py --direct --num-concurrent 8
  
  # 批量验证目录下所有 Python 文件
  python run_cuda_to_ascend_conversion.py --direct --task-desc-dir /path/to/tasks
  
  # 批量验证 + 指定并发数
  python run_cuda_to_ascend_conversion.py --direct --task-desc-dir /path/to/tasks --num-concurrent 8

IPv6 环境变量配置:
  # 启动 Worker 时监听 IPv6（双栈模式）
  export AIKG_WORKER_HOST=::
  ./scripts/server_related/start_worker_service.sh cuda a100 0 9001
  
  # URL 中使用 IPv6 地址（注意方括号）
  export CUDA_WORKER_URL=http://[2001:db8::1]:9001
  export ASCEND_WORKER_URL=http://[2001:db8::2]:9001
""")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)
    
    if sys.argv[1] == "--direct":
        # 解析参数
        num_concurrent = 4  # 默认值
        task_desc_dir = None
        
        i = 2
        while i < len(sys.argv):
            if sys.argv[i] == "--num-concurrent":
                try:
                    num_concurrent = int(sys.argv[i + 1])
                    i += 2
                except (IndexError, ValueError):
                    print("错误: --num-concurrent 需要一个整数参数")
                    print_usage()
                    sys.exit(1)
            elif sys.argv[i] == "--task-desc-dir":
                try:
                    task_desc_dir = sys.argv[i + 1]
                    if not os.path.isdir(task_desc_dir):
                        print(f"错误: 目录不存在: {task_desc_dir}")
                        sys.exit(1)
                    i += 2
                except IndexError:
                    print("错误: --task-desc-dir 需要一个目录路径参数")
                    print_usage()
                    sys.exit(1)
            else:
                print(f"未知参数: {sys.argv[i]}")
                print_usage()
                sys.exit(1)
        
        asyncio.run(run_direct_with_workers(num_concurrent=num_concurrent, task_desc_dir=task_desc_dir))
    
    elif sys.argv[1] == "--verify":
        success = asyncio.run(run_quick_verify())
        sys.exit(0 if success else 1)
    
    else:
        print(f"未知参数: {sys.argv[1]}")
        print_usage()
        sys.exit(1)
