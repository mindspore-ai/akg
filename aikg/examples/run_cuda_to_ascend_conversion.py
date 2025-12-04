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


方式1: 快速验证模式（仅测试参考数据生成和传输，不调用 LLM）
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


方式3: 通过 Server API
    
    # Server 机器上启动 AIKG Server
    ./scripts/server_related/start_server.sh 8000
    
    # [可选] IPv6 环境：
    #   export AIKG_SERVER_HOST=::
    #   ./scripts/server_related/start_server.sh 8000
    
    # 注册 Workers
    ./scripts/server_related/register_worker_to_server.sh http://localhost:8000 http://gpu-server:9001 cuda a100
    ./scripts/server_related/register_worker_to_server.sh http://localhost:8000 http://npu-server:9001 ascend ascend910b4
    
    # IPv6 环境下注册 Workers（URL 中 IPv6 地址需用方括号包围）：
    # ./scripts/server_related/register_worker_to_server.sh http://[::1]:8000 http://[2001:db8::1]:9001 cuda a100
    
    # 运行此脚本
    python examples/run_cuda_to_ascend_conversion.py --server http://localhost:8000

"""

import asyncio
import os
import sys
import time

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


async def run_via_server_api(server_url: str):
    """
    方式1: 通过 Server API 提交任务（推荐）
    
    Server 会自动处理：
    1. 在 CUDA Worker 上生成参考数据
    2. 将参考数据传递给 Ascend Worker
    3. 执行 LLM 代码生成和验证
    """
    import httpx
    
    op_name = get_op_name()
    task_desc = get_task_desc()
    
    print("=" * 60)
    print("CUDA-to-Ascend 转换示例 (Server API 模式)")
    print("=" * 60)
    print(f"Server URL: {server_url}")
    print()
    
    # 构建请求
    request_data = {
        "op_name": op_name,
        "task_desc": task_desc,
        "job_type": "single",
        "backend": "ascend",
        "arch": "ascend910b4",
        "dsl": "triton_ascend",
        "framework": "torch",
        "workflow": "coder_only_workflow",
        # 关键：指定源后端，触发参考数据生成
        "source_backend": "cuda",
        "source_arch": "a100",
    }
    
    print("[Step 1] 提交任务到 Server...")
    print(f"  算子: {op_name}")
    print(f"  源后端: cuda (a100)")
    print(f"  目标后端: ascend (ascend910b4)")
    print()
    
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                f"{server_url}/api/v1/jobs/submit",
                json=request_data
            )
            response.raise_for_status()
            result = response.json()
            job_id = result.get("job_id")
            print(f"  ✓ 任务提交成功，Job ID: {job_id}")
    except Exception as e:
        print(f"  ✗ 任务提交失败: {e}")
        return
    
    # 轮询任务状态
    print()
    print("[Step 2] 等待任务完成...")
    
    max_wait = 600  # 最多等待 10 分钟
    start_time = time.time()
    
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            while time.time() - start_time < max_wait:
                response = await client.get(f"{server_url}/api/v1/jobs/{job_id}/status")
                response.raise_for_status()
                status = response.json()
                
                job_status = status.get("status")
                print(f"  状态: {job_status}", end="\r")
                
                if job_status in ["completed", "failed", "error"]:
                    print()
                    break
                
                await asyncio.sleep(2)
            else:
                print(f"\n  ✗ 任务超时 ({max_wait}秒)")
                return
    except Exception as e:
        print(f"\n  ✗ 查询状态失败: {e}")
        return
    
    # 显示结果
    print()
    print("[结果]")
    if job_status == "completed":
        result_data = status.get("result", {})
        if result_data.get("success"):
            print(f"  ✓ {op_name} 转换成功！")
            code = result_data.get("code", "")
            if code:
                print(f"\n[生成的 Triton-Ascend 代码]")
                print("-" * 40)
                print(code[:800] + ("..." if len(code) > 800 else ""))
        else:
            print(f"  ✗ {op_name} 转换失败")
    else:
        print(f"  ✗ 任务状态: {job_status}")
        if status.get("error"):
            print(f"  错误: {status.get('error')[:300]}...")


async def run_direct_with_workers(num_concurrent: int = 4):
    """
    方式2: 直接使用 Remote Workers（不通过 Server）
    
    手动处理参考数据生成和传递
    
    Args:
        num_concurrent: 同一 task 的并发数量，默认为 4，用于更快找到正确解
    """
    from ai_kernel_generator.config.config_validator import load_config
    from ai_kernel_generator.core.worker.manager import register_remote_worker, get_worker_manager
    from ai_kernel_generator.core.verifier.kernel_verifier import KernelVerifier
    from ai_kernel_generator.core.task import Task
    from ai_kernel_generator.core.async_pool.task_pool import TaskPool
    from tests.utils import process_task_results
    
    op_name = get_op_name()
    task_desc = get_task_desc()
    
    # 从环境变量获取 Worker URL
    cuda_worker_url = os.environ.get("CUDA_WORKER_URL", "http://localhost:9001")
    ascend_worker_url = os.environ.get("ASCEND_WORKER_URL", "http://localhost:9001")
    
    print("=" * 60)
    print("CUDA-to-Ascend 转换示例 (Direct Worker 模式)")
    print("=" * 60)
    print(f"CUDA Worker URL: {cuda_worker_url}")
    print(f"Ascend Worker URL: {ascend_worker_url}")
    print()
    
    # ========== 1. 注册 Remote Workers ==========
    print("[Step 1] 注册 Workers...")
    
    try:
        await register_remote_worker(
            backend="cuda",
            arch="a100",
            worker_url=cuda_worker_url
        )
        print(f"  ✓ CUDA Worker 注册成功")
    except Exception as e:
        print(f"  ✗ CUDA Worker 注册失败: {e}")
        return
    
    try:
        await register_remote_worker(
            backend="ascend",
            arch="ascend910b4",
            worker_url=ascend_worker_url
        )
        print(f"  ✓ Ascend Worker 注册成功")
    except Exception as e:
        print(f"  ✗ Ascend Worker 注册失败: {e}")
        return
    
    worker_manager = get_worker_manager()
    print()
    
    # ========== 2. 在 CUDA Worker 上生成参考数据 ==========
    print("[Step 2] 在 CUDA Worker 上生成参考数据...")
    
    config = load_config("triton_cuda", backend="cuda")
    
    cuda_worker = await worker_manager.select(backend="cuda", arch="a100")
    if not cuda_worker:
        print("  ✗ 无法获取 CUDA Worker")
        return
    
    try:
        verifier = KernelVerifier(
            op_name=op_name,
            framework_code=task_desc,
            task_id="gen_ref_001",
            framework="torch",
            dsl="triton_cuda",
            backend="cuda",
            arch="a100",
            config=config,
            worker=cuda_worker
        )
        
        success, log, ref_bytes = await verifier.generate_reference_data(task_desc, timeout=120)
        
        if not success:
            print(f"  ✗ 参考数据生成失败: {log[:200]}...")
            return
        
        print(f"  ✓ 参考数据生成成功 ({len(ref_bytes)} bytes)")
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
            task_id=f"convert_{i:03d}",
            dsl="triton_ascend",
            backend="ascend",
            arch="ascend910b4",
            config=ascend_config,
            framework="torch",
            workflow="coder_only_workflow"
        )
        task_pool.create_task(task.run)
    
    results = await task_pool.wait_all()
    
    # 使用通用的结果处理函数打印结果
    print()
    print("[结果]")
    success = process_task_results(results, print_summary=True)
    
    # 打印成功的代码
    for result_op_name, task_success, task_info in results:
        if task_success and task_info.get("coder_code"):
            print(f"\n[生成的 Triton-Ascend 代码 - {result_op_name}]")
            print("-" * 40)
            code = task_info.get("coder_code", "")
            print(code[:800] + ("..." if len(code) > 800 else ""))
            break  # 只打印第一个成功的代码


async def run_quick_verify():
    """
    快速验证模式：仅测试参考数据生成和传输，不调用 LLM
    
    流程：
    1. CUDA Worker 生成参考数据 (.pt)
    2. 传输 .pt 到 Ascend Worker
    3. Ascend Worker 执行验证（使用现有的 PyTorch 代码，跳过 base 执行）
    """
    from ai_kernel_generator.config.config_validator import load_config
    from ai_kernel_generator.core.worker.manager import register_remote_worker, get_worker_manager
    from ai_kernel_generator.core.verifier.kernel_verifier import KernelVerifier
    
    op_name = get_op_name()
    task_desc = get_task_desc()
    
    # 从环境变量获取 Worker URL
    cuda_worker_url = os.environ.get("CUDA_WORKER_URL", "http://localhost:9001")
    ascend_worker_url = os.environ.get("ASCEND_WORKER_URL", "http://localhost:9001")
    
    print("=" * 60)
    print("快速验证模式：测试参考数据生成和传输")
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
        print(f"  ✓ CUDA Worker 注册成功")
    except Exception as e:
        print(f"  ✗ CUDA Worker 注册失败: {e}")
        return False
    
    try:
        await register_remote_worker(
            backend="ascend",
            arch="ascend910b4",
            worker_url=ascend_worker_url
        )
        print(f"  ✓ Ascend Worker 注册成功")
    except Exception as e:
        print(f"  ✗ Ascend Worker 注册失败: {e}")
        return False
    
    worker_manager = get_worker_manager()
    print()
    
    # ========== 2. CUDA Worker 生成参考数据 ==========
    print("[Step 2] CUDA Worker 生成参考数据...")
    
    cuda_config = load_config("triton_cuda", backend="cuda")
    
    cuda_worker = await worker_manager.select(backend="cuda", arch="a100")
    if not cuda_worker:
        print("  ✗ 无法获取 CUDA Worker")
        return False
    
    try:
        verifier = KernelVerifier(
            op_name=op_name,
            framework_code=task_desc,
            task_id="quick_verify_001",
            framework="torch",
            dsl="triton_cuda",
            backend="cuda",
            arch="a100",
            config=cuda_config,
            worker=cuda_worker
        )
        
        success, log, ref_bytes = await verifier.generate_reference_data(task_desc, timeout=120)
        
        if not success:
            print(f"  ✗ 参考数据生成失败:")
            print(f"    {log[:300]}...")
            return False
        
        print(f"  ✓ 参考数据生成成功 ({len(ref_bytes)} bytes)")
        
        # 解析并显示参考数据信息
        import tempfile
        import torch
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            f.write(ref_bytes)
            temp_path = f.name
        
        try:
            ref_data = torch.load(temp_path)
            outputs = ref_data.get('outputs', [])
            print(f"    种子: {ref_data.get('seed', 'unknown')}")
            print(f"    输出数量: {len(outputs)} {'(多输出)' if len(outputs) > 1 else ''}")
            for i, out in enumerate(outputs):
                if hasattr(out, 'shape'):
                    print(f"    输出[{i}]: shape={out.shape}, dtype={out.dtype}")
        finally:
            os.unlink(temp_path)
            
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
        print("  ✗ 无法获取 Ascend Worker")
        return False
    
    try:
        # 创建一个简单的验证：使用 PyTorch 原始代码作为 impl
        # 这里只是验证参考数据传输和加载是否正常
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
        # 这里的 coder_code 是一个简单的透传实现，用于验证参考数据传输
        simple_impl_code = '''
import torch
import torch.nn as nn
import torch_npu
from typing import Tuple

class ModelNew(nn.Module):
    """透传实现，用于验证参考数据流程（双输出）"""
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out1 = torch.relu(x)
        out2 = x + y
        return out1, out2
'''
        task_info = {'coder_code': simple_impl_code}
        
        # 运行验证
        verify_result, verify_log = await verifier.run(task_info, current_step=0)
        
        if verify_result:
            print("  ✓ Ascend Worker 验证成功！")
            print("    参考数据传输和加载正常")
        else:
            print("  ✗ Ascend Worker 验证失败:")
            print(f"    {verify_log[:300]}...")
            return False
            
    finally:
        await worker_manager.release(ascend_worker)
    
    print()
    print("=" * 60)
    print("快速验证完成！参考数据生成和传输流程正常")
    print("=" * 60)
    return True


def print_usage():
    print("""
用法:
  python run_cuda_to_ascend_conversion.py --server <server_url>
      通过 Server API 提交任务（推荐）
      
  python run_cuda_to_ascend_conversion.py --direct [--num-concurrent N]
      直接使用 Remote Workers（需设置 CUDA_WORKER_URL 和 ASCEND_WORKER_URL 环境变量）
      --num-concurrent N: 指定同一 task 的并发数量（默认 4），用于更快找到正确解
      
  python run_cuda_to_ascend_conversion.py --verify
      快速验证模式：仅测试参考数据生成和传输，不调用 LLM

示例:
  # Server API 模式（参考 scripts/server_related/ 中的脚本启动服务）
  python run_cuda_to_ascend_conversion.py --server http://localhost:8000
  
  # Direct Worker 模式（需先启动 Worker Service）
  # GPU: ./scripts/server_related/start_worker_service.sh cuda a100 0 9001
  # NPU: ./scripts/server_related/start_worker_service.sh ascend ascend910b4 0 9001
  export CUDA_WORKER_URL=http://gpu-server:9001
  export ASCEND_WORKER_URL=http://npu-server:9001
  python run_cuda_to_ascend_conversion.py --direct
  
  # 指定8个并发运行（更快找到正确解）
  python run_cuda_to_ascend_conversion.py --direct --num-concurrent 8
  
  # 快速验证模式（仅测试参考数据生成和传输）
  export CUDA_WORKER_URL=http://gpu-server:9001
  export ASCEND_WORKER_URL=http://npu-server:9001
  python run_cuda_to_ascend_conversion.py --verify

IPv6 环境变量配置:
  # 启动 Worker 时监听 IPv6（双栈模式）
  export AIKG_WORKER_HOST=::
  ./scripts/server_related/start_worker_service.sh cuda a100 0 9001
  
  # 启动 Server 时监听 IPv6
  export AIKG_SERVER_HOST=::
  ./scripts/server_related/start_server.sh 8000
  
  # URL 中使用 IPv6 地址（注意方括号）
  export CUDA_WORKER_URL=http://[2001:db8::1]:9001
  export ASCEND_WORKER_URL=http://[2001:db8::2]:9001
""")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)
    
    if sys.argv[1] == "--server":
        if len(sys.argv) < 3:
            print("错误: 请指定 Server URL")
            print_usage()
            sys.exit(1)
        server_url = sys.argv[2]
        asyncio.run(run_via_server_api(server_url))
    
    elif sys.argv[1] == "--direct":
        # 解析 --num-concurrent 参数
        num_concurrent = 4  # 默认值
        if "--num-concurrent" in sys.argv:
            try:
                idx = sys.argv.index("--num-concurrent")
                num_concurrent = int(sys.argv[idx + 1])
            except (IndexError, ValueError):
                print("错误: --num-concurrent 需要一个整数参数")
                print_usage()
                sys.exit(1)
        asyncio.run(run_direct_with_workers(num_concurrent=num_concurrent))
    
    elif sys.argv[1] == "--verify":
        success = asyncio.run(run_quick_verify())
        sys.exit(0 if success else 1)
    
    else:
        print(f"未知参数: {sys.argv[1]}")
        print_usage()
        sys.exit(1)

