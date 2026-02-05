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
Kernel性能对比工具
用于对比triton-ascend和原生torch_npu的性能

使用方法：
1. 方法一：直接修改代码
   - 修改 get_custom_op_torch_code() 函数，填入你的torch实现代码
   - 修改 get_custom_op_triton_code() 函数，填入你的triton实现代码
   - 运行 python examples/kernel_profile.py

2. 方法二：通过环境变量读取代码
   - 设置环境变量 TORCH_CODE_PATH 指向你的torch代码文件
   - 设置环境变量 TRITON_CODE_PATH 指向你的triton代码文件
   - 运行 python examples/kernel_profile.py
   
   示例：
   export TORCH_CODE_PATH="/path/to/your/torch_code.py"
   export TRITON_CODE_PATH="/path/to/your/triton_code.py"
   python examples/kernel_profile.py

3. 方法三：通过命令行参数（推荐）
   - 使用 --torch-code-path 指定torch代码文件
   - 使用 --triton-code-path 指定triton代码文件
   - 使用 --op-name 指定算子名称
   - 其他可选参数：--device-id, --run-times, --warmup-times 等
   
   示例：
   python examples/kernel_profile.py --torch-code-path /path/to/torch_code.py --triton-code-path /path/to/triton_code.py --op-name my_op
   
   查看所有参数：
   python examples/kernel_profile.py --help
"""

import os
import argparse
import asyncio
from ai_kernel_generator.core.verifier.kernel_verifier import KernelVerifier
from ai_kernel_generator.config.config_validator import load_config
from ai_kernel_generator.core.worker.manager import register_local_worker, get_worker_manager


def get_custom_op_torch_code(torch_code_path=None):
    """获取自定义算子的torch实现代码（示例：ReLU）"""
    if torch_code_path and os.path.exists(torch_code_path):
        with open(torch_code_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    # 默认代码（当文件不存在时使用）
    return '''
import torch
import torch.nn as nn


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x)


batch_size = 16
dim = 16384


def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]


def get_init_inputs():
    return []  # No special initialization inputs needed
'''


def get_custom_op_triton_code(triton_code_path=None):
    """获取自定义算子的triton实现代码（示例：ReLU）"""
    if triton_code_path and os.path.exists(triton_code_path):
        with open(triton_code_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    # 默认代码（当文件不存在时使用）
    return '''
import torch
import triton
import triton.language as tl


@triton.jit
def custom_op_kernel(
    x_ptr,  # 输入指针
    output_ptr,  # 输出指针
    n_elements,  # 总元素数
    BLOCK_SIZE: tl.constexpr,  # 每个block处理的元素数
):
    # 获取程序ID
    pid = tl.program_id(axis=0)
    # 计算这个block的起始位置
    block_start = pid * BLOCK_SIZE
    # 创建偏移量
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # 创建掩码，确保不越界
    mask = offsets < n_elements

    # 加载输入数据
    x = tl.load(x_ptr + offsets, mask=mask)

    # 执行计算: 这里是ReLU示例 max(0, x)
    output = tl.maximum(x, 0.0)

    # 存储结果
    tl.store(output_ptr + offsets, output, mask=mask)


def custom_op_triton_torch(x):
    x = x.contiguous()
    n_elements = x.numel()
    output = torch.empty_like(x, device=x.device)

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    # 启动kernel
    custom_op_kernel[grid](
        x, output, n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        n_elements = x.numel()
        output = torch.empty_like(x, device=x.device)

        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

        # 启动kernel
        relu_kernel[grid](
            x, output, n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        return output

'''


async def run_kernel_profile(
    op_name="custom_op",
    op_task_str=None,
    kernel_code=None,
    framework="torch",
    dsl="triton_ascend",  # 默认使用triton_ascend，也可以使用triton_cuda
    backend="ascend",
    arch="ascend910b4",
    device_id=0,
    run_times=50,
    warmup_times=5
):
    """
    运行kernel性能对比测试
    
    Args:
        op_name: 算子名称
        op_task_str: 框架实现代码（如torch实现）
        kernel_code: DSL实现代码（如triton实现）
        framework: 框架名称，默认"torch"
        dsl: DSL名称，默认"triton_ascend"（也支持"triton_cuda"）
        backend: 后端名称，默认"ascend"
        arch: 架构名称，默认"ascend910b4"
        device_id: 设备ID，默认0
        run_times: 运行次数，默认50
        warmup_times: 预热次数，默认5
        
    Returns:
        tuple: (gen_time, base_time, speedup) 生成代码性能、原始性能、加速比
    """
    # 新写法：注册 LocalWorker
    await register_local_worker([device_id], backend=backend, arch=arch)
    
    # 从 WorkerManager 获取 worker
    worker = await get_worker_manager().select(backend=backend, arch=arch)
    if not worker:
        raise RuntimeError(f"No available worker for backend={backend}, arch={arch}. Please register a worker first.")
    
    # 加载配置
    config = load_config(dsl)
    
    # 创建验证器，传递 worker
    impl_func_name = f"{op_name}_{dsl}_{framework}"
    verifier = KernelVerifier(
        op_name=op_name,
        framework_code=op_task_str,
        task_id="kernel_profile_001",
        framework=framework,
        dsl=dsl,
        backend=backend,
        arch=arch,
        impl_func_name=impl_func_name,
        config=config,
        worker=worker
    )
    
    task_info = {"coder_code": kernel_code}
    
    # 先进行验证，确保验证通过
    print(f"正在验证 {op_name} kernel的正确性...")
    result, error_log = await verifier.run(task_info, device_id=device_id)
    if not result:
        print(f"❌ 验证失败: {error_log}")
        return None, None, None
    
    print("✅ 正确性验证通过！")
    print("\n" + "="*60)
    print(f"开始性能测试 (预热 {warmup_times} 次，测试 {run_times} 次)")
    print("="*60 + "\n")
    
    # 进行性能分析
    profile_settings = {
        "run_times": run_times,
        "warmup_times": warmup_times
    }
    result = await verifier.run_profile(
        task_info, current_step=0, device_id=device_id, profile_settings=profile_settings
    )
    gen_time = result['gen_time']
    base_time = result['base_time']
    speedup = result['speedup']
    
    # 打印结果
    print("\n" + "="*60)
    print("性能测试结果")
    print("="*60)
    print(f"📊 原生 {framework} 性能:     {base_time:.2f} us")
    print(f"🚀 {dsl} kernel 性能:      {gen_time:.2f} us")
    print(f"⚡ 加速比:                 {speedup:.2f}x")
    print("="*60 + "\n")
    
    return gen_time, base_time, speedup


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Kernel性能对比工具')
    parser.add_argument('--torch-code-path', type=str, 
                       help='Torch代码文件路径')
    parser.add_argument('--triton-code-path', type=str,
                       help='Triton代码文件路径')
    parser.add_argument('--op-name', type=str, default='custom_op',
                       help='算子名称 (默认: custom_op)')
    parser.add_argument('--device-id', type=int, default=None,
                       help='设备ID (默认从环境变量DEVICE_ID获取)')
    parser.add_argument('--run-times', type=int, default=50,
                       help='运行次数 (默认: 50)')
    parser.add_argument('--warmup-times', type=int, default=5,
                       help='预热次数 (默认: 5)')
    
    return parser.parse_args()


async def main():
    """主函数 - 自定义算子性能测试（示例：ReLU）"""
    # 解析命令行参数
    args = parse_args()
    
    # 设置设备ID（优先使用命令行参数，其次环境变量）
    if args.device_id is not None:
        device_id = args.device_id
    else:
        device_id = int(os.getenv("DEVICE_ID", "0"))
    
    # 获取文件路径（优先使用命令行参数，其次环境变量）
    torch_code_path = args.torch_code_path or os.getenv("TORCH_CODE_PATH")
    triton_code_path = args.triton_code_path or os.getenv("TRITON_CODE_PATH")
    
    # 获取op_task_str和kernel_code
    # 如果提供了文件路径，则从文件读取；否则使用默认代码
    op_task_str = get_custom_op_torch_code(torch_code_path)
    kernel_code = get_custom_op_triton_code(triton_code_path)
    
    # 打印使用的代码来源
    if torch_code_path and os.path.exists(torch_code_path):
        print(f"📁 从文件读取 Torch 代码: {torch_code_path}")
    else:
        print("📝 使用默认 Torch 代码")
        
    if triton_code_path and os.path.exists(triton_code_path):
        print(f"📁 从文件读取 Triton 代码: {triton_code_path}")
    else:
        print("📝 使用默认 Triton 代码")
    
    print(f"🔧 算子名称: {args.op_name}")
    print(f"🖥️  设备ID: {device_id}")
    print(f"🔄 运行次数: {args.run_times}, 预热次数: {args.warmup_times}")
    
    # 运行性能测试
    await run_kernel_profile(
        op_name=args.op_name,
        op_task_str=op_task_str,
        kernel_code=kernel_code,
        framework="torch",
        dsl="triton_ascend",  # 默认使用triton_ascend，也可以使用triton_cuda
        backend="ascend",
        arch="ascend910b4",
        device_id=device_id,
        run_times=args.run_times,
        warmup_times=args.warmup_times
    )


if __name__ == "__main__":
    asyncio.run(main())

