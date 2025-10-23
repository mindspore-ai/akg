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
1. 修改 get_custom_op_torch_code() 函数，填入你的torch实现代码
2. 修改 get_custom_op_triton_code() 函数，填入你的triton实现代码
3. 运行 python examples/kernel_profile.py 即可得到性能对比结果
"""

import os
from ai_kernel_generator.core.verifier.kernel_verifier import KernelVerifier
from ai_kernel_generator.config.config_validator import load_config


def get_custom_op_torch_code():
    """获取自定义算子的torch实现代码（示例：ReLU）"""
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


def get_custom_op_triton_code():
    """获取自定义算子的triton实现代码（示例：ReLU）"""
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
'''


def run_kernel_profile(
    op_name="custom_op",
    op_task_str=None,
    kernel_code=None,
    framework="torch",
    dsl="triton",
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
        dsl: DSL名称，默认"triton"
        backend: 后端名称，默认"ascend"
        arch: 架构名称，默认"ascend910b4"
        device_id: 设备ID，默认0
        run_times: 运行次数，默认50
        warmup_times: 预热次数，默认5
        
    Returns:
        tuple: (gen_time, base_time, speedup) 生成代码性能、原始性能、加速比
    """
    # 加载配置
    config = load_config(dsl)
    
    # 创建验证器
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
        config=config
    )
    
    task_info = {"coder_code": kernel_code}
    
    # 先进行验证，确保验证通过
    print(f"正在验证 {op_name} kernel的正确性...")
    result, error_log = verifier.run(task_info, device_id=device_id)
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
    gen_time, base_time, speedup = verifier.run_profile(
        current_step=0, device_id=device_id, profile_settings=profile_settings
    )
    
    # 打印结果
    print("\n" + "="*60)
    print("性能测试结果")
    print("="*60)
    print(f"📊 原生 {framework} 性能:     {base_time:.2f} us")
    print(f"🚀 {dsl} kernel 性能:      {gen_time:.2f} us")
    print(f"⚡ 加速比:                 {speedup:.2f}x")
    print("="*60 + "\n")
    
    return gen_time, base_time, speedup


def main():
    """主函数 - 自定义算子性能测试（示例：ReLU）"""
    # 设置环境变量（可选）
    device_id = int(os.getenv("DEVICE_ID", "0"))
    
    # 获取op_task_str和kernel_code
    # 提示：直接修改 get_custom_op_torch_code() 和 get_custom_op_triton_code() 
    # 两个函数中的代码即可测试不同的算子
    op_task_str = get_custom_op_torch_code()
    kernel_code = get_custom_op_triton_code()
    
    # 运行性能测试
    run_kernel_profile(
        op_name="custom_op",
        op_task_str=op_task_str,
        kernel_code=kernel_code,
        framework="torch",
        dsl="triton",
        backend="ascend",
        arch="ascend910b4",
        device_id=device_id,
        run_times=50,
        warmup_times=5
    )


if __name__ == "__main__":
    main()

