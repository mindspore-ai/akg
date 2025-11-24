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

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def linear_kernel(
    x_ptr,          # 输入指针 [batch_size, in_features]
    weight_ptr,     # 权重指针 [out_features, in_features]
    bias_ptr,       # 偏置指针 [out_features] (可以为None)
    output_ptr,     # 输出指针 [batch_size, out_features]
    batch_size: tl.constexpr,
    in_features: tl.constexpr,
    out_features: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton Linear kernel
    计算: output = input @ weight^T + bias
    """
    # 获取当前程序处理的batch和out_feature索引
    pid = tl.program_id(0)
    batch_idx = pid // out_features
    out_idx = pid % out_features
    
    # 检查边界
    if batch_idx >= batch_size or out_idx >= out_features:
        return
    
    # 计算当前行的结果
    acc = 0.0
    for i in range(0, in_features, BLOCK_SIZE):
        # 创建偏移量
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < in_features
        
        # 加载输入数据 [batch_idx, offsets]
        x_offsets = batch_idx * in_features + offsets
        x_vals = tl.load(x_ptr + x_offsets, mask=mask, other=0.0)
        
        # 加载权重数据 [out_idx, offsets] (注意weight需要转置)
        weight_offsets = out_idx * in_features + offsets
        weight_vals = tl.load(weight_ptr + weight_offsets, mask=mask, other=0.0)
        
        # 累加: x * weight^T
        acc += tl.sum(x_vals * weight_vals)
    
    # 添加bias
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + out_idx)
        acc += bias_val
    
    # 存储结果
    output_offset = batch_idx * out_features + out_idx
    tl.store(output_ptr + output_offset, acc)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # 固定随机种子，确保与原始Model的权重一致
        torch.manual_seed(0)
        # 创建Linear层并提取weight和bias
        linear = nn.Linear(in_features, out_features)
        self.weight = nn.Parameter(linear.weight.clone())
        self.bias = nn.Parameter(linear.bias.clone()) if linear.bias is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Triton Linear for PyTorch framework
        """
        batch_size, in_features = x.shape
        out_features = self.weight.shape[0]
        
        # 确保输入是连续的
        x = x.contiguous()
        weight = self.weight.contiguous()
        
        # 分配输出张量
        output = torch.empty((batch_size, out_features), dtype=x.dtype, device=x.device)
        
        # 设置grid大小：每个元素一个线程
        grid = (batch_size * out_features,)
        
        # 准备bias指针
        bias_ptr = None
        if self.bias is not None:
            bias = self.bias.contiguous()
            bias_ptr = bias
        
        # 启动kernel
        BLOCK_SIZE = 32
        linear_kernel[grid](
            x, weight, bias_ptr, output,
            batch_size, in_features, out_features,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        return output

