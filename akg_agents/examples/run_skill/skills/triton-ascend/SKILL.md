---
name: triton-ascend
description: "Triton Ascend NPU编程指南，包含核心概念、标准模式和完整示例"
level: L3
category: dsl
version: "1.0.0"
license: MIT
metadata:
  backend: ascend
  dsl: triton_ascend
  framework: "torch, mindspore"
---

# Triton Ascend NPU编程指南

> 基于 AKG Agents resources/docs/triton_ascend_docs

## 核心概念

### 内核 (Kernel)
- **定义**: 使用 `@triton.jit` 装饰的Python函数
- **特点**: 并行执行，通过程序ID区分

### 网格与块
- **网格**: 并行维度配置
- **块**: 数据块大小
- **关系**: `grid_size = ceil(total_elements / block_size)`

### 内存层次
- **全局内存**: 所有程序可访问，延迟高
- **共享内存**: 块内共享，延迟低
- **寄存器**: 线程私有，最快

## 标准内核结构

```python
@triton.jit
def standard_kernel(output_ptr, input_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # 1. 获取程序ID和计算偏移
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # 2. 创建边界掩码
    mask = offsets < n_elements
    
    # 3. 加载数据
    data = tl.load(input_ptr + offsets, mask=mask)
    
    # 4. 执行计算
    result = compute_function(data)
    
    # 5. 存储结果
    tl.store(output_ptr + offsets, result, mask=mask)
```

## 三大编程模式

### 1. 向量操作模式
适用于元素级运算。

### 2. 归约模式
适用于聚合操作。

### 3. 矩阵乘法模式
使用分块策略。

## Ascend NPU特性

### NPU架构特点
- AI Core执行计算
- 高带宽内存(HBM)
- 统一虚拟内存

### 优化建议
1. 利用NPU的矩阵计算单元
2. 优化数据布局
3. 使用合适的block size
4. 考虑内存对齐

## 完整示例

参考: `python/akg_agents/op/resources/docs/triton_ascend_docs/examples/`
- torch_matmul.py
- torch_layer_norm.py
- torch_softmax.py
- torch_vector_add.py

## 最佳实践

1. 始终使用mask处理边界
2. Block size选择2的幂
3. 测试不同配置找到最优
4. 注意NPU特定的内存访问模式

## 相关资源

- API文档: triton_ascend_docs/api/api.md
- 建议文档: triton_ascend_docs/suggestion_docs.md
- 示例代码: triton_ascend_docs/examples/
