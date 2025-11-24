# AIKG-Triton

## Triton 简介
Triton是一款高性能AI Kernel内核编程语言，专为深度学习应用优化，支持华为Ascend和NVIDIA GPU设备。当前作为AIKG Triton算子生成后端，提供高效的内核代码生成能力。

## 参考代码

AIKG生成的 `relu_op` 示例如下：

```python
# ref: tests/resources/relu_op/relu_op_triton.py
@triton.jit
def relu_kernel(
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

    # 执行ReLU: max(0, x)
    output = tl.maximum(x, 0.0)

    # 存储结果
    tl.store(output_ptr + offsets, output, mask=mask)
```

## 支持的后端

### 华为Atlas A2训练系列产品 Triton 后端依赖
请参考：https://gitee.com/ascend/triton-ascend

### NVIDIA GPU Triton 后端依赖
请参考：https://github.com/triton-lang/triton 