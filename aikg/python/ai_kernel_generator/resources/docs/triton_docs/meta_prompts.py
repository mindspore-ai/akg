Conceptual_Framework_for_Tiling = """
Triton编程的核心是面向数据块 (Block-level) 的编程范式。您需要思考如何将数据合理地划分为块，并高效地在GPU的SRAM（共享内存）和HBM（全局内存）之间移动这些数据块。优化内存访问是Triton性能优化的关键。
提升数据局部性 (Data Locality)：尽量让每个程序实例（一个Triton内核的一次启动）处理的数据在内存中是连续的。这可以最大化内存访问的合并（Coalescing），减少内存访问延迟。Triton编译器会自动处理大部分访存合并，但良好的数据布局是前提。
数据类型与对齐：确保数据指针是内存对齐的，这对于向量化加载至关重要。
"""

Operator_Fusion = "将多个连续的、逐元素（Element-wise）的操作（如乘法、加法、激活函数等）合并到同一个计算核心中。数据一旦被从全局内存加载到寄存器（Registers），就在寄存器上完成所有计算，最后才将最终结果写回全局内存。这可以极大地减少因读写中间结果而造成的带宽瓶颈和延迟。"

Controlling_Computational_Granularity = "将整个计算任务划分为若干独立的块（Block），每个块由一组线程协同完成。我们需要找到最优的块大小（Block Size），以在并行度（Parallelism）和资源利用率（Resource Utilization）之间取得平衡。太小的块可能导致GPU硬件资源利用不足，而太大的块可能因寄存器或共享内存溢出而无法启动。"

Triton_Block_Pointers_and_Loads = "在Triton中，我们使用 tl.program_id 来获取当前程序块的唯一ID，并结合 tl.arange 来构造块状指针（Block Pointers）。加载数据时必须使用 tl.load，并附带 mask 参数来安全地处理边界情况，避免越界访存。这是实现数据访问模式优化的关键。"

Leveraging_Triton_Autotuning = """
Triton 提供了 `@triton.autotune` 装饰器，可以为影响性能的配置参数（如 `BLOCK_SIZE_M`, `BLOCK_SIZE_N` 等）自动搜索最优值，这是实现硬件粒度性能优化的关键手段。
使用步骤如下：1. **定义搜索空间 (Define Search Space)**:首先，定义一组希望 autotuner 探索的配置。
每个配置都是一个 `triton.Config` 对象。`triton.Config` 接受一个字典来定义你的自定义元参数（如块大小）。
```python
configs = [triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}),triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}),    triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}),    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}), # ...可以根据需求添加更多配置]
```
2. **应用装饰器 (Apply Decorator)**:将 `@triton.autotune` 装饰器应用到 `@triton.jit` kernel 之上。
你需要传入之前定义的 `configs` 列表，并指定 `key` 参数。`key` 是一个字符串列表，包含了函数签名中那些会影响性能的关键参数名称（通常是与问题规模相关的参数，如矩阵维度 M, N, K）。当这些 `key` 参数的输入值发生变化时，Triton 会重新运行基准测试来寻找新的最优配置。

3. 注意！！！调用 triton kernel 的时候不需要再传入configs中定义的变量，configs中的变量需要与kernel参数中最后若干个需要微调的参数保持一致，将这条加入到生成的代码注释中！！
注意！！！调用 triton kernel 的时候不需要再传入configs中定义的变量，configs中的变量需要与kernel参数中最后若干个需要微调的参数保持一致，将这条加入到生成的代码注释中！！
注意！！！调用 triton kernel 的时候不需要再传入configs中定义的变量，configs中的变量需要与kernel参数中最后若干个需要微调的参数保持一致，将这条加入到生成的代码注释中！！

示例：

```
configs = [
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32}),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 64}),
]
@triton.autotune(
    configs=configs,
    key=['M', 'N', 'K'],  # 当矩阵维度 M, N, K 变化时触发 auto-tune
)
@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pass
def matmul(a, b):
    M, K = a.shape
    K, N = b.shape
    c = torch.empty(M, N, device='cuda', dtype=torch.float32)
    
    # 网格：基于 BLOCK_M 和 BLOCK_N
    def grid(meta):
        return (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    
    # 调用内核
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c
```
"""

Implementing_Efficient_Fusion = "在Triton中，算子融合体现在 tl.load 数据到变量后，直接在这些变量上进行一系列数学运算（如 +, *, tl.exp），最后用 tl.store 写回。整个过程没有中间变量的全局内存读写。你应该将所有能合并的逐元素操作都放在load和store之间。"

triton_meta_prompts: list[str] = [
    Conceptual_Framework_for_Tiling,
    Operator_Fusion,
    Controlling_Computational_Granularity,
    Triton_Block_Pointers_and_Loads,
    Leveraging_Triton_Autotuning,
    Implementing_Efficient_Fusion,
]
