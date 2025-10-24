# torch代码
```python
def forward(self, dividend, divisor):
    return dividend / divisor

dividend = torch.randn(1024, 1024, dtype=torch.float16)
divisor = torch.randn(1, 1024, dtype=torch.float16) + 1.0
```

# triton简单实现
```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16}),
    ],
    key=['M', 'N'],
)
@triton.jit
def div_kernel_row(
    dividend_ptr,
    divisor_ptr,
    output_ptr,
    M,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    offs_n = tl.arange(0, N)

    offs_m_2d = offs_m[:, None]
    offs_n_2d = offs_n[None, :]
    
    dividend_offs = offs_m_2d * N + offs_n_2d
    divisor_offs = offs_n_2d
    
    mask_2d = mask_m[:, None]
    
    dividend = tl.load(dividend_ptr + dividend_offs, mask=mask_2d, other=0.0)
    divisor = tl.load(divisor_ptr + divisor_offs)
    
    output = dividend / divisor
    
    tl.store(output_ptr + dividend_offs, output, mask=mask_2d)


def custom_op_triton_torch(dividend: torch.Tensor, divisor: torch.Tensor) -> torch.Tensor:
    assert dividend.ndim == 2 and divisor.ndim == 2, "Both inputs must be 2D"
    assert divisor.shape[0] == 1, "divisor must have shape (1, N)"
    assert dividend.shape[1] == divisor.shape[1], "Column dimension must match"
    
    dividend = dividend.contiguous()
    divisor = divisor.contiguous()
    output = torch.empty_like(dividend)
    
    M, N = dividend.shape
    
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']),)
    
    div_kernel_row[grid](
        dividend, divisor, output,
        M, N,
    )
    
    return output
```

# triton优化实现
```python
@triton.autotune(
    configs=[
        # NUM_BLOCKS 核数，SUB_M 控制内部每次处理行数
        triton.Config({'NUM_BLOCKS': 32, 'SUB_M': 8}), # 最优
        triton.Config({'NUM_BLOCKS': 32, 'SUB_M': 4}), # ub没满
        triton.Config({'NUM_BLOCKS': 64, 'SUB_M': 8}), # 核数 > 40
        triton.Config({'NUM_BLOCKS': 40, 'SUB_M': 8}), # 最优，=核数
    ],
    key=['M', 'N'],
)
@triton.jit
def div_kernel_row(
    dividend_ptr,
    divisor_ptr,
    output_ptr,
    M,
    N: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    SUB_M: tl.constexpr,
):
    pid = tl.program_id(0)
    
    rows_per_block = (M + NUM_BLOCKS - 1) // NUM_BLOCKS
    row_start = pid * rows_per_block
    row_end = tl.minimum(row_start + rows_per_block, M)
    
    offs_n = tl.arange(0, N)
    divisor = tl.load(divisor_ptr + offs_n)

    for sub_start in range(row_start, row_end, SUB_M):
        offs_m = sub_start + tl.arange(0, SUB_M)
        mask_m = offs_m < row_end

        offs_m_2d = offs_m[:, None]
        offs_n_2d = offs_n[None, :]

        dividend_offs = offs_m_2d * N + offs_n_2d

        mask_2d = mask_m[:, None]

        dividend = tl.load(dividend_ptr + dividend_offs, mask=mask_2d, other=0.0)

        output = dividend / divisor

        tl.store(output_ptr + dividend_offs, output, mask=mask_2d)


def custom_op_triton_torch(dividend: torch.Tensor, divisor: torch.Tensor) -> torch.Tensor:
    assert dividend.ndim == 2 and divisor.ndim == 2, "Both inputs must be 2D"
    assert divisor.shape[0] == 1, "divisor must have shape (1, N)"
    assert dividend.shape[1] == divisor.shape[1], "Column dimension must match"
    
    dividend = dividend.contiguous()
    divisor = divisor.contiguous()
    output = torch.empty_like(dividend)
    
    M, N = dividend.shape
    
    grid = lambda meta: (meta['NUM_BLOCKS'],)
    
    div_kernel_row[grid](
        dividend, divisor, output,
        M, N,
    )
    
    return output
```

# 任务特征
**操作类型**：elementwise，broadcast第一根轴；2D Tensor输入，2D Tensor输出
**数据尺寸**：(中，中)
**数据类型**：float16
**任务特点**：操作类型为elementwise，可以向量化操作；triton kernel里面可以直接load一个向量进行单词操作；需要对第一维进行广播；第二维不大，选择切分第一维，将多行分配给每个线程块，并在 kernel 内部通过 for 循环分块处理，既保证了内存访问的连续性，又为 double buffering 和指令流水提供了优化空间。同时，将 grid size 显式限制在物理核数以内，以匹配硬件并行能力，提升整体执行效率。

# 关键代码切片

## 优化1
```python
# 简单Triton
grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']),)

# 优化Triton
grid = lambda meta: (meta['NUM_BLOCKS'],)
```
**优化内容**：通过设置grid大小小于等于物理核数，降低调度开销

**总结**：[通用优化] 在Ascend平台上，当triton kernel的grid数较高时，可以调整grid设置，使得其降低至真实物理核（AI core）数，使 kernel 启动的并行粒度与硬件并行能力对齐，减少调度器负担，提升缓存局部性和计算资源利用率。

## 优化2
```python
# 简单Triton
offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
mask_m = offs_m < M
# 优化Triton
for sub_start in range(row_start, row_end, SUB_M):
    # 当前子块的行索引
    offs_m = sub_start + tl.arange(0, SUB_M)
    mask_m = offs_m < row_end
```
**优化内容**：triton kernel使用for循环，提高指令级并行度，提高硬件利用率

**总结**：[通用优化] 在Ascend平台上，当triton kernel进行一次性操作时，可以将数据分块，添加for循环，触发底层硬件的多级流水线（如 double buffering、指令重叠执行），提升指令级并行度（ILP）和内存带宽利用率