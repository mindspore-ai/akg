# 任务特征
**操作类型**:矩阵乘法 (Matrix Multiplication),A[M, K] @ B[K, N] = C[M, N]
**数据尺寸**:A[2048, 7168]、B[7168, 16384] -> C[2048, 16384],算子规格大
**数据类型**:float16、bfloat16
**任务特点**:计算密集型操作,输出矩阵可分解为(NUM_BLOCKS_M × NUM_BLOCKS_N)个独立块并行计算;核心分配策略对缓存命中率和负载均衡影响显著;硬件约束为20个AI Core,Cube缓存(L0A:64KB, L0B:64KB, L0C:128KB, L1:1MB)。

**分块大小**(受Cube缓存容量限制):
- float16/bfloat16: BLOCK_M=128, BLOCK_K=256, BLOCK_N=256
- float32: BLOCK_M=BLOCK_K=BLOCK_N=128

# 关键代码切片

## 优化1: 固定核心数启动(最重要!)
```python
# ❌ 错误:启动所有块,grid=(NUM_BLOCKS_M*NUM_BLOCKS_N,)
def grid(meta):
    NUM_BLOCKS_M = triton.cdiv(M, meta['BLOCK_M'])
    NUM_BLOCKS_N = triton.cdiv(N, meta['BLOCK_N'])
    return (NUM_BLOCKS_M * NUM_BLOCKS_N,)  # 错误!会启动1024个程序

matmul_kernel[grid](...)

# ✅ 正确:固定核心数启动,grid=(num_cores,)
num_cores = 20  # Ascend 910B4有20个AI Core

# Kernel内部:每个核心循环处理多个块
@triton.jit
def matmul_kernel(..., num_cores: tl.constexpr):
    pid = tl.program_id(axis=0)  # 0~19
    NUM_BLOCKS = NUM_BLOCKS_M * NUM_BLOCKS_N  # 总块数,如1024

    # 每个核心处理多个块: pid, pid+20, pid+40, ...
    for block_idx in range(pid, NUM_BLOCKS, num_cores):
        # 处理第block_idx个块
        ...

# 启动:固定20个核心
matmul_kernel[(num_cores,)](...)  # grid=(20,)
```
**优化内容**:
- ❌ 错误做法:grid=(NUM_BLOCKS_M*NUM_BLOCKS_N,),为每个块启动一个程序(如1024个程序),超出硬件核心数(20个)
- ✅ 正确做法:grid=(num_cores,),固定启动20个核心,每个核心通过for循环处理多个块(每个核心处理~51块)
- Ascend 910B4只有20个AI Core,启动超过20个程序会导致调度开销和性能下降
**总结**:[关键优化] Ascend NPU必须使用固定核心数启动,grid=(num_cores,)即(20,),每个核心循环处理多个块,不要使用grid=(NUM_BLOCKS,)!

## 优化2: Swizzle2D块分组重排
```python
@triton.autotune(
    configs=[
        triton.Config({'GROUP_SIZE': 1}),
        triton.Config({'GROUP_SIZE': 2}),
        triton.Config({'GROUP_SIZE': 3}),
        triton.Config({'GROUP_SIZE': 4}),  # 推荐
        triton.Config({'GROUP_SIZE': 5}),
        triton.Config({'GROUP_SIZE': 8}),
    ],
    key=['M', 'N', 'K']
)
@triton.jit
def matmul_kernel_swizzle2d(..., GROUP_SIZE: tl.constexpr, DIRECTION: tl.constexpr):
    pid = tl.program_id(axis=0)  # 0~19
    NUM_BLOCKS_M = triton.cdiv(M, BLOCK_M)
    NUM_BLOCKS_N = triton.cdiv(N, BLOCK_N)
    NUM_BLOCKS = NUM_BLOCKS_M * NUM_BLOCKS_N

    for block_idx in range(pid, NUM_BLOCKS, num_cores):
        block_m = block_idx // NUM_BLOCKS_N
        block_n = block_idx % NUM_BLOCKS_N

        if DIRECTION == 0:
            # M≥N: 行优先分组,使用tl.swizzle2d
            task_m_idx, task_n_idx = tl.swizzle2d(
                block_m, block_n,
                NUM_BLOCKS_M, NUM_BLOCKS_N,
                GROUP_SIZE
            )
        else:
            # M<N: 列优先分组,手动实现
            size_gj = GROUP_SIZE * NUM_BLOCKS_M
            group_id = block_idx // size_gj
            off_n = group_id * GROUP_SIZE
            cur_size_g = tl.minimum(NUM_BLOCKS_N - off_n, GROUP_SIZE)
            local_ij = block_idx % size_gj
            task_m_idx = local_ij // cur_size_g
            task_n_idx = off_n + local_ij % cur_size_g

        m_start = task_m_idx * BLOCK_M
        n_start = task_n_idx * BLOCK_N
        compute_matmul_block(mat_a, mat_b, mat_c, m_start, n_start, ...)
```
**优化内容**:
- Swizzle2D通过GROUP_SIZE将块按组重排,组内块共享mat_a行或mat_b列数据,提升缓存局部性
- GROUP_SIZE推荐值为4,可通过autotune在[1,2,3,4,5,8]中搜索最优值
**总结**:[通用优化] 矩阵乘法避免简单线性分配,采用分组重排(swizzle)改善缓存局部性和负载均衡

## 优化3: 矩阵形状自适应
```python
def triton_matmul(mat_a, mat_b, dtype=torch.bfloat16):
    m, k, n = mat_a.shape[0], mat_a.shape[1], mat_b.shape[1]

    # 根据矩阵形状选择分组方向
    DIRECTION = 1 if m < n else 0  # M<N列优先, M≥N行优先

    # 根据数据类型选择块大小
    if dtype in [torch.float16, torch.bfloat16]:
        BLOCK_M, BLOCK_K, BLOCK_N = 128, 256, 256
    elif dtype == torch.float32:
        BLOCK_M, BLOCK_K, BLOCK_N = 128, 128, 128

    num_cores = 20  # 固定核心数

    # 关键:使用固定核心数启动,grid=(20,)
    matmul_kernel_swizzle2d[(num_cores,)](
        mat_a, mat_b, mat_c,
        m, n, k, num_cores,
        BLOCK_M, BLOCK_N, BLOCK_K,
        DIRECTION=DIRECTION,
        OUTPUT_DTYPE=output_dtype
    )
```
**优化内容**:
- M≥N时,DIRECTION=0行优先分组,减少mat_a重复加载(可用tl.swizzle2d API)
- M<N时,DIRECTION=1列优先分组,减少mat_b重复加载(需手动实现)
- 示例[2048,7168]@[7168,16384]: M<N,选DIRECTION=1,沿N方向分组
**总结**:[通用优化] 根据输出矩阵M/N比例自适应选择块分配方向,让相同索引的块聚集以提升缓存复用率
