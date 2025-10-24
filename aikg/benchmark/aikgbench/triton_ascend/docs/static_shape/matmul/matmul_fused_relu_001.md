# 任务特征
**操作类型**：matmul + relu 算子，matmul的后向融合，但是对于Ascend后端，可以随路计算，不属于MIX算子
**数据尺寸**：(1000, 8192),(8192, 8192) 算子规格较大
**数据类型**：输入输出均为float16类型
**任务特点**：计算密集型，性能瓶颈通常是内存带宽，易于并行(m轴和n轴)。

# 关键代码切片

## 优化1
```python
# 初始Triton，切分配置
BLOCK_SIZE_M = 256
BLOCK_SIZE_N = 256
BLOCK_SIZE_K = 128

# 优化Triton，切分配置
BLOCK_SIZE_M = 128
BLOCK_SIZE_N = 256
BLOCK_SIZE_K = 256
```
**优化内容**：调整切分值，将 BLOCK_SIZE_K 的大小修改为256，并根据硬件约束自适应调整 BLOCK_SIZE_M 和 BLOCK_SIZE_N
**总结**：[通用优化] 为了充分发挥带宽，尽量让基算子行宽为 512B 的整数倍，且单次行数尽量大，因此该场景中行宽 BLOCK_SIZE_K 的取值为 256，在硬件约束范围内调整 BLOCK_SIZE_M 和 BLOCK_SIZE_N。

## 优化2
```python
# 优化Triton，使能swizzle
# 以nZ方向为例
in_batch_idx = block_idx % (NUM_BLOCKS_M * NUM_BLOCKS_N)
in_batch_idx = block_idx % NUM_BLOCKS
tile_block_loop = (NUM_BLOCKS_M + SWIZZLE_COUNT - 1) // SWIZZLE_COUNT
tile_block_idx = in_batch_idx // (SWIZZLE_COUNT * NUM_BLOCKS_N)
in_tile_block_idx = in_batch_idx % (SWIZZLE_COUNT * NUM_BLOCKS_N)

n_row = SWIZZLE_COUNT
if tile_block_idx == tile_block_loop - 1:
    n_row = NUM_BLOCKS_M - SWIZZLE_COUNT * tile_block_idx
task_m_idx = tile_block_idx * SWIZZLE_COUNT + in_tile_block_idx % n_row
task_n_idx = in_tile_block_idx // n_row
if tile_block_idx % 2 != 0:
    task_n_idx = NUM_BLOCKS_N - task_n_idx - 1
```
**优化内容**：使能swizzle，提升L2 Cache命中率
**总结**：
- [通用优化] 在矩阵乘法中，数据重排可以提升内存带宽利用率，并减少缓存未命中，提高数据重用率，因此 swizzle 是提升矩阵乘法性能的重要优化技术。