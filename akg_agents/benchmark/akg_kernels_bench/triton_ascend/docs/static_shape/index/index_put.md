# 任务特征
**操作类型**：索引赋值操作(index_put)，根据索引映射将数据写入目标缓冲区
**数据尺寸**：输入分数(16384, 4)，组缓冲区(8, 65536) -> 算子规格较大
**数据类型**：输入输出均为float32类型，索引为int32类型
**任务特点**：操作类型为​索引赋值​，将输入分数按照单元组索引和位置映射写入组缓冲区；涉及不规则的内存访问模式，需要逐元素处理以避免写冲突；通过​​一维分块并行​优化，将总元素数(65536)按块大小切分，每个线程块处理一个连续的数据块；采用​​Auto-Tuning机制​​动态选择块大小配置，平衡并行度与内存访问效率。

# 关键代码切片

## 优化1
```python
# 简单Triton（每次循环都重新加载索引数据）
for i in tl.range(0, BLOCK_SIZE):
    if start_idx + i < total_elements:
        # 每次循环都从全局内存加载索引
        unit_idx = tl.load(unit_indices_ptr + start_idx + i)
        pos_idx = tl.load(position_map_ptr + start_idx + i)
        # ... 后续处理

# 优化Triton（批量加载索引数据到UB，循环内复用）
# 先批量加载一片索引数据到UB（统一缓冲区）
unit_indices_tile = tl.load(unit_indices_ptr + offsets, mask=mask, other=0)
position_map_tile = tl.load(position_map_ptr + offsets, mask=mask, other=0)

# 循环内通过get_element从UB中取数，复用数据
for i in tl.range(0, BLOCK_SIZE):
    if start_idx + i < total_elements:
        # 从UB中取数，避免重复访问全局内存
        unit_idx = tl.get_element(unit_indices_tile, [i])
        pos_idx = tl.get_element(position_map_tile, [i])
        # ... 后续处理
```
**优化内容**：
采用批量加载+数据复用策略：
- 在循环外，通过一次tl.load操作将整个BLOCK_SIZE的索引数据批量加载到UB（统一缓冲区/片上缓存）中；
- 在循环内，通过tl.get_element从UB中逐个取出索引值，避免每次循环都访问全局内存；
- 这种方式将多次全局内存访问转换为一次批量加载+多次片上缓存访问，显著降低内存访问延迟。
**总结**：[通用优化] 当需要在循环中多次访问同一片数据时，应先批量加载到片上缓存（UB），然后通过get_element逐个取用，实现数据复用，减少全局内存访问次数，提升性能。
