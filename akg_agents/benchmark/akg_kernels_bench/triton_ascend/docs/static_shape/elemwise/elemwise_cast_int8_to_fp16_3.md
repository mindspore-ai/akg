# 任务特征
**操作类型**：Elemwise，类型转换操作
**数据尺寸**：(128, 1024, 1024)，shape较大。
**数据类型**：输入int8，输出fp16
**任务特点**：操作类型为elementwise，可以按照轴的顺序（可flatten为一根轴），外层并行，内层向量化，若UB存不下，可考虑多次切分。

# 关键代码切片

## 优化1
```python
# 优化Triton内核实现：将BLOCK_SIZE分块，每次搬运TILE_SIZE大小的数据
configs = [
    triton.Config({"BLOCK_SIZE": 65536, "TILE_SIZE": 65536}), # 核数2048, 性能最优！在一定范围内提高核数，并尝试用满UB。并且核内没有二次切分。
    triton.Config({"BLOCK_SIZE": 65536, "TILE_SIZE": 32768}), # 核数2048， 但是UB未用满。
    triton.Config({"BLOCK_SIZE": 2097152, "TILE_SIZE": 65536}), # 核数64, 并行度低。
    triton.Config({"BLOCK_SIZE": 4194304, "TILE_SIZE": 65536}), # 核数32, 并行度低。性能低于核数64的情形。
]
    # 内核操作：
    block_start = pid * BLOCK_SIZE
    for i in range(0, BLOCK_SIZE, TILE_SIZE):
        offsets = block_start + tl.arange(0, TILE_SIZE)
        mask = offsets < n_elements
        input_data = tl.load(input_ptr + offsets, mask=mask)
        output_data = tl.cast(input_data, tl.float16)
        tl.store(output_ptr + offsets, output_data, mask=mask)
```
**优化内容**：triton 内核部分使用for循环，尝试进行二次切分，每次搬运TILE_SIZE大小的数据，提高UB的利用率。

**总结**：
- 当数据的shape较大的时候，为了获得更佳的性能，切分值设置尽量能被shape的大小整除。
- 对于单纯的Elementwise操作，将多根轴的元素展开为一根轴，然后在这根轴上进行切分，将block分配给每个线程块，若UB存不下，可考虑多次切分。