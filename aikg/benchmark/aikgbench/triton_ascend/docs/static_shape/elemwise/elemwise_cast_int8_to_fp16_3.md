# 任务特征
**操作类型**：Elemwise类型：包含torch的cast等类型转换操作
**数据尺寸**：(128, 1024, 1024)数据shape特别大。
**数据类型**：float32, float16, int32, int16, int8
**任务特点**：操作类型为elementwise，将输入的tensor展开为一个轴，直接在一根轴上进行切分，将block分配给每个线程块，并在 kernel 内部通过 for 循环分块处理，进行二次切分。为 double buffer提供可能。同时，通过调整grid的数目调节并行的核数，提升整体执行性能。

# 关键代码切片

## 优化点
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
[通用优化点1] 在Ascend设备上，Triton 内核可以将数据分块，通过添加核内for循环，开启二次切分，提高UB的利用率。
[通用优化点2] 当数据的shape较大的时候，为了获得更佳的性能，核数设置尽量能被shape的大小整除。
[通用优化点3] 选择添加核内切分的时候，TILE_SIZE的设置会很大程度影响性能，需要依据Ascend设备中UB的大小（192KB）实际情况进行调整TILE_SIZE，使得TILE_SIZE的大小的数据量用满UB的空间，并且尽量不要出现尾块。TILE_SIZE的大小可以根据UB的大小以及内核要加载的数据量tl.load()、存储的数据量tl.store()进行计算。取得一个合适的TILE_SIZE大小，能够最大程度利用UB的空间的同时，请不要超出BLOCK_SIZE的范围。
[通用优化点4] 对于单纯的Elementwise操作，如zeros、full、arange、cast操作，将多根轴的元素展开为一根轴，然后在这根轴上进行切分，将block分配给每个线程块，并在 kernel 内部通过 for 循环分块处理，进行二次切分。为 double buffer提供可能。同时，通过调整grid的数目调节并行的核数，提升整体执行性能。