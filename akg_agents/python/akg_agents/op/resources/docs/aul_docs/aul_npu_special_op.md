# NPU-AUL特定算子调度方案

## 任务代码中包含np.where
**若任务代码中包含np.where，请阅读以下材料：**
对于numpy中的np.where，比如`result = np.where(condition, expr1, expr2)`，它的含义是当condition为真时，result=expr1，否则，result=expr2，即
```python
alpha = 1 if condition else 0
result = alpha * expr1 + (1 - alpha) * expr2
```
但是，由于缺少对应布尔计算的API，我们不能按照上面的方法直接计算alpha，而是需要对alpha的计算进行等价转换，具体方法如下：
- 若condition中包含“=”的情况，将condition转换成的等价的大于等于0的形式，保证运算符是“>=”。记condition等价于con_std>=0，M为一个正数常量，能使M*con_std的绝对值大于等于1，则当con_std>=0，alpha=1，否则alpha=0。这时，可以得到alpha的计算公式
```python
alpha = max(1, -con_std * M) + min(0, con_std * M)
```
- 若condition中不包含“=”的情况，步骤稍许复杂。首先，记condition2为condition的互补命题，例如，condition为x>1，则condition2应为x<=1。然后，将condition2转换成con_std2>=0的形式。记N为一个正数常量，能使N*con_std2的绝对值大于等于1，则当con_std2>=0时，alpha=0，否则alpha=1。这时，可以得到alpha的计算公式
```python
alpha = 1 - max(1, -con_std2 * N) - min(0, con_std2 * N)
```
【重要提示】请参照这个思路，利用max、min等API完成np.where到合法AUL语句的转换。

----------------------------------------------------------

## reduce算子

### reduce
- **连续多个reduce轴**：在reduce操作之前，将所有连续的reduce轴合并为一根reduce轴，如：
```python
# input_tile(non_reduce, reduce_dim0, reduce_dim1)
U.vunary_op(op="reshape", dst=reshape_tile, src=input_tile, newshape=[non_reduce, reduce_dim0 * reduce_dim1])
U.vreduce_op(op="sum", dst=reduce_tile, src=reshape_tile, axis=-1)
```

- **最后一根轴不是reduce轴**：使用for循环加普通向量运算替代`U.vreduce_op`，**注意：不同运算初始值不同**，如：
- **最后一根轴不是reduce轴**：使用for循环加普通向量运算替代`U.vreduce_op`，**注意：不同运算初始值不同**，如：
- **最后一根轴不是reduce轴**：使用for循环加普通向量运算替代`U.vreduce_op`，**注意：不同运算初始值不同**，如：
```python
# input_tile(1, reduce_dim, non_reduce_dim)
U.vunary_op(op="vector_dup", dst=output_tile, fill_shape=[1, 1, non_reduce_dim], fill_value=0.0)
for j in U.Pipelined(iterations=reduce_dim):
    U.data_copy(dst=input_tile, src=input_np[0:1, j:j+1, 0:non_reduce_dim], 
                src_pos=U.GlobalMem, dst_pos=U.VecBuf)
    U.vbinary_op(op="sum", dst=output_tile, src1=output_tile, src2=input_tile)
```

- 未指定reduce_axis时，表示所有轴都是reduce轴，即all_reduce
- reduce_sum前需要将输入类型转换为fp32，输出类型转换为原始输入类型


### 各类reduce算子示例

#### reduce融合算子

```python
import aul as U

@sub_kernel
def reduce_sum_fused_pipelined(input_np: U.TensorPtr, output_np: U.TensorPtr):
    # 硬编码参数（来自Tiling）
    BATCH_SIZE = 64
    DIM = 1024
    CORE_NUM = 4
    SAMPLES_PER_CORE = BATCH_SIZE // CORE_NUM
    
    # 获取当前核ID
    core_idx = U.get_core_idx()
    
    # 创建Tile（使用正确形状）
    input_tile = U.Tile(shape=(1, DIM), dtype=U.float16, pos=U.VecBuf)
    input_fp32_tile = U.Tile(shape=(1, DIM), dtype=U.float32, pos=U.VecBuf)
    reduce_fp32_tile = U.Tile(shape=(1, 1), dtype=U.float32, pos=U.VecBuf)
    reduce_tile = U.Tile(shape=(1, 1), dtype=U.float16, pos=U.VecBuf)
    output_tile = U.Tile(shape=(1, DIM), dtype=U.float16, pos=U.VecBuf)

    for i in U.Pipelined(iterations=SAMPLES_PER_CORE):
        start_batch = core_idx * SAMPLES_PER_CORE + i
        end_batch = start_batch + 1
        # 加载数据
        U.data_copy(dst=input_tile, src=input_np[start_batch:end_batch, 0:DIM], 
                    src_pos=U.GlobalMem, dst_pos=U.VecBuf)
        
        # 求最大值
        # 对整个批次向量化求最大值
        U.vunary_op(op="cast_fp16_to_fp32", dst=input_fp32_tile, src=input_tile) 
        U.vreduce_op(op="sum", dst=reduce_fp32_tile, src=input_fp32_tile, axis=1)
        U.vunary_op(op="cast_fp32_to_fp16", dst=reduce_tile, src=reduce_fp32_tile) 
        U.vectorscalar_op(op="adds", dst=output_tile, src=input_tile, factor=reduce_tile)
        
        # 写回结果
        U.data_copy(dst=output_np[start_batch:end_batch, 0:DIM], src=output_tile,
                    src_pos=U.VecBuf, dst_pos=U.GlobalMem)

def reduce_sum_fused(input_np: U.TensorPtr, output_np: U.TensorPtr):
    core_num = 4
    reduce_sum_fused_pipelined[core_num](input_np, output_np)
```

#### 多个连续reduce轴

```python
import aul as U

@sub_kernel
def reduce_max_pipelined(input_np: U.TensorPtr, output_np: U.TensorPtr):
    # 硬编码参数（来自Tiling）
    BATCH_SIZE = 64
    REDUCE_DIM0 = 64
    REDUCE_DIM1 = 64
    CORE_NUM = 1
    SAMPLES_PER_CORE = BATCH_SIZE // CORE_NUM
    
    # 获取当前核ID
    core_idx = U.get_core_idx()
    
    # 创建Tile（使用正确形状）
    input_tile = U.Tile(shape=(1, REDUCE_DIM0, REDUCE_DIM1), dtype=U.float16, pos=U.VecBuf)
    reshape_tile = U.Tile(shape=(1, REDUCE_DIM0 * REDUCE_DIM1), dtype=U.float16, pos=U.VecBuf)
    output_tile = U.Tile(shape=(BATCH_SIZE), dtype=U.float16, pos=U.VecBuf)

    for i in U.Pipelined(iterations=SAMPLES_PER_CORE):
        start_batch = core_idx * SAMPLES_PER_CORE + i
        end_batch = start_batch + 1
        # 加载数据
        U.data_copy(dst=input_tile, src=input_np[start_batch:end_batch, 0:REDUCE_DIM0, 0:REDUCE_DIM1], 
                    src_pos=U.GlobalMem, dst_pos=U.VecBuf)
        
        # 求最大值
        U.vunary_op(op="reshape", dst=reshape_tile, src=input_tile, new_shape=[1, REDUCE_DIM0 * REDUCE_DIM1])
        U.vreduce_op(op="max", dst=reduce_tile, src=reshape_tile, axis=-1)
        
        # 写回结果
        U.data_copy(dst=output_np[start_batch:end_batch, 0:1], src=output_tile,
                    src_pos=U.VecBuf, dst_pos=U.GlobalMem)

def reduce_max(input_np: U.TensorPtr, output_np: U.TensorPtr):
    core_num = 1
    reduce_max_pipelined[core_num](input_np, output_np)
```

#### 最后一根轴不是reduce轴

```python
import aul as U

@sub_kernel
def reduce_max_pipelined(input_np: U.TensorPtr, output_np: U.TensorPtr):
    # 硬编码参数（来自Tiling）
    NON_REDUCE_DIM0 = 64
    REDUCE_DIM0 = 64
    NON_REDUCE_DIM1 = 64
    CORE_NUM = 1
    SAMPLES_PER_CORE = BATCH_SIZE // CORE_NUM
    
    # 获取当前核ID
    core_idx = U.get_core_idx()
    
    # 创建Tile（使用正确形状）
    input_tile = U.Tile(shape=(1, REDUCE_DIM0, NON_REDUCE_DIM1), dtype=U.float16, pos=U.VecBuf)
    reshape_tile = U.Tile(shape=(1, REDUCE_DIM0 * REDUCE_DIM1), dtype=U.float16, pos=U.VecBuf)
    output_tile = U.Tile(shape=(1, 1, NON_REDUCE_DIM1), dtype=U.float16, pos=U.VecBuf)

    for i in U.Pipelined(iterations=SAMPLES_PER_CORE):
        start_batch = core_idx * SAMPLES_PER_CORE + i
        end_batch = start_batch + 1
        # 加载数据

        U.vunary_op(op="vector_dup", dst=output_tile, fill_shape=[1, 1, NON_REDUCE_DIM1], fill_value=-65504.0)
        for j in U.Pipelined(iterations=REDUCE_DIM0):
            U.data_copy(dst=input_tile, src=input_np[start_batch:end_batch, j:j+1, 0:NON_REDUCE_DIM1], 
                        src_pos=U.GlobalMem, dst_pos=U.VecBuf)
            
            # 求最大值
            U.vbinary_op(op="max", dst=output_tile, src1=output_tile, src2=input_tile)
            
        # 写回结果
        U.data_copy(dst=output_np[start_batch:end_batch, 0:1, 0:NON_REDUCE_DIM1], src=output_tile,
                    src_pos=U.VecBuf, dst_pos=U.GlobalMem)

def reduce_max(input_np: U.TensorPtr, output_np: U.TensorPtr):
    core_num = 1
    reduce_max_pipelined[core_num](input_np, output_np)
```

----------------------------------------------------------
