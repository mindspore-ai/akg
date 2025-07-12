# NPU-AUL样例代码

## 完整流水线示例

```python
import aul as U

@sub_kernel
def vector_add_pipelined(A: U.TensorPtr, B: U.TensorPtr, C: U.TensorPtr):
    
    # 1. 解析配置参数
    TILE_LEN = 256
    LOOP_COUNT = 5
    total_len = 10240
    CORE_NUM = *
    
    # 2. 获取核心ID并计算数据范围
    core_idx = U.get_core_idx()
    len_per_core = total_len // CORE_NUM
    start_idx = core_idx * len_per_core
    end_idx = start_idx + len_per_core
    
    # 3. 使用Pipelined循环实现流水线
    for i in U.Pipelined(iterations=LOOP_COUNT):
        # 3.1 计算当前迭代数据索引
        current_start = start_idx + i * TILE_LEN
        current_end = current_start + TILE_LEN
        
        # 3.2 创建Tile
        a_tile = U.Tile(shape=(TILE_LEN,), dtype=A.dtype, pos=U.VecBuf)
        b_tile = U.Tile(shape=(TILE_LEN,), dtype=B.dtype, pos=U.VecBuf)
        c_tile = U.Tile(shape=(TILE_LEN,), dtype=C.dtype, pos=U.VecBuf)
        
        # 3.3 流水线阶段1: 加载输入数据
        U.data_copy(dst=a_tile, src=A[current_start:current_end],
                    src_pos=U.GlobalMem, dst_pos=U.VecBuf)
        U.data_copy(dst=b_tile, src=B[current_start:current_end],
                    src_pos=U.GlobalMem, dst_pos=U.VecBuf)
        
        # 3.4 流水线阶段2: 执行计算
        U.vbinary_op(op="add", dst=c_tile, src1=a_tile, src2=b_tile)
        
        # 3.5 流水线阶段3: 写回结果
        U.data_copy(dst=C[current_start:current_end], src=c_tile,
                    src_pos=U.VecBuf, dst_pos=U.GlobalMem)

def vector_add(A: U.TensorPtr, B: U.TensorPtr, C: U.TensorPtr):
    core_num = 8
    vector_add_pipelined[core_num](A, B, C)
```

## 多算子组合

**不支持核间同步**：在不支持核间同步情况下，需要组合不同核的数据，可以采用多个算子组合计算的形式

### 示例
```python
import aul as U

@sub_kernel
def ReductionKernel(input: U.TensorPtr, partial_sum: U.TensorPtr, N: int):
    core_idx = U.get_core_idx()
    BLOCK_SIZE = 40
    elements_per_core = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    start_idx = core_idx * elements_per_core
    end_idx = min(start_idx + elements_per_core, N)
    
    acc_tile = U.Tile((1,), U.float32, U.VecBuf)
    U.vectorscalar_op(op="adds", dst=acc_tile, fill_value=0.0)
    
    BLOCK_SIZE = 256
    data_tile = U.Tile((BLOCK_SIZE,), U.float32, U.VecBuf)
    
    for idx in U.Pipelined(iterations=(end_idx - start_idx + BLOCK_SIZE - 1) // BLOCK_SIZE):
        offset = start_idx + idx * BLOCK_SIZE
        valid_count = min(BLOCK_SIZE, end_idx - offset)
        U.data_copy(dst=data_tile, src=input + offset, 
                   src_pos=U.GlobalMem, dst_pos=U.VecBuf)
        block_acc = U.Tile((1,), U.float32, U.VecBuf)
        U.vreduce_op(op="sum", dst=block_acc, src=data_tile, axis=0)
        U.vbinary_op(op="add", dst=acc_tile, src1=acc_tile, src2=block_acc)
    
    U.data_copy(dst=partial_sum + core_idx, src=acc_tile, src_pos=U.VecBuf, dst_pos=U.GlobalMem)

@sub_kernel
def FinalReductionKernel(partial_sum: U.TensorPtr, output: U.TensorPtr):
    core_idx = U.get_core_idx()
    BLOCK_SIZE = 1
    final_acc = U.Tile((1,), U.float32, U.VecBuf)
    U.vectorscalar_op(op="adds", dst=final_acc, fill_value=0.0)
    data_tile = U.Tile((40,), U.float32, U.VecBuf)
    U.data_copy(dst=data_tile, src=partial_sum, src_pos=U.GlobalMem, dst_pos=U.VecBuf)
    U.vreduce_op(op="sum", dst=final_acc, src=data_tile, axis=0)
    U.data_copy(dst=output, src=final_acc, src_pos=U.VecBuf, dst_pos=U.GlobalMem)

def ComputeTotalSum(input: U.TensorPtr, output: U.TensorPtr, N: int):
    partial_sum = U.SetGlobalTensorPtr(length=(40,), dtype=U.float32)
    core_num1 = 40
    ReductionKernel[core_num1](input, partial_sum, N)
    core_num2 = 1
    FinalReductionKernel[core_num2](partial_sum, output)
```

## 错误示例与推荐示例对比

### 错误示例1：未生成host侧调用代码

**点评：** AUL需要有一个host侧代码对kernel进行调用。

### 推荐示例1：使用正确的host侧代码

```python
@sub_kernel
def op_kernel(input, output):
    pass

def op(input, output):
    core_num = *
    op_kernel[core_num](input, output)
```
**点评：** 只用AUL语法，避免混用Python原生语法。

----------------------------------------------------------
