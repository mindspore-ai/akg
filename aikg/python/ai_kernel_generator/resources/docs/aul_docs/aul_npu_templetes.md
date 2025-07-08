# NPU-AUL样例代码

## 完整流水线示例

```python
import aul as U

def vector_add_pipelined(A: U.TensorPtr, B: U.TensorPtr, C: U.TensorPtr):
    
    # 1. 解析配置参数
    TILE_LEN = 256
    LOOP_COUNT = 5
    total_len = 10240
    CORE_NUM = 8
    
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
```

## 错误示例与推荐示例对比

### 错误示例1：非法使用Python原生语法

```python
with open('file.txt') as f:
    ...
```
**点评：** AUL不允许使用Python原生控制流语法，如with、try等。

### 推荐示例1：仅用AUL定义的语法

```python
# 仅允许AUL定义的算子、数据类型、操作
x = U.Tile((16,), U.float32)
```
**点评：** 只用AUL语法，避免混用Python原生语法。

----------------------------------------------------------
