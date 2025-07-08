# move.py



## move_to_gm

### 函数说明  
数据搬运指令，可将输入 Tensor 数据搬运到GM区域。

### 参数说明  
- src0: input, Tensor("UB")  。

### 返回值
- out: output, 类型同src0完全一致。

### sample0  
```python
    out = move_to_gm(src0)
```





## move_to_scalar

### 函数说明  
数据搬运指令，可将输入 Tensor 数据转换为scalar，输入Tensor的shape必须为[1]。

### 参数说明  
- src0: input, Tensor("UB"， [1])  

### 返回值
- out: output, 类型同src0完全一致

### sample0  
```python
    out = move_to_scalar(src0)
```



## move_to_ub

### 函数说明  
数据搬运指令，可将输入 Tensor 数据搬运到UB区。

### 参数说明  
- src0: input, Tensor("L1"/"GM"/"L0C"/"UB") 。
- dtype：可选参数，指定输出Tensor的数据类型，缺省值为输入Tensor数据类型。
- relu：可选参数，bool类型，指定L0C->UB是否需要随路relu，缺省值为False。

### 返回值
- out: output。

### sample0  
```python
    out = move_to_ub(src0, "FP16")
```





## move_to_l1

### 函数说明  
数据搬运指令，可将输入 Tensor 数据搬运到L1区。

### 参数说明  
- src0: input, Tensor("GM"/"UB") 。

### 返回值
- out: output, 类型同src0完全一致。

### sample0  
```python
    out = move_to_l1(src0)
```





## move_to_l0A

### 函数说明  
数据搬运指令，可将输入 Tensor 数据搬运到L0A区。

### 参数说明  
- src0: input, Tensor("L1") 。
- Transpose:可选参数，bool类型，表示搬运过程是否需要转置，缺省值为False。
- load3d:可选参数，bool类型，表示搬运过程是否需要image to column操作，缺省值为False。
- k_w:卷积核w维度大小，缺省值为0。
- k_h:卷积核h维度大小，缺省值为0。

### 返回值
- out: output, 类型同src0完全一致。

### sample0  
```python
    out = move_to_l0A(src0, False, False)
```



## move_to_l0B

### 函数说明  
数据搬运指令，可将输入 Tensor 数据搬运到L0B区。

### 参数说明  
- src0: input, Tensor("L1") 。
- Transpose:可选参数，bool类型，表示搬运过程是否需要转置，缺省值为False。
- load3d:可选参数，bool类型，表示搬运过程是否需要image to column操作，缺省值为False。
- k_w:卷积核w维度大小，缺省值为0。
- k_h:卷积核h维度大小，缺省值为0。

### 返回值
- out: output, 类型同src0完全一致。

### sample0  
```python
    out = move_to_l0B(src0, False, False)
```



## move_to_l0C

### 函数说明  
数据搬运指令，可将输入 Tensor 数据搬运到L0C区。

### 参数说明  
- src0: input, Tensor("UB") 。
- out_shape:表示输出Tensor的shape。
- multi_core:可选参数，bool类型，表示输出Tensor是否需要广播到batch轴，缺省值为True。

### 返回值
- out: output, 类型同src0完全一致。

### sample0  
```python
    out_shape = [16, 128]
    out = move_to_l0C(src0, out_shape, True)