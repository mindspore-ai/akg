# move.py



## move_to_gm

### 函数说明  
数据搬运指令，可将输入 Tensor 数据搬运到GM区域。

### 参数说明  
- src: input, Tensor("UB")  。
- no_autopad:input, bool类型，表示是否需要自动pad。默认值为False,表示需要自动pad。
### 返回值
- out: output, 类型同src完全一致。

### sample0  
```python
    src = Tensor("UB", "FP32", [100], format="ND")
    out = move_to_gm(src)
```





## move_to_scalar

### 函数说明  
数据搬运指令，可将输入 Tensor 数据转换为scalar，输入Tensor的shape必须为[1]。

### 参数说明  
- src: input, Tensor("UB"， [1])  

### 返回值
- out: output, 类型同src完全一致

### sample0  
```python
    src = Tensor("UB", "FP32", [1], format="ND")
    out = move_to_scalar(src)
```



## move_to_ub

### 函数说明  
数据搬运指令，可将输入 Tensor 数据搬运到UB区。

### 参数说明  
- src: input, Tensor("L1"/"GM"/"L0C"/"UB") 。
- dtype：可选参数，指定输出Tensor的数据类型，缺省值为输入Tensor数据类型。
- relu：可选参数，bool类型，指定L0C->UB是否需要随路relu，缺省值为False。
- no_autopad:input, bool类型，表示是否需要自动pad。默认值为False,表示需要自动pad。
### 返回值
- out: output。

### sample0  
```python
    src = Tensor("GM", "FP32", [16], format="ND")
    out = move_to_ub(src, "FP16")
```





## move_to_l1

### 函数说明  
数据搬运指令，可将输入 Tensor 数据搬运到L1区。

### 参数说明  
- src: input, Tensor("GM"/"UB") 。

### 返回值
- out: output, 类型同src完全一致。

### sample0  
```python
    src = Tensor("UB", "FP32", [16], format="ND")
    out = move_to_l1(src)
```





## move_to_l0A

### 函数说明  
数据搬运指令，可将输入 Tensor 数据搬运到L0A区。

### 参数说明  
- src: input, Tensor("L1") 。
- Transpose:可选参数，bool类型，表示搬运过程是否需要转置，缺省值为False。
- load3d:可选参数，bool类型，表示搬运过程是否需要image to column操作，缺省值为False。
- k_w:卷积核w维度大小，缺省值为0。
- k_h:卷积核h维度大小，缺省值为0。

### 返回值
- out: output, 类型同src完全一致。

### sample0  
```python
    src = Tensor("L1", "FP32", [16], format="ND")
    out = move_to_l0A(src, False, False)
```



## move_to_l0B

### 函数说明  
数据搬运指令，可将输入 Tensor 数据搬运到L0B区。

### 参数说明  
- src: input, Tensor("L1") 。
- Transpose:可选参数，bool类型，表示搬运过程是否需要转置，缺省值为False。
- load3d:可选参数，bool类型，表示搬运过程是否需要image to column操作，缺省值为False。
- k_w:卷积核w维度大小，缺省值为0。
- k_h:卷积核h维度大小，缺省值为0。

### 返回值
- out: output, 类型同src完全一致。

### sample0  
```python
    src = Tensor("L1", "FP32", [16], format="ND")
    out = move_to_l0B(src, False, False)
```



## move_to_l0C

### 函数说明  
数据搬运指令，可将输入 Tensor 数据搬运到L0C区。

### 参数说明  
- src: input, Tensor("UB") 。
- out_shape:表示输出Tensor的shape。
- multi_core:可选参数，bool类型，表示输出Tensor是否需要广播到batch轴，缺省值为True。

### 返回值
- out: output, 类型同src完全一致。

### sample0  
```python
    src = Tensor("UB", "FP32", [1, 128], format="ND")
    out_shape = [16, 128]
    out = move_to_l0C(src, out_shape, True)
```

## move_scalar_to_ub

### 函数说明  
数据搬运指令，可将输入Scalar数据搬运到指定索引位置的UB区。

### 参数说明  
- src_s: input, Scalar类型。
- src_ub:input, 搬运目标张量，Tensor("UB") 。
- index: input, int类型，表示Scalar数据需要搬运到src_ub的指定位置索引。

### 返回值
- out: output, 和src_ub一致。

### sample0  
```python
    src_s = Scalar("FP32", 10)
    src_ub = Tensor("UB", "FP32", [ 128], format="ND")
    index = 15
    src_ub = move_scalar_to_ub(src_s, src_ub, index)
```