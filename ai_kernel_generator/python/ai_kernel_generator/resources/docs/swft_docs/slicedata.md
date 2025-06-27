# slicedata.py


## slice_to_ub

### 函数说明  
数据切片搬运指令，可将输入 Tensor 进行切片并搬运到UB区。

### 参数说明  
- src0: input, Tensor("L1"/"GM"/"L0C") 。
- begin:input, 指定拆分的起始位置，长度和src0一致。
- slicesize:input, 表示拆分出来的输出Tensor的shape，长度和src0一致。

### 返回值
- out: output, 返回一个Tensor, Tensor的shape为slicesize。类型同src0完全一致。

### sample0  
```python
    out = slice_to_ub(src0, [0, 0], [4, 128])
```

## split_to_ub

### 函数说明  
数据拆分搬运指令，可将输入 Tensor 进行拆分并搬运到UB区，建议使用slice_to_ub替代。

### 参数说明  
- src0: input, Tensor("L1"/"GM"/"L0C"/"UB") 。
- split_size:input, 指定拆分后的每一个输出Tensor特定切分轴的shape。
- split_axis:input, 表示沿着指定维度拆分。
- relu:可选输入参数，bool类型，指定L0C->UB是否需要随路relu，缺省值为False。

### 返回值
- out: output, 返回一个TensorList, 每一个Tensor的shape都是split_size。类型同src0完全一致。

### sample0  
```python
    out = split_to_ub(src0, range(90, 90, 90), 1，False)
```


## pad_to_ub

### 函数说明  
数据pad搬运指令，可将输入 Tensor 进行pad并搬运到UB区。

### 参数说明  
- src0: input, Tensor("UB") 。
- pad_shape:input, 指定pad后的输出Tensor的shape。除M轴外，其他维度size和src0一致。

### 返回值
- out: output，类型同src0完全一致。

### sample0  
```python
    out = pad_to_ub(src0, [16, 256])
```


## slice_to_l1

### 函数说明  
数据切片搬运指令，可将输入 Tensor 进行切片并搬运到L1区。

### 参数说明  
- src0: input, Tensor("GM"/"UB") 。
- begin:input, 指定拆分的起始位置，长度和src0一致。
- slicesize:input, 表示拆分出来的输出Tensor的shape，长度和src0一致。

### 返回值
- out: output, 返回一个Tensor, Tensor的shape为slicesize。类型同src0完全一致。

### sample0  
```python
    out = slice_to_l1(src0, [0, 0], [4, 128])
```




## split_to_l1

### 函数说明  
数据拆分搬运指令，可将输入 Tensor 进行拆分并搬运到L1区，建议使用slice_to_l1。

### 参数说明  
- src0: input, Tensor("GM"/"UB") 。
- split_size:input, 指定拆分后的每一个输出Tensor的shape。
- split_axis:input, 表示沿着指定维度拆分。

### 返回值
- out: output, 返回一个TensorList, 每一个Tensor的shape都是split_size。类型同src0完全一致。

### sample0  
```python
    out = split_to_l1(src0, range(90, 90, 90), 1)
```


## slice_to_l0A

### 函数说明  
数据切片搬运指令，可将输入 Tensor 进行切片并搬运到L0A区。

### 参数说明  
- src0: input, Tensor("L1") 。
- begin:input, 指定拆分的起始位置，长度和src0一致。
- slicesize:input, 表示拆分出来的Tensor的shape，长度和src0一致，后两个轴的值需为16的倍数。
- transpose:可选参数，bool类型，表示搬运过程是否需要转置，缺省值为False。

### 返回值
- out: output, 返回一个Tensor,如果transpose为True，则shape为转置后的slicesize。类型同src0完全一致。

### sample0  
```python
    out = slice_to_l0A(src0, [0, 0], [4, 128], True)
```



## split_to_l0A

### 函数说明  
数据拆分搬运指令，可将输入 Tensor 进行拆分并搬运到L0A区，建议使用slice_to_l0A。

### 参数说明  
- src0: input, Tensor("L1") 。
- split_size:input, 指定拆分后的每一个Tensor的shape。
- split_axis:input, 表示沿着指定维度拆分。
- transpose:可选参数，bool类型，表示搬运过程是否需要转置，缺省值为False。

### 返回值
- out: output, 返回一个TensorList, 每一个Tensor的shape都是split_size。如果transpose为True，每一个Tensor的shape为转置后的split_size。类型同src0完全一致

### sample0  
```python
    out = split_to_l0A(src0, range(90, 90, 90), 1)
```



## slice_to_l0B

### 函数说明  
数据切片搬运指令，可将输入 Tensor 进行切片并搬运到L0B区。

### 参数说明  
- src0: input, Tensor("L1") 。
- begin:input, 指定拆分的起始位置，长度和src0一致。
- slicesize:input, 表示拆分出来的Tensor的shape，长度和src0一致，后两个轴的值需为16的倍数。
- transpose:可选参数，bool类型，表示搬运过程是否需要转置，缺省值为False。

### 返回值
- out: output, 返回一个Tensor,如果transpose为True，则shape为转置后的slicesize。类型同src0完全一致。

### sample0  
```python
    out = slice_to_l0B(src0, [0, 0], [4, 128], True)
```




## split_to_l0B

### 函数说明  
数据拆分搬运指令，可将输入 Tensor 进行拆分并搬运到L0B区，建议使用slice_to_l0B。

### 参数说明  
- src0: input, Tensor("L1") 。
- split_size:input, 指定拆分后的每一个Tensor的shape。
- split_axis:input, 表示沿着指定维度拆分。
- transpose:可选参数，bool类型，表示搬运过程是否需要转置，缺省值为False。

### 返回值
- out: output, 返回一个TensorList, 每一个Tensor的shape都是split_size。如果transpose为True，每一个Tensor的shape为转置后的split_size。类型同src0完全一致。

### sample0  
```python
    out = split_to_l0B(src0, range(90, 90, 90), 1)
```





## concat

### 函数说明  
数据合并指令，可将输入TensorList里的每一个Tensor按照指定轴合并为一个tensor。

### 参数说明  
- src_lst: input, 一个TensorList。每一个Tensor都在UB区。
- concat_axis:input, 表示沿着指定的维度合并。

### 返回值
- out:output, Tensor("UB")，类型同src_lst里的Tensor完全一致。

### sample0  
```python
    src_lst = []
    src_lst.append(src0)
    src_lst.append(src1)
    out = concat(src_lst, 1)
```





## concat_to_l1

### 函数说明  
数据合并指令，可将输入TensorList里的每一个Tensor按照指定轴合并为一个tensor并搬运到L1区。

### 参数说明  
- src_lst: input, 一个TensorList。每一个Tensor都在UB区。
- concat_axis:input, 表示沿着指定的维度合并。

### 返回值
- out:output, Tensor("L1")，类型同src_lst里的Tensor完全一致。

### sample0  
```python
    src_lst = []
    src_lst.append(src0)
    src_lst.append(src1)
    out = concat_to_l1(src_lst, 1)
```


## insert_to_gm

### 函数说明  
数据搬运指令，可将输入Tensor搬运到输出Tensor的指定位置。

### 参数说明
- dst: 既是输入也是输出。 Tensor("GM")。
- src0: input, 一个Tensor，shape大小为slicesize。
- begin:input, 指定搬运目标地址的索引起始位置。
- slicesize:input, 每一次搬运的Tensor的shape。

### sample0  
```python
    insert_to_gm(dst, src0, [0, 0], [128, 128])
```




## concat_to_gm

### 函数说明  
数据合并指令，可将输入TensorList里的每一个Tensor按照指定轴合并为一个tensor并搬运到GM区，建议使用insert_to_gm。

### 参数说明
- dst: 既是输入也是输出。 Tensor("GM")。shape需要和输入Tensor合并后的shape大小保持一致。
- src_lst: input, 一个TensorList。
- concat_axis:input, 表示沿着指定的维度合并。

### sample0  
```python
    src_lst = []
    src0 = Tensor("UB", "FP16", [M1, K], format="ND", multi_core=False)
    src1 = Tensor("UB", "FP16", [M2, K], format="ND", multi_core=False)
    src_lst.append(src0)
    src_lst.append(src1)
    dst = Tensor("GM", "FP16", [M1+M2, K], format="ND", multi_core=False)
    out = concat_to_gm(dst, src_lst, 0)
```




## slice

### 函数说明  
数据切分指令，可将GM区的输入Tensor切片。

### 参数说明
- src0: input, 需要切片的Tensor("GM")。
- begin:input, 指定切片的起始位置，长度和src0一致。
- slicesize：切片的大小。

### 返回值
- out:output, 切片后的Tensor，shape为slicesize。

### sample0  
```python
    out = slice(src0, [0, 0], [16, 16])
```