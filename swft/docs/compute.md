# compute.py



## mmad

### 函数说明  
矩阵乘运算指令，可对两个输入 Tensor 进行矩阵乘运算。（可选：额外第三输入src2做累加）

### 参数说明  
- src0: input, Tensor("L0A", "FP16"/"INT8", size=[M0, K0], ...) 
- src1: input, Tensor("L0B", "FP16"/"INT8", size=[M0, K0], ...) ，与src0 dtype完全一致
- src2: input, 可选Tensor；当非None时参与计算，默认为None  
### 返回值
- out: output, Tensor

### sample0  
```python
if k == 0: # k轴分块
    l0c = mmad(l0a, l0b)
else:
    l0c = mmad(l0a, l0b, l0c)

```





## vadd
### 函数说明
向量-向量 加法指令。
### 参数说明
- src0: input, Tensor("UB", "FP16"/"FP32"/"INT32", format="ND")
- src1: input, 类型同src0完全一致，否则需要Cast(vconv指令)对齐
### 返回值
- out: output, Tensor, 类型同src0完全一致

### sample0
```python
ub_m = move_to_ub(gm_m) # Tensor
ub_v = move_to_ub(gm_v) # Tensor
ub_out = vadd(ub_m, ub_v)
```



## vsub
### 函数说明
向量-向量 减法指令.

### 参数说明
- src0: input, Tensor("UB", "FP16"/"FP32"/"INT32", format="ND")
- src1: input, 类型同src0完全一致，否则需要Cast(vconv指令)对齐
### 返回值
- out: output, 类型同src0完全一致

### sample0
```python
ub_a = Tensor("UB", "FP32", [1024], format="ND")
ub_b = Tensor("UB", "FP32", [1024], format="ND")
ub_out = vsub(ub_a, ub_b)
```





## vmul
### 函数说明
向量-向量 乘法指令。
### 参数说明
- src0: input, Tensor("UB", "FP16"/"FP32"/"INT32", format="ND")
- src1: input, 类型同src0完全一致，否则需要Cast(vconv指令)对齐
### 返回值
- out: output, 类型同src0完全一致

### sample0
```python
ub_m = move_to_ub(gm_m) # Tensor
ub_v = move_to_ub(gm_v) # Tensor
ub_out = vmul(ub_m, ub_v)
```





## vdiv
### 函数说明
向量-向量 除法指令。
### 参数说明
- src0: input, Tensor("UB", "FP16"/"FP32"/"INT32", format="ND")
- src1: input, 类型同src0完全一致，否则需要Cast(vconv指令)对齐
### 返回值
- out: output, 类型同src0完全一致

### sample0
```python
ub_a = Tensor("UB", "FP32", [1024], format="ND", multi_core=False)
ub_b = Tensor("UB", "FP32", [1024], format="ND", multi_core=False)
ub_out = vdiv(ub_a, ub_b)
```



## vmax
### 函数说明
向量-向量 最大值指令。
### 参数说明
- src0: input, Tensor("UB", "FP16"/"FP32"/"INT32", format="ND")
- src1: input, 类型同src0完全一致，否则需要Cast(vconv指令)对齐
### 返回值
- out: output, 类型同src0完全一致，对应位置元素最大值

### sample0
```python
ub_max = vmax(ub_a, ub_b)
```





## vmin
### 函数说明
向量-向量 最小值操作。
### 参数说明
- src0: input, Tensor("UB", "FP16"/"FP32"/"INT32", format="ND")
- src1: input, 类型同src0完全一致，否则需要Cast(vconv指令)对齐
### 返回值
- out: output, 类型同src0完全一致，对应位置元素最小值

### sample0
```python
ub_m = move_to_ub(gm_m) # Tensor("UB", "FP32", ...)
ub_v = move_to_ub(gm_v) # Tensor("UB", "FP32", ...)
ub_out = vmin(ub_m, ub_v)
```





## vand
### 函数说明
向量-向量 按位与操作指令。
### 参数说明
- src0: input, Tensor("UB", "UINT16/"INT16", format="ND")
- src1: input, 类型同src0完全一致，否则需要Cast(vconv指令)对齐
### 返回值
- out: output, Tensor

### sample0
```python
ub_a = Tensor("UB", "INT16", [128], format="ND")
ub_b = Tensor("UB", "INT16", [128], format="ND")
ub_result = vand(ub_a, ub_b)
```





## vor
### 函数说明
向量-向量 "或"二元操作指令。
### 参数说明
- src0: input, Tensor("UB", "UINT16/"INT16", format="ND")
- src1: input, 类型同src0完全一致，否则需要Cast(vconv指令)对齐
### 返回值
- out: output, Tensor

### sample0
```python
ub_m = move_to_ub(gm_m) # Tensor("UB", "INT16", [1024], format="ND")
ub_v = move_to_ub(gm_v) # Tensor("UB", "INT16", [1024], format="ND")
out_ub = vor(ub_m, ub_v)
```



## vadds
### 函数说明
向量-标量 加法指令。
### 参数说明
- src: input, Tensor("UB", "FP16"/"FP32"/"INT32", format="ND")
- factor: input, Scalar，dtype与src一致

### 返回值
- out: output, Tensor，dtype与src一致

### sample0
```python
ub_m = move_to_ub(gm_m) # Tensor
factor = Scalar("FP32", 2.0)
ub_out = vadds(ub_m, factor)
```




## vsubs
### 函数说明
向量-标量 减法指令。
### 参数说明
- src: input, Tensor("UB", "FP16"/"FP32"/"INT32", format="ND")
- factor: input, Scalar，dtype与src一致

### 返回值
- out: output, Tensor，dtype与src一致

### sample0
```python
ub_m = move_to_ub(gm_m) # Tensor
factor = Scalar("FP32", 2.0)
ub_out = vsubs(ub_m, factor)
```


## vdivs
### 函数说明
向量-标量 除法指令。
### 参数说明
- src: input, Tensor("UB", "FP16"/"FP32"/"INT32", format="ND")
- factor: input, Scalar，dtype与src一致

### 返回值
- out: output, Tensor，dtype与src一致

### sample0
```python
ub_v = Tensor("UB", "FP32", [1024], format="ND")
factor = Scalar("FP32", 2.0)
scaled_v = vdivs(ub_v, factor)
```


## vmuls
### 函数说明
向量-标量 乘法指令。
### 参数说明
- src: input, Tensor("UB", "FP16"/"FP32"/"INT32", format="ND")
- factor: input, Scalar，dtype与src一致

### 返回值
- out: output, Tensor，dtype与src一致

### sample0
```python
ub_v = Tensor("UB", "FP32", [1024], format="ND")
factor = Scalar("FP32", 2.0)
scaled_v = vmuls(ub_v, factor)
```





## vmaxs
### 函数说明
向量-标量 最大值操作。
### 参数说明
- src: input, Tensor("UB", "FP16"/"FP32"/"INT32", format="ND")
- factor: input, Scalar，dtype与src一致

### 返回值
- out: output, Tensor，dtype与src一致

### sample0
```python
ub_v = Tensor("UB", "FP32", [1024], format="ND")
factor = Scalar("FP32", 2.0)
scaled_v = vmaxs(ub_v, factor)
```



## vmins
### 函数说明
向量-标量 最小值操作。

### 参数说明
- src: input, Tensor("UB", "FP16"/"FP32"/"INT32", format="ND")
  
- factor: input, Scalar，dtype与src一致

### 返回值
- out: output, Tensor，dtype与src一致

### sample0
```python
ub_m = Tensor("UB", "FP32", [1024], format="ND")
ub_factor = Scalar("FP32", 0.5)
ub_out = vmins(ub_m, ub_factor)
```



## vexp
### 函数说明
向量 Exp运算指令。
### 参数说明
- src: input, Tensor("UB", "FP16"/"FP32", format="ND")
### 返回值
- out: output, Tensor，dtype与src一致。

### sample0
```python
ub_exp = vexp(ub_m)
```





## vsqrt
### 函数说明
向量平方根运算指令。
### 参数说明
- src: input, Tensor("UB", "FP16"/"FP32", format="ND")

### 返回值
- out: output, Tensor，dtype与src一致。

### sample0
```python
ub_input = Tensor("UB", "FP32", [1024], format="ND", multi_core=False)
ub_output = vsqrt(ub_input)
```





## vrelu
### 函数说明
向量ReLU操作，将输入中的负值替换为零，并对正值保持不变。
### 参数说明
- src: input, Tensor("UB", "FP16"/"FP32"/"INT32", format="ND") 
### 返回值
- out: output, Tensor，dtype与src一致。

### sample0
```python
ub_v = move_to_ub(gm_v) # Tensor
ub_out_v = vrelu(ub_v)
```





## vln
### 函数说明
向量按元素取自然对数。
### 参数说明
- src: input, Tensor("UB", "FP16"/"FP32", format="ND") 

### 返回值
- out: output, Tensor，dtype与src一致。

### sample0
```python
ub_out = vln(ub_v)
```





## vrec
### 函数说明  
向量取倒数指令(reciprocal )。  
### 参数说明  
- src: input, Tensor("UB", "FP16"/"FP32", format="ND") 
### 返回值
- out: output, Tensor，dtype与src一致。

### sample0  
```python 
ub_v = Tensor("UB", "FP32", [1024], format="ND", multi_core=False) 
ub_out_v = vrec(ub_v) 
```





## vabs
### 函数说明  
向量元素绝对值运算指令。

### 参数说明  
- src: input, Tensor("UB", "FP16"/"FP32"/"INT32", format="ND") 

### 返回值
- out: output, Tensor。

### sample0  
```python
ub_input = Tensor("UB", "FP32", [1024], format="ND")
ub_output = vabs(ub_input)  
```





## vnot
### 函数说明
向量逻辑非运算指令。
### 参数说明
- src: input, Tensor("UB", "UINT16"/"INT16", format="ND") 

### 返回值
- out: output, Tensor。

### sample0
```python
ub_not_v = vnot(ub_v)
```





## vconv

### 函数说明  
向量Cast类型转换运算（vector conversion ），仅支持：
1. INT8 <==> FP16
2. FP16 <==> FP32
3. FP16 <==> INT32
4. FP32 <==> INT32

### 参数说明  
- src: input, Tensor("UB", "FP16"/"FP32"/"INT8"/"INT32", format="ND") 
- dtype: target data type for conversion  

### 返回值
- out: output, Tensor，数据类型为dtype

### sample0  
```python
ub_m = move_to_ub(gm_m)  # Tensor("UB", "FP32", ...)
ub_m_int32 = vconv(ub_m, "INT32")
```





## vcmax
### 函数说明
向量规约最大值指令，对一个张量的某一轴元素进行比较并返回最大值。
支持指定reduce_axis进行降维操作，当reduce_axis不为负时表示从0开始计数，而当为负时表示从末尾开始计数。

### 参数说明
- src: input, Tensor("GM"/"UB",任意Dtype,任意Shape, format="ND", multi_core=True/False)
- reduce_axis: param, 整数类型，表示在哪个轴上进行减少（降维）操作。当negative时会自动调整为正数索引
### 返回值
- out: output, Tensor("UB", 与src一致Dtype, Reduce后的Shape)  


### sample0
```python
gm_input = Tensor("GM", "FP32", [256, 256], format="ND")
ub_input = move_to_ub(gm_input) # Tensor 
ub_output = vcmax(ub_input, reduce_axis=-1) # output shape [256, 1]
```





## vcmin
### 函数说明
向量规约最小值操作，对一个张量的某一轴元素进行比较并返回最大值。
### 参数说明
- src: input, Tensor("UB", 任意Dtype, 任意Shape)
- reduce_axis: param, 需要减少的一个维度（支持负轴）
### 返回值
- out: output, Tensor("UB", 与src一致Dtype, Reduce后的Shape)

### sample0
```python
# 假设ub_tensor = Tensor("UB", "FP32", [16], format="ND")
ub_min = vcmin(ub_tensor, reduce_axis=0)  # 输出形状为[1]
```





## vcadd
### 函数说明
向量规约加法操作，对一个张量的某一轴元素进行求和并累加值。
### 参数说明
- src: input, Tensor("UB", 任意Dtype, format="ND")
- reduce_axis: param, 需要进行加法操作的轴（支持负索引）
### 返回值
- out: output, 与src类型一致，但在指定轴上已被压缩后的结果
### sample0
```python
gm_input = Tensor("GM", "FP16", [256, 256], format="ND")
ub_input = move_to_ub(gm_input) # Tensor 
ub_output = vcadd(ub_input, reduce_axis=-1) # output shape [256, 1]
```






## vbrcb

### 函数说明  
向量广播指令，将源张量沿指定轴进行广播扩展，以匹配指定大小。支持多核处理，但默认单核执行。

### 参数说明  
- src: input, Tensor("UB"/"GM", 任意Dtype, format="ND")  
- broadcast_axis: param, 广播扩展的轴索引，可为负数（从后往前计数）  
- broad_size: param, 扩展后的维度大小  

### 返回值  
返回一个新的Tensor对象
- out: output, 与src具有相同dtype、mem_type和format，但shape已被广播扩展  


### sample0  
```python
# 示例：将一个形状为[1024] 的张量沿第0轴扩展到2048元素
src_tensor = Tensor("UB", "FP32", [1024, 1], format="ND")
out_tensor = vbrcb(src_tensor, 1, 32)
print(out_tensor.size)  # 输出应为[1024, 32]
```





## vector_dup
### 函数说明
向量复制指令，将标量因子复制到指定大小的新向量中。（vector duplicate）
### 参数说明
- factor: input, 必须为Scalar类型
- size: param，表示输出向量的大小, list of int
### 返回值
- out: output, Tensor("UB", ...) dtype与factor一致, size与param一致

### sample0
```python
factor = Scalar("FP32", 2.5)
new_vector = vector_dup(factor, [1024], multi_core=False) #[1024]
```


## vcmpv
### 函数说明
向量比较指令，用于逐个元素比较，支持多种比较操作，支持广播扩展。
### 参数说明
- src0: input, Tensor("UB", 任意Dtype, format="ND")
- src1: input, Tensor("UB", 任意Dtype, format="ND"), 数据类型和src0一致。
- opType: input, 操作类型，支持["EQ", "NE", "LT", "GT", "GE", "LE"]。
### 返回值
- out: output, Tensor("UB", Dtype="BOOL"), shape和广播扩展后的shape一致。

### sample0
```python
x_ub = Tensor("UB", "FP32", [1024], format="ND")
y_ub = Tensor("UB", "FP32", [512], format="ND")
ub_out = vcmpv(x_ub, y_ub, "EQ")
```

## vcmpvs
### 函数说明
向量比较指令，用于张量的逐个元素和标量的比较，支持多种比较操作。
### 参数说明
- src: input, Tensor("UB", 任意Dtype, format="ND")
- factor: input, 必须为Scalar类型，数据类型和src一致。
- opType: input, 操作类型，支持["EQ", "NE", "LT", "GT", "GE", "LE"]。
### 返回值
- out: output, Tensor("UB", Dtype="BOOL"), shape和src的shape一致。

### sample0
```python
x_ub = Tensor("UB", "FP32", [1024], format="ND")
factor = Scalar("FP32", 2.5)
ub_out = vcmpv(x_ub, factor, "EQ")
```

## where
### 函数说明
向量选择指令，用于依据给定的条件，从两个张量中选取元素，支持输入参数condition广播扩展。
### 参数说明
- src0: input, Tensor("UB", 任意Dtype, format="ND")
- src1: input, Tensor("UB", format="ND"), 数据类型和src0一致。
- condition: input，当condition中的元素为True,结果张量对应位置元素取自src0，当condition中的元素为False,
结果张量对应位置的元素取自src1。
### 返回值
- out: output, Tensor("UB"), 数据类型和src0，src1保持一致，shape和广播扩展后的shape一致。

### sample0
```python
src0 = Tensor("UB", "FP32", [512], format="ND")
src1 = Tensor("UB", "FP32", [512], format="ND")
condition = Tensor("UB", "BOOL", [1024], format="ND")
out = where(src0, src1, condition)
```