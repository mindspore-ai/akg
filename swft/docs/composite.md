# composite.py



## tanh
### 函数说明
向量按元素取tanh激活函数。
### 参数说明
- src: input, Tensor("UB", "FP16"/"FP32", format="ND") 

### 返回值
- out: output, Tensor，dtype与src一致。

### sample0
```python
ub_out = tanh(ub_v)
```

## arange
### 函数说明
用于创建一维张量，其元素为等差数列。
### 参数说明
- start: input, 序列的起始值，类型为int。
- stop: input, 序列的结束值，类型为int。
- step: input, 序列的步长，类型为int, 默认为1。
- dtype: input, 张量的数据类型，默认值为"INT32"。

### 返回值
- out: output, 一维Tensor，从start到end(不包含end),步长为step,数据类型为dtype的一维张量。

### sample0
```python
out = arange(0, 533, 1, "INT32")
```