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