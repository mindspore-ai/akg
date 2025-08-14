# SWFT 开发建议与提示

## 1. reduce类算子

### reduce_y
reduce API仅支持最后一根轴时reduce轴，使用for循环和向量二元操作指令（如vadd, vmin, vmax等）替代reduce API
```python 
init_fp16 = Scalar("FP16", 0.0)
reduce_buf = vector_dup(init_fp16, [1, 1, DIM2], False)
for j in range(DIM1):
    ub_input = slice_to_ub(gm_input, [current_batch, j, 0], [1, 1, DIM2])
    reduce_buf = vadd(reduce_buf, ub_input)
```

### reduce_x
- reduce轴小于32字节：不管非reduce轴的大小，block_dim均为1
- reduce轴大于32字节：非reduc轴映射到多核之后，需要保证每个核上非reduce轴是32字节对齐的
- reduce轴超大：UB无法计算全部的reduce，需要将reduce轴进行切分，使用for循环和向量二元操作指令计算部分值之后再规约
```python 
init_fp32 = Scalar("FP32", 0.0)
ub_tmp = vector_dup(init_fp32, [rows_per_core, static_cols], False)
for j in range(DIM1):
    ub_input = slice_to_ub(gm_input, [start_row, start_cols], [rows_per_core, static_cols])
    ub_input_fp32 = vconv(ub_input, "FP32")
    ub_tmp = vadd(ub_tmp, ub_input_fp32)
ub_output = vcadd(ub_tmp, reduce_axis=reduce_axis)
```

### 连续多个reduce轴
reduce API仅支持一根轴reduce轴，使用reshape将连续多个reduce轴转换为一根轴


### reduce 后向融合 
reduce的结果为标量时，使用向量-标量运算替代后向的broadcast操作，注意仅仅在双目运算中使用
```python 
ub_max = vcmax(ub_input, reduce_axis=-1)
ub_exp = vexp(ub_max)
ub_exp_scalar = move_to_scalar(ub_exp)
ub_sub = vsubs(ub_input, ub_exp_scalar)
```

### 2. 精度保障
- 将1/x转换为x/(x*x)计算，禁止使用vrec
```python 
ub_mul = vmul(x, x)
ub_output = vdiv(x, ub_mul)
```
- 累加，累乘，tanh，除法等操作使用高精度（FP32）防止精度丢失
```python 
ub_input_fp32 = vconv(ub_input, "FP32")
ub_sum = vcadd(ub_input_fp32, reduce_axis=-1)
```

## 3. API 使用要求

### 数据切片维度匹配：slice_to_ub等搬移操作的维度必须与Tensor声明维度严格一致
错误示例​：Tensor是2D [1,2688]，切片参数却用1D [input_start]
​正确做法​：用1或0占位保持维度对齐
```python
gm_input = Tensor("GM", "FP16", [1, 2688], "ND", False)
ub_input = slice_to_ub(gm_input, [1, input_start], [0, ELEMS_PER_CORE])
```

### 其他 
- 仅支持特定类型间的转换，具体参考vconv的API文档
- 如果生成的cce过长（也就是所有嵌套for循环的乘积过大），将for循环中的range替换为dynamic_loop
- 禁止使用嵌套函数调用（funcA(funcB())）
