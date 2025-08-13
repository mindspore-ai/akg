# SWFT 开发建议与提示

## SWFT 错误示例与推荐示例

### 错误示例1：SWFT不支持int8到float32类型之间的直接转换

```python
offset_fp32 = vconv(offset_int8, "FP32")
```

### 推荐示例1：使用中间整数类型

```python
# 先将int8扩展为float16，再转为float32
offset_fp16 = vconv(offset_int8, "FP16")
offset_fp32 = vconv(offset_fp16, "FP32")
```

### 错误示例2：Tensor的shape维度和数据搬移时的shape维度不相同

```python
gm_input = Tensor("GM", "FP16", [1, 2688], "ND", False)
ub_input = slice_to_ub(gm_input, [input_start], [ELEMS_PER_CORE])
```

### 推荐示例2：保持Tensor的shape维度和数据搬移时的shape维度相同，使用1占位

```python
gm_input = Tensor("GM", "FP16", [1, 2688], "ND", False)
ub_input = slice_to_ub(gm_input, [1, input_start], [0, ELEMS_PER_CORE])
```

### 错误示例3：计算1/x时使用了vrec，而使用vrec会导致精度有问题

```python 
output_ub = vrec(input_ub)  # 1/x
```

### 推荐示例3：将1/x转换为x/(x*x)计算，组合vsub和vdiv以避免使用vrec，精度正确

```python 
pow_ub = vmul(input_ub, input_ub)  # x*x
output_ub = vdiv(input_ub, pow_ub)  # x/(x*x) = 1/x
```

### 推荐示例4：reduce的结果为标量时，使用向量-标量运算替代后向的broadcast操作，注意仅仅在双目运算中使用
```python 
ub_max = vcmax(ub_input, reduce_axis=-1)
ub_exp = vexp(ub_max)
ub_sub = vsubs(ub_input, move_to_scalar(ub_exp))
```

### 推荐示例5：为了保证结果的正确性，累加操作必须转为fp32
```python 
ub_input_fp32 = vconv(ub_input, "FP32")
ub_sum = vcadd(ub_input_fp32, reduce_axis=-1)
```

### 推荐示例6：如果生成的cce过长（也就是所有嵌套for循环的乘积过大），将for循环中的range替换为dynamic_loop
```python 
for i in dynamic_loop(SAMPLES_PER_CORE):
```