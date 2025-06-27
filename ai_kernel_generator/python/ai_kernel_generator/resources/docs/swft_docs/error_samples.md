# SWFT 错误示例与推荐示例

## 错误示例1：SWFT不支持int8到float32类型之间的直接转换

```python
offset_fp32 = vconv(offset_int8, "FP32")
```

## 推荐示例1：使用中间整数类型

```python
# 先将int8扩展为float16，再转为float32
offset_fp16 = vconv(offset_int8, "FP16")
offset_fp32 = vconv(offset_fp16, "FP32")
```

## 错误示例2：Tensor的shape维度和数据搬移时的shape维度不相同

```python
gm_input = Tensor("GM", "FP16", [1, 2688], "ND", False)
ub_input = slice_to_ub(gm_input, [input_start], [ELEMS_PER_CORE])
```

## 推荐示例2：保持Tensor的shape维度和数据搬移时的shape维度相同，使用1占位

```python
gm_input = Tensor("GM", "FP16", [1, 2688], "ND", False)
ub_input = slice_to_ub(gm_input, [1, input_start], [0, ELEMS_PER_CORE])
```

## 错误示例3：slice_to_ub，vmul，vadd等API没有返回值

```python
slice_to_ub(gm_input, begin=[current_batch, 0], slicesize=[1, DIM])
vmul(input_ub, tmp_ub)
vadd(input_ub, tmp_ub)
```
## 推荐示例3：slice_to_ub，vmul，vadd等API有返回值

```python
input_ub = slice_to_ub(gm_input, begin=[current_batch, 0], slicesize=[1, DIM])
mul_ub = vmul(input_ub, tmp_ub)
add_ub = vadd(input_ub, tmp_ub)
```

## 错误示例4：insert_to_gm缺少参数

```python 
insert_to_gm(gm_output, output_ub, begin=[start_batch, 0])
```
## 推荐示例4：insert_to_gm有4个入参

```python 
insert_to_gm(gm_output, output_ub, begin=[start_batch, 0], slicesize=[SAMPLES_PER_CORE, DIM])
```

## 错误示例5：计算1/x时使用了vrec，而使用vrec会导致精度有问题

```python 
output_ub = vrec(input_ub)  # 1/x
```

## 推荐示例5：将1/x转换为x/(x*x)计算，组合vsub和vdiv以避免使用vrec，精度正确

```python 
pow_ub = vmul(input_ub, input_ub)  # x*x
output_ub = vdiv(input_ub, pow_ub)  # x/(x*x) = 1/x
```