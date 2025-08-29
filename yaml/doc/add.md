# add算子

## 描述

`add`算子对两个输入做逐元素加法，支持广播、隐式类型转换与类型提升。其计算形式为：

$$
\begin{array}{ll} 
    out_i = x_i + y_i
\end{array}
$$

## 输入参数

| Name   | DType                                 | Shape                 | Description |
|--------|----------------------------------------|-----------------------|-------------|
| x  | Tensor[float32/float16/int8/int32]                | 与 `y` 可广播     | 第一个输入 |
| y  | Tensor[float32/float16/int8/int32]                | 与 `x` 可广播     | 第二个输入 |

## 输出参数

| Name  | DType                      | Shape                | Description |
|-------|----------------------------|----------------------|-------------|
| out   | Tensor[float32/float16/int8/int32] | 与广播后shape一致     | 逐元素加法结果 |

## 使用示例

```python
import mindspore as ms
import ms_custom_ops

x = ms.tensor([1., 2., 3.])
y = ms.tensor([4., 5., 6.])
out = ms_custom_ops.add(x, y)
print(out)
# [5. 7. 9.]
```
