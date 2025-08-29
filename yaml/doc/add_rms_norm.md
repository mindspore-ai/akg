# add_rms_norm算子

## 描述

`add_rms_norm`算子是将`Add`与`RmsNorm`融合的算子，用于减少数据在内存中来回搬运的开销。其计算公式为：


$$
\begin{array}{ll} 
    x_i = x_{1i} + x_{2i} \\ 
    y_i = \text{RmsNorm}(x_i) = \frac{x_i}{\sqrt{\frac{1}{n}\sum_{i=1}^{n}{x_i^2} + \varepsilon}} \gamma_i 
\end{array}
$$


## 输入参数

| Name  | DType                           | Shape  | Description |
|-------|---------------------------------|--------|-------------|
| x1    | Tensor[float16/float32/bfloat16] | 与 `x2` 相同 | 第一个输入张量 |
| x2    | Tensor[float16/float32/bfloat16] | 与 `x1` 相同 | 第二个输入张量，需与 `x1` shape相同 |
| gamma | Tensor[float16/float32/bfloat16] | 与归一化维度对齐，通常为最后一维大小（可广播） | 可学习参数 gamma |
| epsilon (optional) | float | - | 防止除零的微小常数，取值范围 (0, 1]，默认 `1e-6` |

## 输出参数

| Name   | DType                           | Shape            | Description |
|--------|---------------------------------|------------------|-------------|
| y      | Tensor[float16/float32/bfloat16] | 与 `x1` 相同     | 归一化结果 |
| rstd   | Tensor[float32]                 | 与归一化维度对齐（可广播到 `x1`） | 输入标准差的倒数，用于反向计算 |
| x | Tensor[float16/float32/bfloat16] | 与 `x1` 相同     | `x1` 与 `x2` 的逐元素和 |

## 使用示例

```python
import mindspore as ms
import numpy as np
from mindspore import Tensor
import ms_custom_ops

# 构造输入张量
x1 = Tensor(np.array([[0.5, 1.0, 1.5], [0.5, 1.0, 1.5]]), ms.float32)
x2 = Tensor(np.array([[0.5, 1.0, 1.5], [0.5, 1.0, 1.5]]), ms.float32)
gamma = Tensor(np.ones([3]), ms.float32)

# 调用算子（返回 y, rstd, x）
y, rstd, x = ms_custom_ops.add_rms_norm(x1, x2, gamma)
print(y)
# [[0.46291003  0.92582005  1.38873]
#  [0.46291003  0.92582005  1.38873]]
```
