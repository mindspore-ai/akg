最终文件必须是**单一自包含 Python 文件**，包含以下 4 个部分：

```python
# 1. Imports 区（只允许标准库和 PyTorch 相关包）
import torch
import torch.nn as nn

# 2. Model 类
class Model(nn.Module):
    def __init__(self, <init_params>):
        super(Model, self).__init__()

    def forward(self, <forward_inputs>) -> torch.Tensor:
        return output

# 3. get_inputs()：返回 forward() 的输入参数列表
def get_inputs():
    input1 = torch.randn(batch_size, dim)
    input2 = torch.randn(batch_size, dim)
    return [input1, input2]

# 4. get_init_inputs()：返回 __init__() 的初始化参数列表
def get_init_inputs():
    return [dim_value]
```