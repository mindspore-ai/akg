# ModelNew 格式规范 — Triton + MindSpore

适用于 `dsl` 为 `triton_cuda` 或 `triton_ascend`，`framework` 为 `mindspore` 的场景。

**所有生成的代码必须是 ModelNew 类格式，不使用函数形式。**

## 1. ModelNew 类模板

```python
import mindspore as ms
from mindspore import nn
import triton
import triton.language as tl

@triton.jit
def {op_name}_kernel(...):
    # kernel 实现
    ...

class ModelNew(nn.Cell):
    def __init__(self, ...):
        super().__init__()
        # 如果有 nn.Cell 参数（如 nn.Dense, nn.Conv2d），需要：
        # 1. 在 __init__ 开始时设置固定随机种子（与验证模板中的种子一致）
        # 2. 通过 nn.Dense/nn.Conv2d 等构建参数，然后提取 weight 和 bias
        # 3. 使用 Parameter 包装，确保与原始 Model 的权重一致
        ms.set_seed(0)  # 固定种子（与 kernel_verify_template.j2 中的种子一致）
        # 例如：
        # from mindspore import Parameter
        # dense = nn.Dense(in_features, out_features)
        # self.weight = Parameter(dense.weight.clone(), name="weight")
        # self.bias = Parameter(dense.bias.clone(), name="bias") if dense.bias is not None else None

    def construct(self, ...):
        # 调用 kernel 函数
        return {op_name}_kernel(...)
```

**注意**：以上 `self` 属性只针对于需要用到内置参数的实现，对于使用 `get_inputs` 或 `get_init_inputs` 传入的参数，需要直接调用。

## 2. 无参数算子

如果算子没有可学习参数（如 ReLU），`__init__` 可以为空：

```python
class ModelNew(nn.Cell):
    def __init__(self):
        super().__init__()

    def construct(self, x):
        return xxx_kernel(x)
```

## 3. 有参数算子

对于有可学习参数的算子（如 Dense, Conv2d），必须在 `__init__` 中通过固定随机种子构建参数：

```python
class ModelNew(nn.Cell):
    def __init__(self, in_features, out_features):
        super().__init__()
        ms.set_seed(0)  # 固定种子，确保与原始 Model 权重一致
        from mindspore import Parameter
        dense = nn.Dense(in_features, out_features)
        self.weight = Parameter(dense.weight.clone(), name="weight")
        self.bias = Parameter(dense.bias.clone(), name="bias") if dense.bias is not None else None

    def construct(self, x):
        return xxx_kernel(x, self.weight, self.bias)
```

**注意**：以上 `self` 属性只针对于需要用到内置参数的实现，对于使用 `get_inputs` 或 `get_init_inputs` 传入的参数，需要直接调用。

## 4. Shape 参数获取

对于任务输入中固定写死的 `init_inputs` 参数，以及 class 外定义的参数，硬编码至代码中。
所需要的 shape 参数，要从 inputs 的数据形状中获取，以适应不同的输入（**重要**：需要仔细检查输入的变量和 shape 与代码中一一对应）：

```python
def construct(self, input_tensor, ...):
    # 硬编码的参数（如果有无法从 inputs 中获取的参数）
    # args = ...  # 无法从 inputs 中获取的参数信息硬编码于此

    # 从输入张量获取 shape 参数
    P1, P2, P3 = input_tensor.shape  # 变量名应该与 inputs 构造时对应的变量保持一致
    ...
    # 执行 kernel 函数
    ...
```

## 5. 卷积类算子注意事项

如果检测到给出的任务是卷积类的算子任务，为了保证 ModelNew 的卷积核权重与当前任务代码的卷积核权重一致，需要在 ModelNew 的 `__init__` 中通过固定随机种子构建参数：

```python
class ModelNew(nn.Cell):
    def __init__(self, in_channels, out_channels, kernel_size, ...):
        super().__init__()
        ms.set_seed(0)  # 固定种子，确保与原始 Model 权重一致
        from mindspore import Parameter
        # 创建 Conv 层并提取 weight 和 bias
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, ...)
        self.weight = Parameter(conv.weight.clone(), name="weight")
        self.bias = Parameter(conv.bias.clone(), name="bias") if conv.bias is not None else None

    def construct(self, x):
        return conv_kernel(x, self.weight, self.bias, ...)
```

请务必保证 `nn` 中调用的 module 要与任务代码中调用的 module 要一致，使用的参数要一致。
