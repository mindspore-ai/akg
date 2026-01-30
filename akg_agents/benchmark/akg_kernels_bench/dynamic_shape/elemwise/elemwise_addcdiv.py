import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, value=1.0):
        super(Model, self).__init__()
        self.value = value

    def forward(self, input_tensor, tensor1, tensor2):
        # torch.addcdiv(input, tensor1, tensor2, *, value=1, out=None)
        # Performs the element-wise division of tensor1 by tensor2, multiplies the result by value, and adds it to input.
        # This operation is commonly used in neural networks for:
        # - Implementing specific mathematical formulas
        # - Normalization operations
        return torch.addcdiv(input_tensor, tensor1, tensor2, value=self.value)


def get_inputs_dyn_list():
    # Case 1: Small (batch=256, hidden=512)
    input_tensor1 = torch.randn(256, 512, dtype=torch.float32)
    tensor1_1 = torch.randn(256, 512, dtype=torch.float32)
    tensor2_1 = torch.randn(256, 512, dtype=torch.float32)

    # Case 2: Middle (batch=1024, hidden=4096)
    input_tensor2 = torch.randn(1024, 4096, dtype=torch.float32)
    tensor1_2 = torch.randn(1024, 4096, dtype=torch.float32)
    tensor2_2 = torch.randn(1024, 4096, dtype=torch.float32)

    # Case 3: Large (batch=2048, hidden=4096)
    input_tensor3 = torch.randn(2048, 4096, dtype=torch.float32)
    tensor1_3 = torch.randn(2048, 4096, dtype=torch.float32)
    tensor2_3 = torch.randn(2048, 4096, dtype=torch.float32)

    # Case 4: Non-aligned (batch=768, hidden=2688)
    input_tensor4 = torch.randn(768, 2688, dtype=torch.float32)
    tensor1_4 = torch.randn(768, 2688, dtype=torch.float32)
    tensor2_4 = torch.randn(768, 2688, dtype=torch.float32)

    return [
        [input_tensor1, tensor1_1, tensor2_1],
        [input_tensor2, tensor1_2, tensor2_2],
        [input_tensor3, tensor1_3, tensor2_3],
        [input_tensor4, tensor1_4, tensor2_4]
    ]


def get_init_inputs():
    # Parameters for addcdiv
    value = 1.0  # Scale factor
    return [value]