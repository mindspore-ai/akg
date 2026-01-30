import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, value=1.0):
        super(Model, self).__init__()
        self.value = value

    def forward(self, input_tensor, tensor1, tensor2):
        # torch.addcmul(input, tensor1, tensor2, *, value=1, out=None)
        # Performs the element-wise multiplication of tensor1 by tensor2, multiplies the result by value, and adds it to input.
        # This operation is commonly used in neural networks for:
        # - Implementing specific mathematical formulas
        # - Attention mechanisms
        # - Fusion operations
        return torch.addcmul(input_tensor, tensor1, tensor2, value=self.value)


def get_inputs_dyn_list():
    # Addcmul variation cases with both aligned and non-aligned shapes
    
    # Case 1: 16-aligned batch, 16-aligned hidden
    # Shape (256, 4096) represents a batch of 256 samples with 4096 features each
    input_tensor1 = torch.randn(256, 4096, dtype=torch.float32)
    tensor1_1 = torch.randn(256, 4096, dtype=torch.float32)
    tensor2_1 = torch.randn(256, 4096, dtype=torch.float32)
    
    # Case 2: Non-16-aligned batch, 16-aligned hidden
    # Shape (125, 5120) represents a batch of 125 samples with 5120 features each
    input_tensor2 = torch.randn(125, 5120, dtype=torch.float32)
    tensor1_2 = torch.randn(125, 5120, dtype=torch.float32)
    tensor2_2 = torch.randn(125, 5120, dtype=torch.float32)
    
    # Case 3: 16-aligned batch, non-16-aligned hidden
    # Shape (512, 6144) represents a batch of 512 samples with 6144 features each
    input_tensor3 = torch.randn(512, 6144, dtype=torch.float32)
    tensor1_3 = torch.randn(512, 6144, dtype=torch.float32)
    tensor2_3 = torch.randn(512, 6144, dtype=torch.float32)
    
    # Case 4: Large batch size
    # Shape (1024, 8192) represents a batch of 1024 samples with 8192 features each
    input_tensor4 = torch.randn(1024, 8192, dtype=torch.float32)
    tensor1_4 = torch.randn(1024, 8192, dtype=torch.float32)
    tensor2_4 = torch.randn(1024, 8192, dtype=torch.float32)
    
    return [
        [input_tensor1, tensor1_1, tensor2_1],
        [input_tensor2, tensor1_2, tensor2_2],
        [input_tensor3, tensor1_3, tensor2_3],
        [input_tensor4, tensor1_4, tensor2_4]
    ]


def get_init_inputs():
    # Fixed parameters for addcmul
    value = 1.0  # Scale factor
    return [value]