import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, normalized_shape=4096, eps=1e-6):
        super(Model, self).__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps

    def forward(self, tensor1, tensor2, weight, bias):
        # AddRmsNorm operation
        # This operation is commonly used in neural networks for:
        # - Combining residual connections with RMS normalization
        # - Used in transformer architectures like T5
        # - Providing an alternative to LayerNorm with potentially better performance
        
        # Perform addition
        added = tensor1 + tensor2
        
        # Apply RMS normalization
        rms = torch.sqrt(torch.mean(added ** 2, dim=-1, keepdim=True) + self.eps)
        result = added / rms * weight + bias
        
        return result


def get_inputs_dyn_list():
    # Case 1: Small
    tensor1_1 = torch.randn(512, 512, dtype=torch.float32)
    tensor1_2 = torch.randn(512, 512, dtype=torch.float32)
    weight1 = torch.ones(512, dtype=torch.float32)
    bias1 = torch.zeros(512, dtype=torch.float32)

    # Case 2: Middle
    tensor2_1 = torch.randn(2048, 4096, dtype=torch.float32)
    tensor2_2 = torch.randn(2048, 4096, dtype=torch.float32)
    weight2 = torch.ones(4096, dtype=torch.float32)
    bias2 = torch.zeros(4096, dtype=torch.float32)

    # Case 3: Large
    tensor3_1 = torch.randn(4096, 4096, dtype=torch.float32)
    tensor3_2 = torch.randn(4096, 4096, dtype=torch.float32)
    weight3 = torch.ones(4096, dtype=torch.float32)
    bias3 = torch.zeros(4096, dtype=torch.float32)

    # Case 4: Non-aligned
    tensor4_1 = torch.randn(1536, 2688, dtype=torch.float32)
    tensor4_2 = torch.randn(1536, 2688, dtype=torch.float32)
    weight4 = torch.ones(2688, dtype=torch.float32)
    bias4 = torch.zeros(2688, dtype=torch.float32)

    return [
        [tensor1_1, tensor1_2, weight1, bias1],
        [tensor2_1, tensor2_2, weight2, bias2],
        [tensor3_1, tensor3_2, weight3, bias3],
        [tensor4_1, tensor4_2, weight4, bias4]
    ]

def get_init_inputs():
    normalized_shape = 4096
    eps = 1e-6
    return [normalized_shape, eps]