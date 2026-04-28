import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, normalized_shape=4096, eps=1e-6):
        super(Model, self).__init__()
        # In a real RMSNorm implementation, we would store the normalized_shape
        self.normalized_shape = normalized_shape
        self.eps = eps

    def forward(self, input_tensor, weight):
        # RMSNorm (Root Mean Square Layer Normalization)
        # This operation is commonly used in neural networks for:
        # - Normalizing activations in transformer models
        # - Providing an alternative to LayerNorm with potentially better performance
        # - Used in models like T5 and some variants of LLaMA
        
        # Calculate root mean square
        rms = torch.sqrt(torch.mean(input_tensor ** 2, dim=-1, keepdim=True) + self.eps)
        
        # Normalize and apply weight
        result = input_tensor / rms * weight
        
        return result


def get_inputs_dyn_list():
    # Case 1: Small
    input_tensor1 = torch.randn(512, 512, dtype=torch.float32)
    weight1 = torch.ones(512, dtype=torch.float32)

    # Case 2: Middle
    input_tensor2 = torch.randn(2048, 4096, dtype=torch.float32)
    weight2 = torch.ones(4096, dtype=torch.float32)

    # Case 3: Large
    input_tensor3 = torch.randn(4096, 4096, dtype=torch.float32)
    weight3 = torch.ones(4096, dtype=torch.float32)

    # Case 4: Non-aligned
    input_tensor4 = torch.randn(1536, 2688, dtype=torch.float32)
    weight4 = torch.ones(2688, dtype=torch.float32)

    return [
        [input_tensor1, weight1],
        [input_tensor2, weight2],
        [input_tensor3, weight3],
        [input_tensor4, weight4]
    ]

def get_init_inputs():
    normalized_shape = 4096
    eps = 1e-6
    return [normalized_shape, eps]