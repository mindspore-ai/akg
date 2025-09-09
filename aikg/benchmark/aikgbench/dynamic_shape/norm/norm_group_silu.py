import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, num_groups=32):
        super(Model, self).__init__()
        self.num_groups = num_groups

    def forward(self, input_tensor):
        # GroupNormSilu operation
        # This operation is commonly used in neural networks for:
        # - Normalizing activations in groups
        # - Applying SiLU activation function
        # - Used in models like EfficientNet and some transformer variants
        
        # Apply GroupNorm
        group_norm = torch.nn.functional.group_norm(input_tensor, self.num_groups)
        
        # Apply SiLU activation
        result = torch.nn.functional.silu(group_norm)
        
        return result


def get_inputs_dyn_list():
    # Case 1: Small
    input_tensor1 = torch.randn(16, 512, 28, 28, dtype=torch.float32)

    # Case 2: Middle
    input_tensor2 = torch.randn(32, 1344, 56, 56, dtype=torch.float32)

    # Case 3: Large
    input_tensor3 = torch.randn(64, 2048, 112, 112, dtype=torch.float32)

    # Case 4: Non-aligned
    input_tensor4 = torch.randn(48, 2688, 64, 64, dtype=torch.float32)

    return [
        input_tensor1,
        input_tensor2,
        input_tensor3,
        input_tensor4
    ]

def get_init_inputs():
    # Parameters for GroupNormSilu operation
    num_groups = 32  # Number of groups for GroupNorm
    return [num_groups]