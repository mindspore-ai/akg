import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Fast Softmax Gradient operation for backpropagation.
    This operation is commonly used in neural networks for:
    - Computing gradients of softmax operation during backpropagation
    - Optimized version of standard softmax gradient computation
    - Used in efficient transformer implementations
    
    Formula: gradient = softmax_output * (grad_output - sum(softmax_output * grad_output))
    """
    def __init__(self, dim=-1):
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, softmax_output, grad_input):
        # Fast softmax gradient operation
        # softmax_output: output of softmax
        # grad_input: gradient from next layer
        
        # Compute softmax gradient: output * (grad_input - sum(output * grad_input))
        sum_grad = torch.sum(softmax_output * grad_input, dim=self.dim, keepdim=True)
        result = softmax_output * (grad_input - sum_grad)
        return result

def get_inputs_dyn_list():
    # Fast Softmax Gradient variation cases with both aligned and non-aligned shapes
    
    # Case 1:
    softmax_output1 = torch.randn(16, 4, 128, 1024, dtype=torch.float16)
    grad_input1 = torch.randn(16, 4, 128, 1024, dtype=torch.float16)

    # Case 2:
    softmax_output2 = torch.randn(32, 8, 256, 4096, dtype=torch.float16)
    grad_input2 = torch.randn(32, 8, 256, 4096, dtype=torch.float16)

    # Case 3:
    softmax_output3 = torch.randn(17, 16, 512, 8192, dtype=torch.float16)
    grad_input3 = torch.randn(17, 16, 512, 8192, dtype=torch.float16)

    # Case 4:
    softmax_output4 = torch.randn(64, 16, 1024, 4096, dtype=torch.float16)
    grad_input4 = torch.randn(64, 16, 1024, 4096, dtype=torch.float16)


    return [
        [softmax_output1, grad_input1],
        [softmax_output2, grad_input2],
        [softmax_output3, grad_input3],
        [softmax_output4, grad_input4]
    ]


def get_init_inputs():
    # Parameters for Fast Softmax Gradient operation
    dim = -1  # Dimension along which to apply softmax gradient
    return [dim]