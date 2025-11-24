import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Matrix multiplication operation that performs linear transformation.
    This operation is commonly used in neural networks for:
    - Fully connected layers in neural networks
    - Projection layers in transformers
    - Used in virtually all neural network architectures
    
    Formula: output = input @ weight^T + bias (if has_bias=True)
    """
    def __init__(self, has_bias=False, transpose_b=False):
        super(Model, self).__init__()
        self.has_bias = has_bias
        self.transpose_b = transpose_b

    def forward(self, input, weight, bias=None):
        # Linear transformation: input @ weight^T
        
        # Transpose weight matrix
        weight_t = weight.t()
        
        # Perform matrix multiplication
        result = torch.matmul(input, weight_t)
        
        # Add bias
        result = result + bias
            
        return result

def get_inputs_dyn_list():
    # Linear transformation with realistic shapes for large models
    
    # Case 1: Small batch, small dimensions
    input1 = torch.randn(4, 256, dtype=torch.float16)
    weight1 = torch.randn(512, 256, dtype=torch.float16)
    bias1 = torch.randn(512, dtype=torch.float16)
    
    # Case 2: Small batch, medium dimensions
    input2 = torch.randn(8, 512, dtype=torch.float16)
    weight2 = torch.randn(1024, 512, dtype=torch.float16)
    bias2 = torch.randn(1024, dtype=torch.float16)
    
    # Case 3: Medium batch, medium dimensions
    input3 = torch.randn(16, 1024, dtype=torch.float16)
    weight3 = torch.randn(2048, 1024, dtype=torch.float16)
    bias3 = torch.randn(2048, dtype=torch.float16)
    
    # Case 4: Medium batch, large dimensions
    input4 = torch.randn(32, 2048, dtype=torch.float16)
    weight4 = torch.randn(4096, 2048, dtype=torch.float16)
    bias4 = torch.randn(4096, dtype=torch.float16)
    
    # Case 5: Large batch, small dimensions
    input5 = torch.randn(64, 256, dtype=torch.float16)
    weight5 = torch.randn(512, 256, dtype=torch.float16)
    bias5 = torch.randn(512, dtype=torch.float16)
    
    # Case 6: Large batch, medium dimensions
    input6 = torch.randn(128, 1024, dtype=torch.float16)
    weight6 = torch.randn(2048, 1024, dtype=torch.float16)
    bias6 = torch.randn(2048, dtype=torch.float16)
    
    # Case 7: Large batch, large dimensions
    input7 = torch.randn(192, 2048, dtype=torch.float16)
    weight7 = torch.randn(4096, 2048, dtype=torch.float16)
    bias7 = torch.randn(4096, dtype=torch.float16)
    
    # Case 8: Special case: small batch, very large dimensions
    input8 = torch.randn(2, 4096, dtype=torch.float16)
    weight8 = torch.randn(8192, 4096, dtype=torch.float16)
    bias8 = torch.randn(8192, dtype=torch.float16)
    
    # Case 9: Special case: large batch, small dimensions
    input9 = torch.randn(256, 128, dtype=torch.float16)
    weight9 = torch.randn(256, 128, dtype=torch.float16)
    bias9 = torch.randn(256, dtype=torch.float16)
    
    # Case 10: Balanced case
    input10 = torch.randn(96, 1024, dtype=torch.float16)
    weight10 = torch.randn(2048, 1024, dtype=torch.float16)
    bias10 = torch.randn(2048, dtype=torch.float16)
    
    return [
        [input1, weight1, bias1],
        [input2, weight2, bias2],
        [input3, weight3, bias3],
        [input4, weight4, bias4],
        [input5, weight5, bias5],
        [input6, weight6, bias6],
        [input7, weight7, bias7],
        [input8, weight8, bias8],
        [input9, weight9, bias9],
        [input10, weight10, bias10]
    ]

def get_init_inputs():
    # Fixed parameters for Linear operation
    has_bias = True     # Whether to include bias term
    transpose_b = True  # Whether to transpose the weight matrix
    return [has_bias, transpose_b]