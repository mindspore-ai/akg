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

    def forward(self, input, weight, bias):
        # Linear transformation: input @ weight^T
        
        # Transpose weight matrix
        weight_t = weight.t()
        
        # Perform matrix multiplication
        result = torch.matmul(input, weight_t)
        
        # Add bias
        result = result + bias
            
        return result

def get_inputs_dyn_list():
    # Linear transformation with large shapes variation cases with both aligned and non-aligned shapes
    
    # Case 1: Small tensor size 15x15 (non-aligned)
    input1 = torch.randn(15, 15, dtype=torch.float16)
    weight1 = torch.randn(15, 15, dtype=torch.float16)
    bias1 = torch.randn(15, dtype=torch.float16)
    
    # Case 2: Small tensor size 31x31 (non-aligned)
    input2 = torch.randn(31, 31, dtype=torch.float16)
    weight2 = torch.randn(31, 31, dtype=torch.float16)
    bias2 = torch.randn(31, dtype=torch.float16)
    
    # Case 3: Small tensor size 32x32 (aligned)
    input3 = torch.randn(32, 32, dtype=torch.float16)
    weight3 = torch.randn(32, 32, dtype=torch.float16)
    bias3 = torch.randn(32, dtype=torch.float16)
    
    # Case 4: Medium tensor size 127x127 (non-aligned)
    input4 = torch.randn(127, 127, dtype=torch.float16)
    weight4 = torch.randn(127, 127, dtype=torch.float16)
    bias4 = torch.randn(127, dtype=torch.float16)
    
    # Case 5: Medium tensor size 128x128 (aligned)
    input5 = torch.randn(128, 128, dtype=torch.float16)
    weight5 = torch.randn(128, 128, dtype=torch.float16)
    bias5 = torch.randn(128, dtype=torch.float16)
    
    # Case 6: Large tensor size 511x511 (non-aligned)
    input6 = torch.randn(511, 511, dtype=torch.float16)
    weight6 = torch.randn(511, 511, dtype=torch.float16)
    bias6 = torch.randn(511, dtype=torch.float16)
    
    # Case 7: Large tensor size 512x512 (aligned)
    input7 = torch.randn(512, 512, dtype=torch.float16)
    weight7 = torch.randn(512, 512, dtype=torch.float16)
    bias7 = torch.randn(512, dtype=torch.float16)
    
    # Case 8: Very large tensor size 1023x1023 (non-aligned)
    input8 = torch.randn(1023, 1023, dtype=torch.float16)
    weight8 = torch.randn(1023, 1023, dtype=torch.float16)
    bias8 = torch.randn(1023, dtype=torch.float16)
    
    # Case 9: Very large tensor size 1024x1024 (aligned)
    input9 = torch.randn(1024, 1024, dtype=torch.float16)
    weight9 = torch.randn(1024, 1024, dtype=torch.float16)
    bias9 = torch.randn(1024, dtype=torch.float16)
    
    # Case 10: Extreme tensor size 4095x4095 (non-aligned)
    input10 = torch.randn(4095, 4095, dtype=torch.float16)
    weight10 = torch.randn(4095, 4095, dtype=torch.float16)
    bias10 = torch.randn(4095, dtype=torch.float16)
    
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
    has_bias = True      # Whether to include bias term
    transpose_b = True   # Whether to transpose the weight matrix
    return [has_bias, transpose_b]