import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input, weight, bias):
        # Matrix multiplication with bias addition
        # torch.matmul(input, weight) + bias
        # This operation is commonly used in neural networks for:
        # - Linear transformations in fully connected layers
        # - Attention computations in transformers
        # - Convolution operations when expressed as matrix multiplication
        return torch.matmul(input, weight) + bias


def get_inputs_dyn_list():
    # Matrix multiplication with bias addition using realistic shapes for large models
    
    # Case 1: Small batch, small dimensions
    input1 = torch.randn(4, 256, dtype=torch.float16)
    weight1 = torch.randn(256, 512, dtype=torch.float16)
    bias1 = torch.randn(4, 512, dtype=torch.float16)
    
    # Case 2: Small batch, medium dimensions
    input2 = torch.randn(8, 512, dtype=torch.float16)
    weight2 = torch.randn(512, 1024, dtype=torch.float16)
    bias2 = torch.randn(8, 1024, dtype=torch.float16)
    
    # Case 3: Medium batch, medium dimensions
    input3 = torch.randn(16, 1024, dtype=torch.float16)
    weight3 = torch.randn(1024, 2048, dtype=torch.float16)
    bias3 = torch.randn(16, 2048, dtype=torch.float16)
    
    # Case 4: Medium batch, large dimensions
    input4 = torch.randn(32, 2048, dtype=torch.float16)
    weight4 = torch.randn(2048, 4096, dtype=torch.float16)
    bias4 = torch.randn(32, 4096, dtype=torch.float16)
    
    # Case 5: Large batch, small dimensions
    input5 = torch.randn(64, 256, dtype=torch.float16)
    weight5 = torch.randn(256, 512, dtype=torch.float16)
    bias5 = torch.randn(64, 512, dtype=torch.float16)
    
    # Case 6: Large batch, medium dimensions
    input6 = torch.randn(128, 1024, dtype=torch.float16)
    weight6 = torch.randn(1024, 2048, dtype=torch.float16)
    bias6 = torch.randn(128, 2048, dtype=torch.float16)
    
    # Case 7: Large batch, large dimensions
    input7 = torch.randn(192, 2048, dtype=torch.float16)
    weight7 = torch.randn(2048, 4096, dtype=torch.float16)
    bias7 = torch.randn(192, 4096, dtype=torch.float16)
    
    # Case 8: Special case: small batch, very large dimensions
    input8 = torch.randn(2, 4096, dtype=torch.float16)
    weight8 = torch.randn(4096, 8192, dtype=torch.float16)
    bias8 = torch.randn(2, 8192, dtype=torch.float16)
    
    # Case 9: Special case: large batch, small dimensions
    input9 = torch.randn(256, 128, dtype=torch.float16)
    weight9 = torch.randn(128, 256, dtype=torch.float16)
    bias9 = torch.randn(256, 256, dtype=torch.float16)
    
    # Case 10: Balanced case
    input10 = torch.randn(96, 1024, dtype=torch.float16)
    weight10 = torch.randn(1024, 2048, dtype=torch.float16)
    bias10 = torch.randn(96, 2048, dtype=torch.float16)
    
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
    # No parameters needed for matmul with addition
    return []