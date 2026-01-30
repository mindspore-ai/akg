import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, transpose_a=False, transpose_b=True):
        super(Model, self).__init__()
        self.transpose_a = transpose_a
        self.transpose_b = transpose_b

    def forward(self, input, weight):
        # Transpose only weight matrix
        mat1 = input
        mat2 = weight.t()
        
        # torch.matmul(input, other, *, out=None)
        # Performs matrix multiplication of the matrices input and other.
        return torch.matmul(mat1, mat2)

def get_inputs_dyn_list():
    # Matrix multiplication with transpose B only using realistic shapes for large models
    
    # Case 1: Small batch, small dimensions
    input1 = torch.randn(4, 256, dtype=torch.float16)
    weight1 = torch.randn(512, 256, dtype=torch.float16)  # Will be transposed to 256x512
    
    # Case 2: Small batch, medium dimensions
    input2 = torch.randn(8, 512, dtype=torch.float16)
    weight2 = torch.randn(1024, 512, dtype=torch.float16)  # Will be transposed to 512x1024
    
    # Case 3: Medium batch, medium dimensions
    input3 = torch.randn(16, 1024, dtype=torch.float16)
    weight3 = torch.randn(2048, 1024, dtype=torch.float16)  # Will be transposed to 1024x2048
    
    # Case 4: Medium batch, large dimensions
    input4 = torch.randn(32, 2048, dtype=torch.float16)
    weight4 = torch.randn(4096, 2048, dtype=torch.float16)  # Will be transposed to 2048x4096
    
    # Case 5: Large batch, small dimensions
    input5 = torch.randn(64, 256, dtype=torch.float16)
    weight5 = torch.randn(512, 256, dtype=torch.float16)  # Will be transposed to 256x512
    
    # Case 6: Large batch, medium dimensions
    input6 = torch.randn(128, 1024, dtype=torch.float16)
    weight6 = torch.randn(2048, 1024, dtype=torch.float16)  # Will be transposed to 1024x2048
    
    # Case 7: Large batch, large dimensions
    input7 = torch.randn(192, 2048, dtype=torch.float16)
    weight7 = torch.randn(4096, 2048, dtype=torch.float16)  # Will be transposed to 2048x4096
    
    # Case 8: Special case: small batch, very large dimensions
    input8 = torch.randn(2, 4096, dtype=torch.float16)
    weight8 = torch.randn(8192, 4096, dtype=torch.float16)  # Will be transposed to 4096x8192
    
    # Case 9: Special case: large batch, small dimensions
    input9 = torch.randn(256, 128, dtype=torch.float16)
    weight9 = torch.randn(256, 128, dtype=torch.float16)  # Will be transposed to 128x256
    
    # Case 10: Balanced case
    input10 = torch.randn(96, 1024, dtype=torch.float16)
    weight10 = torch.randn(2048, 1024, dtype=torch.float16)  # Will be transposed to 1024x2048
    
    return [
        [input1, weight1],
        [input2, weight2],
        [input3, weight3],
        [input4, weight4],
        [input5, weight5],
        [input6, weight6],
        [input7, weight7],
        [input8, weight8],
        [input9, weight9],
        [input10, weight10]
    ]

def get_init_inputs():
    # Parameters for transpose matmul
    transpose_a = False  # Do not transpose first matrix
    transpose_b = True   # Transpose second matrix
    return [transpose_a, transpose_b]

