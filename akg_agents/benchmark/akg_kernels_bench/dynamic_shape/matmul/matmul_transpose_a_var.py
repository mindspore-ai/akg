import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, transpose_a=True, transpose_b=False):
        super(Model, self).__init__()
        self.transpose_a = transpose_a
        self.transpose_b = transpose_b

    def forward(self, input, weight):
        # Transpose only input matrix
        mat1 = input.t()
        mat2 = weight
        
        # torch.matmul(input, other, *, out=None)
        # Performs matrix multiplication of the matrices input and other.
        return torch.matmul(mat1, mat2)

def get_inputs_dyn_list():
    # Matrix multiplication with transpose A only using realistic shapes for large models
    
    # Case 1: Small batch, small dimensions
    input1 = torch.randn(256, 4, dtype=torch.float16)  # Will be transposed to 4x256
    weight1 = torch.randn(256, 512, dtype=torch.float16)
    
    # Case 2: Small batch, medium dimensions
    input2 = torch.randn(512, 8, dtype=torch.float16)  # Will be transposed to 8x512
    weight2 = torch.randn(512, 1024, dtype=torch.float16)
    
    # Case 3: Medium batch, medium dimensions
    input3 = torch.randn(1024, 16, dtype=torch.float16)  # Will be transposed to 16x1024
    weight3 = torch.randn(1024, 2048, dtype=torch.float16)
    
    # Case 4: Medium batch, large dimensions
    input4 = torch.randn(2048, 32, dtype=torch.float16)  # Will be transposed to 32x2048
    weight4 = torch.randn(2048, 4096, dtype=torch.float16)
    
    # Case 5: Large batch, small dimensions
    input5 = torch.randn(256, 64, dtype=torch.float16)  # Will be transposed to 64x256
    weight5 = torch.randn(256, 512, dtype=torch.float16)
    
    # Case 6: Large batch, medium dimensions
    input6 = torch.randn(1024, 128, dtype=torch.float16)  # Will be transposed to 128x1024
    weight6 = torch.randn(1024, 2048, dtype=torch.float16)
    
    # Case 7: Large batch, large dimensions
    input7 = torch.randn(2048, 192, dtype=torch.float16)  # Will be transposed to 192x2048
    weight7 = torch.randn(2048, 4096, dtype=torch.float16)
    
    # Case 8: Special case: small batch, very large dimensions
    input8 = torch.randn(4096, 2, dtype=torch.float16)  # Will be transposed to 2x4096
    weight8 = torch.randn(4096, 8192, dtype=torch.float16)
    
    # Case 9: Special case: large batch, small dimensions
    input9 = torch.randn(128, 256, dtype=torch.float16)  # Will be transposed to 256x128
    weight9 = torch.randn(128, 256, dtype=torch.float16)
    
    # Case 10: Balanced case
    input10 = torch.randn(1024, 96, dtype=torch.float16)  # Will be transposed to 96x1024
    weight10 = torch.randn(1024, 2048, dtype=torch.float16)
    
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
    transpose_a = True   # Transpose first matrix
    transpose_b = False  # Do not transpose second matrix
    return [transpose_a, transpose_b]

