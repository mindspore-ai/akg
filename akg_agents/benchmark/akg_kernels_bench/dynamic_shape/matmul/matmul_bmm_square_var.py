import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input, weight):
        # BatchMatmul operation
        # This operation is commonly used in neural networks for:
        # - Performing matrix multiplication on batches of matrices
        # - Used in multi-head attention mechanisms
        # - Processing multiple samples in parallel
        
        # Perform batch matrix multiplication
        result = torch.bmm(input, weight)
        
        return result

def get_inputs_dyn_list():
    # BatchMatmul variation cases with different batch sizes (max 128) and matrix dimensions
    
    # Case 1: Small batch size 8, matrix size 32x32
    input1 = torch.randn(8, 32, 32, dtype=torch.float16)
    weight1 = torch.randn(8, 32, 32, dtype=torch.float16)
    
    # Case 2: Small batch size 16, matrix size 64x64
    input2 = torch.randn(16, 64, 64, dtype=torch.float16)
    weight2 = torch.randn(16, 64, 64, dtype=torch.float16)
    
    # Case 3: Medium batch size 24, matrix size 96x96
    input3 = torch.randn(24, 96, 96, dtype=torch.float16)
    weight3 = torch.randn(24, 96, 96, dtype=torch.float16)
    
    # Case 4: Medium batch size 32, matrix size 128x128
    input4 = torch.randn(32, 128, 128, dtype=torch.float16)
    weight4 = torch.randn(32, 128, 128, dtype=torch.float16)
    
    # Case 5: Medium batch size 48, matrix size 160x160
    input5 = torch.randn(48, 160, 160, dtype=torch.float16)
    weight5 = torch.randn(48, 160, 160, dtype=torch.float16)
    
    # Case 6: Large batch size 64, matrix size 192x192
    input6 = torch.randn(64, 192, 192, dtype=torch.float16)
    weight6 = torch.randn(64, 192, 192, dtype=torch.float16)
    
    # Case 7: Large batch size 80, matrix size 224x224
    input7 = torch.randn(80, 224, 224, dtype=torch.float16)
    weight7 = torch.randn(80, 224, 224, dtype=torch.float16)
    
    # Case 8: Large batch size 96, matrix size 256x256
    input8 = torch.randn(96, 256, 256, dtype=torch.float16)
    weight8 = torch.randn(96, 256, 256, dtype=torch.float16)
    
    # Case 9: Large batch size 112, matrix size 288x288
    input9 = torch.randn(112, 288, 288, dtype=torch.float16)
    weight9 = torch.randn(112, 288, 288, dtype=torch.float16)
    
    # Case 10: Maximum batch size 128, matrix size 320x320
    input10 = torch.randn(128, 320, 320, dtype=torch.float16)
    weight10 = torch.randn(128, 320, 320, dtype=torch.float16)
    
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
    # No parameters needed for BatchMatmul operation
    return []