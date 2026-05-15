import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, activation="relu", inner_precise=1, tokens_index_flag=True):
        super(Model, self).__init__()
        self.activation = activation
        self.inner_precise = inner_precise
        self.tokens_index_flag = tokens_index_flag
        
    def forward(self, x, weight1, weight2, bias1, bias2):
        # Feed-Forward Network (FFN) with expert routing
        # This operation is commonly used in neural networks for:
        # - Implementing position-wise feed-forward networks in transformer models
        # - Adding non-linearity and capacity to transformer blocks
        # - Processing tokens independently with learned transformations
        
        # First linear transformation with bias
        hidden = torch.matmul(x, weight1) + bias1
        
        # Apply activation function
        if self.activation == "relu":
            hidden = torch.relu(hidden)
        elif self.activation == "gelu":
            hidden = torch.nn.functional.gelu(hidden)
        
        # Second linear transformation with bias
        output = torch.matmul(hidden, weight2) + bias2
        
        return output

def get_inputs_dyn_list():
    # Case 1: Small batch, small hidden (non-aligned batch)
    x1 = torch.randn(15, 1344, dtype=torch.float16)
    weight1_1 = torch.randn(1344, 2688, dtype=torch.float16)
    weight2_1 = torch.randn(2688, 1344, dtype=torch.float16)
    bias1_1 = torch.randn(2688, dtype=torch.float16)
    bias2_1 = torch.randn(1344, dtype=torch.float16)
    
    # Case 2: Small batch, large hidden (aligned batch)
    x2 = torch.randn(16, 4096, dtype=torch.float16)
    weight1_2 = torch.randn(4096, 8192, dtype=torch.float16)
    weight2_2 = torch.randn(8192, 4096, dtype=torch.float16)
    bias1_2 = torch.randn(8192, dtype=torch.float16)
    bias2_2 = torch.randn(4096, dtype=torch.float16)
    
    # Case 3: Medium batch, medium hidden (non-aligned batch)
    x3 = torch.randn(127, 2688, dtype=torch.float16)
    weight1_3 = torch.randn(2688, 5376, dtype=torch.float16)
    weight2_3 = torch.randn(5376, 2688, dtype=torch.float16)
    bias1_3 = torch.randn(5376, dtype=torch.float16)
    bias2_3 = torch.randn(2688, dtype=torch.float16)
    
    # Case 4: Large batch, large hidden (aligned batch)
    x4 = torch.randn(128, 5120, dtype=torch.float16)
    weight1_4 = torch.randn(5120, 10240, dtype=torch.float16)
    weight2_4 = torch.randn(10240, 5120, dtype=torch.float16)
    bias1_4 = torch.randn(10240, dtype=torch.float16)
    bias2_4 = torch.randn(5120, dtype=torch.float16)
    
    # Case 5: Very large batch, very large hidden (non-aligned batch)
    x5 = torch.randn(255, 8192, dtype=torch.float16)
    weight1_5 = torch.randn(8192, 16384, dtype=torch.float16)
    weight2_5 = torch.randn(16384, 8192, dtype=torch.float16)
    bias1_5 = torch.randn(16384, dtype=torch.float16)
    bias2_5 = torch.randn(8192, dtype=torch.float16)
    
    return [
        [x1, weight1_1, weight2_1, bias1_1, bias2_1],
        [x2, weight1_2, weight2_2, bias1_2, bias2_2],
        [x3, weight1_3, weight2_3, bias1_3, bias2_3],
        [x4, weight1_4, weight2_4, bias1_4, bias2_4],
        [x5, weight1_5, weight2_5, bias1_5, bias2_5]
    ]

def get_init_inputs():
    # Parameters for FFN (configuration only)
    activation = "relu"      # Activation function
    inner_precise = 1        # Precision flag
    tokens_index_flag = True # Token indexing flag
    return [activation, inner_precise, tokens_index_flag]