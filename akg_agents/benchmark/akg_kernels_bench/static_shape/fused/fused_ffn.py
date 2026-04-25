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

def get_inputs():
    # Batch size: 64
    # Hidden dimension: 4096
    # Intermediate dimension: 8192
    x = torch.randn(64, 4096, dtype=torch.float16)
    weight1 = torch.randn(4096, 8192, dtype=torch.float16)
    weight2 = torch.randn(8192, 4096, dtype=torch.float16)
    bias1 = torch.randn(8192, dtype=torch.float16)
    bias2 = torch.randn(4096, dtype=torch.float16)
    return [x, weight1, weight2, bias1, bias2]

def get_init_inputs():
    # Parameters for FFN (configuration only)
    activation = "relu"      # Activation function
    inner_precise = 1        # Precision flag
    tokens_index_flag = True # Token indexing flag
    return [activation, inner_precise, tokens_index_flag]