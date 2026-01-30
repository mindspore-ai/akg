import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Rotary Positional Embedding (RoPE) operation.
    This operation is commonly used in neural networks for:
    - Adding positional information to transformer models
    - Used in LLaMA, PaLM, and other large language models
    - Provides rotation-based positional encoding that generalizes well to different sequence lengths
    
    Formula: Applies rotary embedding to query and key tensors using cosine and sine functions
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, state, cos, sin):
        _, _, dim = state.shape
        state_x = state[:, :, 0:dim:2]
        state_y = state[:, :, 1:dim:2]
        out_x = state_x * cos - state_y * sin
        out_y = state_x * sin + state_y * cos
        out = torch.empty_like(state)
        out[:, :, 0:dim:2] = out_x
        out[:, :, 1:dim:2] = out_y
        return out

def get_inputs_dyn_list():
    # RoPE variation cases with both aligned and non-aligned shapes
    
    # Case 1: 16-aligned batch
    # Shape (256, 64, 128) represents a batch of 256 samples with 64 sequence length and 128 hidden size
    state1 = torch.randn(256, 64, 128, dtype=torch.float32)
    cos1 = torch.randn((256, 1, 128 // 2), dtype=torch.float32)
    sin1 = torch.randn((256, 1, 128 // 2), dtype=torch.float32)
    
    # Case 2: Non-16-aligned batch
    # Shape (125, 64, 256) represents a batch of 125 samples with 64 sequence length and 256 hidden size
    state2 = torch.randn(125, 64, 256, dtype=torch.float32)
    cos2 = torch.randn((125, 1, 256 // 2), dtype=torch.float32)
    sin2 = torch.randn((125, 1, 256 // 2), dtype=torch.float32)

    # Case 3: Large batch size
    # Shape (1024, 64, 512) represents a batch of 1024 samples with 64 sequence length and 512 hidden size
    state3 = torch.randn(1024, 64, 512, dtype=torch.float32)
    cos3 = torch.randn((1024, 1, 512 // 2), dtype=torch.float32)
    sin3 = torch.randn((1024, 1, 512 // 2), dtype=torch.float32)

    return [
        [state1, cos1, sin1],
        [state2, cos2, sin2],
        [state3, cos3, sin3]
    ]

def get_init_inputs():
    return []