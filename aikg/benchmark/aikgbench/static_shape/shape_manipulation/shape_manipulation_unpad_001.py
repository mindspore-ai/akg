import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Unpad operation that removes padding from sequences.
    This operation is commonly used in neural networks for:
    - Removing padding tokens from variable-length sequences
    - Used in transformer models for efficient processing of variable-length sequences
    - Improves computational efficiency by avoiding processing of padding tokens
    
    Formula: Extract elements from input tensor based on sequence lengths
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input_tensor, seq_lengths):
        # Unpad operation using input_tensor as input tensor and seq_lengths as sequence lengths
        # Create a mask based on sequence lengths
        batch_size, max_seq_len = input_tensor.shape[0], input_tensor.shape[1]
        mask = torch.arange(max_seq_len, device=input_tensor.device).expand(batch_size, max_seq_len) < seq_lengths.unsqueeze(1)
        
        # Apply mask to input
        result = input_tensor * mask.unsqueeze(-1).float()
        return result

def get_inputs():
    # Batch size: 32
    # Max sequence length: 1024
    # Hidden size: 4096
    # seq_lengths shape: [batch_size] = [32]
    input_tensor = torch.randn(32, 1024, 4096, dtype=torch.float16)
    seq_lengths = torch.randint(512, 1024, (32,), dtype=torch.int32)  # Sequence lengths
    return [input_tensor, seq_lengths]

def get_init_inputs():
    # No parameters for Unpad operation
    return []
