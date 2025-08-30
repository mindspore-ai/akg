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


def get_inputs_dyn_list():
    # Small shape case
    input1 = torch.randn(128, 512, 1024, dtype=torch.float16)  # batch=128 > 100
    seq_lengths1 = torch.randint(256, 512, (128,), dtype=torch.int32)

    # Middle shape case
    input2 = torch.randn(512, 2048, 4096, dtype=torch.float16)  # batch=512
    seq_lengths2 = torch.randint(1024, 2048, (512,), dtype=torch.int32)

    # Large shape case
    input3 = torch.randn(1024, 4096, 8192, dtype=torch.float16)
    seq_lengths3 = torch.randint(2048, 4096, (1024,), dtype=torch.int32)

    # Noaligned shape case
    input4 = torch.randn(513, 3000, 6144, dtype=torch.float16)
    seq_lengths4 = torch.randint(1500, 3000, (513,), dtype=torch.int32)

    return [
        [input1, seq_lengths1],
        [input2, seq_lengths2],
        [input3, seq_lengths3],
        [input4, seq_lengths4]
    ]

def get_init_inputs():
    # No parameters for Unpad operation
    return []