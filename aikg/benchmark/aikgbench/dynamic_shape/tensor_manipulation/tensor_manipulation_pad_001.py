import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Pad operation that adds padding to sequences.
    This operation is commonly used in neural networks for:
    - Adding padding tokens to variable-length sequences to make them uniform
    - Used in transformer models for processing variable-length sequences
    - Ensures consistent tensor shapes for batch processing
    
    Formula: Add padding elements to input tensor to reach target shape
    """
    def __init__(self, pad_value=0.0):
        super(Model, self).__init__()
        self.pad_value = pad_value

    def forward(self, input_tensor, target_seq_len):
        # Pad operation using input_tensor as input tensor and target_seq_len as pad specification
        # Create padded tensor
        padded_shape = list(input_tensor.shape)
        padded_shape[1] = target_seq_len  # Set sequence length to target
        
        # Create result tensor filled with pad_value
        result = torch.full(padded_shape, self.pad_value, dtype=input_tensor.dtype, device=input_tensor.device)
        
        # Copy original data
        result[:, :input_tensor.shape[1], :] = input_tensor
        return result

def get_inputs_dyn_list():
    # Small shape case
    input1 = torch.randn(256, 128, 64, dtype=torch.float16)
    target_seq_len1 = 256

    # Middle shape case (增大参数)
    input2 = torch.randn(1024, 1024, 256, dtype=torch.float16)
    target_seq_len2 = 2048

    # Large shape case (增大参数)
    input3 = torch.randn(2048, 4096, 512, dtype=torch.float16)
    target_seq_len3 = 8192

    # Noaligned shape case
    input4 = torch.randn(513, 3000, 384, dtype=torch.float16)
    target_seq_len4 = 6000

    return [
        [input1, target_seq_len1],
        [input2, target_seq_len2],
        [input3, target_seq_len3],
        [input4, target_seq_len4]
    ]

def get_init_inputs():
    # Parameters for Pad operation
    pad_value = 0.0  # Value to use for padding
    return [pad_value]