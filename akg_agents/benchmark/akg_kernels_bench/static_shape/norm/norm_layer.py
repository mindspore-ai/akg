import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Model that performs Layer Normalization operation.
    Layer normalization normalizes across the feature dimension for each sample.
    """
    def __init__(self, normalized_shape):
        super(Model, self).__init__()
        self.normalized_shape = normalized_shape

    def forward(self, input_tensor):
        """
        Perform Layer Normalization operation.
        
        Args:
            input_tensor: Input tensor of shape (batch_size, seq_len, hidden_size)
            
        Returns:
            Layer normalized output tensor
        """
        # Create weight and bias tensors with the correct shape
        weight = torch.ones(self.normalized_shape, dtype=input_tensor.dtype, device=input_tensor.device)
        bias = torch.zeros(self.normalized_shape, dtype=input_tensor.dtype, device=input_tensor.device)
        
        return torch.layer_norm(input_tensor, self.normalized_shape, weight, bias)


def get_inputs():
    """
    Generate random input tensors for testing with large model shapes.
    """
    # Batch size: 32, Sequence length: 1024, Hidden size: 4096
    batch_size, seq_len, hidden_size = 32, 1024, 4096
    input_tensor = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float32)
    return [input_tensor]


def get_init_inputs():
    # Parameters for layer_norm
    normalized_shape = (4096,)
    return [normalized_shape]