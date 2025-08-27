import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Model that performs Feeds Repeat operation.
    This operation repeats input tensors along specified dimensions, which is commonly used
    in attention mechanisms, data augmentation, and other scenarios where tensor expansion is needed.
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, repeats):
        """
        Perform Feeds Repeat operation.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size) with float32 dtype
            repeats: Repeat counts for each dimension of shape (3,) with int64 dtype

        Returns:
            Repeated tensor with expanded dimensions
        """
        # Use torch.repeat to repeat the tensor along specified dimensions
        output = x.repeat(*repeats)

        return output


def get_inputs():
    """
    Generate random input tensors for testing with large model shapes.
    """
    # Batch size: 32
    # Sequence length: 1024
    # Hidden size: 2048
    batch_size, seq_len, hidden_size = 32, 1024, 2048

    # Generate input tensor
    x = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float32)
    
    # Generate repeat counts for each dimension
    repeats = torch.tensor([1, 2, 1], dtype=torch.int64)  # Repeat sequence length by 2

    return [x, repeats]


def get_init_inputs():
    """
    Return initialization parameters for the model.
    """
    return []
