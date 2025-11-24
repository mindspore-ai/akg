import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Model that performs Masked Select V3 operation.
    Masked Select operation selects elements from a tensor based on a boolean mask.
    This operation is commonly used in attention mechanisms, sparse operations, and
    other scenarios where conditional element selection is needed.
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input_tensor, mask):
        """
        Perform Masked Select operation.

        Args:
            input_tensor: Input tensor of shape (batch_size, seq_len, hidden_size) with float32 dtype
            mask: Boolean mask tensor of shape (batch_size, seq_len, hidden_size) with bool dtype

        Returns:
            Selected elements as a 1D tensor
        """
        # Use torch.masked_select to select elements based on the mask
        output = torch.masked_select(input_tensor, mask)

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
    input_tensor = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float32)
    
    # Generate boolean mask (approximately 50% True values)
    mask = torch.rand(batch_size, seq_len, hidden_size) > 0.5

    return [input_tensor, mask]


def get_init_inputs():
    """
    Return initialization parameters for the model.
    """
    return []
