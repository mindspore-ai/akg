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


def get_inputs_dyn_list():
    """
    Generate dynamic input tensors for testing with various batch and sequence lengths.
    """
    # Case 1: Small batch, small seq, small hidden (non-aligned batch)
    batch_size1, seq_len1, hidden_size1 = 15, 63, 1344
    input_tensor1 = torch.randn(batch_size1, seq_len1, hidden_size1, dtype=torch.float32)
    mask1 = torch.rand(batch_size1, seq_len1, hidden_size1) > 0.5
    
    # Case 2: Small batch, medium seq, large hidden (aligned batch)
    batch_size2, seq_len2, hidden_size2 = 16, 512, 2688
    input_tensor2 = torch.randn(batch_size2, seq_len2, hidden_size2, dtype=torch.float32)
    mask2 = torch.rand(batch_size2, seq_len2, hidden_size2) > 0.5
    
    # Case 3: Medium batch, large seq, large hidden (non-aligned batch)
    batch_size3, seq_len3, hidden_size3 = 63, 2047, 4096
    input_tensor3 = torch.randn(batch_size3, seq_len3, hidden_size3, dtype=torch.float32)
    mask3 = torch.rand(batch_size3, seq_len3, hidden_size3) > 0.5
    
    # Case 4: Large batch, large seq, large hidden (aligned batch)
    batch_size4, seq_len4, hidden_size4 = 64, 2048, 5120
    input_tensor4 = torch.randn(batch_size4, seq_len4, hidden_size4, dtype=torch.float32)
    mask4 = torch.rand(batch_size4, seq_len4, hidden_size4) > 0.5
    
    # Case 5: Very large batch, very large seq, very large hidden (non-aligned batch)
    batch_size5, seq_len5, hidden_size5 = 127, 4095, 8192
    input_tensor5 = torch.randn(batch_size5, seq_len5, hidden_size5, dtype=torch.float32)
    mask5 = torch.rand(batch_size5, seq_len5, hidden_size5) > 0.5

    return [
        [input_tensor1, mask1],
        [input_tensor2, mask2],
        [input_tensor3, mask3],
        [input_tensor4, mask4],
        [input_tensor5, mask5]
    ]


def get_init_inputs():
    """
    Return initialization parameters for the model.
    """
    return []