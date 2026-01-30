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


def get_inputs_dyn_list():
    """
    Generate dynamic input tensors for testing with various batch and sequence lengths.
    """
    # Case 1: Small batch, small seq, small hidden (non-aligned batch)
    batch_size1, seq_len1, hidden_size1 = 15, 63, 1344
    x1 = torch.randn(batch_size1, seq_len1, hidden_size1, dtype=torch.float32)
    repeats1 = torch.tensor([1, 2, 1], dtype=torch.int64)
    
    # Case 2: Small batch, medium seq, large hidden (aligned batch)
    batch_size2, seq_len2, hidden_size2 = 16, 512, 2688
    x2 = torch.randn(batch_size2, seq_len2, hidden_size2, dtype=torch.float32)
    repeats2 = torch.tensor([1, 2, 1], dtype=torch.int64)
    
    # Case 3: Medium batch, large seq, large hidden (non-aligned batch)
    batch_size3, seq_len3, hidden_size3 = 63, 2047, 4096
    x3 = torch.randn(batch_size3, seq_len3, hidden_size3, dtype=torch.float32)
    repeats3 = torch.tensor([1, 2, 1], dtype=torch.int64)
    
    # Case 4: Large batch, large seq, large hidden (aligned batch)
    batch_size4, seq_len4, hidden_size4 = 64, 2048, 5120
    x4 = torch.randn(batch_size4, seq_len4, hidden_size4, dtype=torch.float32)
    repeats4 = torch.tensor([1, 2, 1], dtype=torch.int64)
    
    # Case 5: Very large batch, very large seq, very large hidden (non-aligned batch)
    batch_size5, seq_len5, hidden_size5 = 127, 4095, 8192
    x5 = torch.randn(batch_size5, seq_len5, hidden_size5, dtype=torch.float32)
    repeats5 = torch.tensor([1, 2, 1], dtype=torch.int64)

    return [
        [x1, repeats1],
        [x2, repeats2],
        [x3, repeats3],
        [x4, repeats4],
        [x5, repeats5]
    ]


def get_init_inputs():
    """
    Return initialization parameters for the model.
    """
    return []