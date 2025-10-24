import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Split operation on 4D tensor (batch, channels, height, width).
    This operation is commonly used in neural networks for:
    - Processing image data with multiple channels
    - Implementing spatial attention mechanisms
    - Used in computer vision architectures for feature map processing
    """
    def __init__(self, split_size=None, dim=None):
        super(Model, self).__init__()
        self.split_size = split_size
        self.dim = dim

    def forward(self, input_tensor):
        # torch.split(tensor, split_size_or_sections, dim=1)
        # Splits the tensor along channel dimension for multi-channel processing
        return torch.split(input_tensor, self.split_size, dim=self.dim)


def get_inputs():
    # Splitting along channel dimension with split_size 16 gives us 4 chunks of (8, 15, 224, 224) and one chunk of (8, 4, 224, 224)
    input_tensor = torch.randn(8, 64, 224, 224, dtype=torch.float32)
    return [input_tensor]


def get_init_inputs():
    # Parameters for split
    split_size = 15  # Size of each chunk along channel dimension
    dim = 1          # Dimension along which to split (channel dimension)
    return [split_size, dim]
