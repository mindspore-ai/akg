import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Simple model that performs ScatterElementsV2 operation.
    """

    def __init__(self, dim=0):
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, self_tensor, index, src):
        """
        Perform ScatterElementsV2 operation.

        Args:
            self_tensor: Base tensor to scatter into
            index: Index tensor specifying positions
            src: Source tensor with values to scatter

        Returns:
            Scattered tensor
        """
        # Create a copy of self_tensor to avoid modifying the original
        output = self_tensor.clone()

        # Use scatter_ to perform in-place scatter operation along specified dimension
        output.scatter_(self.dim, index, src)

        return output


def get_inputs():
    """
    Generate random input tensors for testing.
    """
    # Use shapes from gen_data.py: self_shape = [3, 4], index_shape = [2, 3], src_shape = [2, 3]
    self_shape = [3, 4]
    index_shape = [2, 3]
    src_shape = [2, 3]

    self_tensor = torch.randn(self_shape, dtype=torch.float32)
    # indices within valid range
    index = torch.randint(0, 3, index_shape, dtype=torch.int64)
    src = torch.randn(src_shape, dtype=torch.float32)

    return [self_tensor, index, src]


def get_init_inputs():
    """
    Return initialization parameters for the model.
    """
    return [0]  # dim=0
