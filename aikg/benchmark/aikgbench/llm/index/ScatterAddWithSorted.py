import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Simple model that performs ScatterAddWithSorted operation.
    """

    def __init__(self, dim=0):
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, var, value, sorted_index, pos, reduction="add"):
        """
        Perform ScatterAddWithSorted operation.

        Args:
            var: Base tensor to scatter into
            value: Source tensor with values to scatter
            sorted_index: Sorted index tensor
            pos: Position tensor
            reduction: Reduction method ("add" for scatter_add)

        Returns:
            Scattered tensor with accumulated values
        """
        # Create a copy of var to avoid modifying the original
        output = var.clone()

        # Use scatter_add_ to perform in-place scatter add operation
        # This accumulates values at the same indices
        output.scatter_add_(self.dim, sorted_index, value)

        return output


def get_inputs():
    """
    Generate random input tensors for testing.
    """
    # Use shapes from gen_data.py: self = torch.randn(4, 4), index = torch.randint(0, 3, (3, 4))
    var_shape = [4, 4]
    value_shape = [4, 4]
    sorted_index_shape = [3, 4]
    pos_shape = [3, 4]

    var = torch.randn(var_shape, dtype=torch.float32)
    value = torch.randn(value_shape, dtype=torch.float32)
    sorted_index = torch.randint(
        0, 3, sorted_index_shape, dtype=torch.int64)  # 改为 int64
    pos = torch.randint(0, 4, pos_shape, dtype=torch.int32)

    return [var, value, sorted_index, pos]


def get_init_inputs():
    """
    Return initialization parameters for the model.
    """
    return [0]  # dim=0
