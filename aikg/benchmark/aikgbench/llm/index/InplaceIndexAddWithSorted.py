import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Simple model that performs InplaceIndexAddWithSorted operation.
    """

    def __init__(self, axis=0):
        super(Model, self).__init__()
        self.axis = axis

    def forward(self, var, value, sorted_indices, pos, alpha=1.0):
        """
        Perform InplaceIndexAddWithSorted operation.

        Args:
            var: Base tensor to add into
            value: Source tensor with values to add
            sorted_indices: Sorted index tensor
            pos: Position tensor
            alpha: Scaling factor

        Returns:
            Tensor with values added at specified indices
        """
        # Create a copy of var to avoid modifying the original
        output = var.clone()

        # Use index_add_ to perform in-place index add operation
        # This adds values at the specified indices along the given axis
        output.index_add_(self.axis, sorted_indices, value, alpha=alpha)

        return output


def get_inputs():
    """
    Generate random input tensors for testing.
    """
    # Use shapes from gen_data.py: self = np.random.randn(4, 2), index = torch.randint(0, 4, (4,))
    var_shape = [4, 2]
    value_shape = [4, 2]
    sorted_indices_shape = [4]
    pos_shape = [4]

    var = torch.randn(var_shape, dtype=torch.float32)
    value = torch.randn(value_shape, dtype=torch.float32)
    sorted_indices = torch.randint(
        0, 4, sorted_indices_shape, dtype=torch.int32)
    pos = torch.randint(0, 4, pos_shape, dtype=torch.int32)

    return [var, value, sorted_indices, pos]


def get_init_inputs():
    """
    Return initialization parameters for the model.
    """
    return [0]  # axis=0
