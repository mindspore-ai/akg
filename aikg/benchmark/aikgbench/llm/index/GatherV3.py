import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Simple model that performs GatherV3 operation.
    """

    def __init__(self, axis=0):
        super(Model, self).__init__()
        self.axis = axis

    def forward(self, self_tensor, indices, axis_tensor=None):
        """
        Perform GatherV3 operation.

        Args:
            self_tensor: Input tensor
            indices: Index tensor
            axis_tensor: Axis tensor (optional, uses self.axis if not provided)

        Returns:
            Gathered tensor
        """
        # Use the provided axis_tensor or fall back to self.axis
        axis = axis_tensor.item() if axis_tensor is not None else self.axis

        # Use torch.gather to gather values along the specified axis
        result = torch.gather(self_tensor, axis, indices)
        return result


def get_inputs():
    """
    Generate random input tensors for testing.
    """
    # Use shapes from gen_data.py: input_data = torch.randn(4, 2), index = torch.randint(0, 4, (2,))
    self_shape = [4, 2]
    indices_shape = [2]

    self_tensor = torch.randn(self_shape, dtype=torch.float32)
    indices = torch.randint(0, 4, indices_shape, dtype=torch.int64)  # 改为 int64
    axis_tensor = torch.tensor([0], dtype=torch.int64)

    return [self_tensor, indices, axis_tensor]


def get_init_inputs():
    """
    Return initialization parameters for the model.
    """
    return [0]  # axis=0
