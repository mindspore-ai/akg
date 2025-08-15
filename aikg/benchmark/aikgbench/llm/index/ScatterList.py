import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Simple model that performs ScatterList operation.
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, self_tensor, index, src):
        """
        Perform ScatterList operation.

        Args:
            self_tensor: Base tensor to scatter into
            index: Index tensor specifying positions
            src: Source tensor with values to scatter

        Returns:
            Scattered tensor
        """
        # Create a copy of self_tensor to avoid modifying the original
        output = self_tensor.clone()

        # Use scatter_ to perform in-place scatter operation
        # This is equivalent to scatter operation where we update values at specified indices
        output.scatter_(0, index, src)

        return output


def get_inputs():
    """
    Generate random input tensors for testing.
    """
    # Use shapes from gen_data.py: varRefShape = [5, 3, 4], indiceShape = [1, 2], updatesShape = [1, 5, 3, 4]
    varRefShape = [5, 3, 4]
    indiceShape = [1, 2]
    updatesShape = [1, 5, 3, 4]

    self_tensor = torch.randn(varRefShape, dtype=torch.float32)
    # indices within valid range
    index = torch.randint(0, 3, indiceShape, dtype=torch.int64)
    src = torch.randn(updatesShape, dtype=torch.float32)

    return [self_tensor, index, src]


def get_init_inputs():
    """
    Return initialization parameters for the model.
    """
    return []
