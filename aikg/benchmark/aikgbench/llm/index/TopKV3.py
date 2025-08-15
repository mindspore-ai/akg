import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Simple model that performs TopKV3 operation.
    """

    def __init__(self, k=2, dim=1, largest=True, sorted=True):
        super(Model, self).__init__()
        self.k = k
        self.dim = dim
        self.largest = largest
        self.sorted = sorted

    def forward(self, self_tensor):
        """
        Perform TopKV3 operation.

        Args:
            self_tensor: Input tensor

        Returns:
            Tuple of (values, indices) tensors
        """
        values, indices = torch.topk(self_tensor, k=self.k, dim=self.dim,
                                     largest=self.largest, sorted=self.sorted)
        return values, indices


def get_inputs():
    """
    Generate random input tensors for testing.
    """
    # Use shapes from gen_data.py: input_shape = [2, 16]
    input_shape = [2, 16]
    self_tensor = torch.randn(input_shape, dtype=torch.float32)
    return [self_tensor]


def get_init_inputs():
    """
    Return initialization parameters for the model.
    """
    return [2, 1, True, True]  # k=2, dim=1, largest=True, sorted=True
