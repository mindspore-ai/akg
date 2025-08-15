import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Simple model that performs MaskedSelectV3 operation.
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, mask):
        """
        Perform MaskedSelectV3 operation.

        Args:
            x: Input tensor
            mask: Boolean mask tensor

        Returns:
            Selected elements as 1D tensor
        """
        # Use torch.masked_select to select elements based on boolean mask
        result = torch.masked_select(x, mask)
        return result


def get_inputs():
    """
    Generate random input tensors for testing.
    """
    # Use shapes from gen_data.py: input_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
    x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32)
    mask = torch.tensor(
        [[True, False, True], [False, True, False]], dtype=torch.bool)

    return [x, mask]


def get_init_inputs():
    """
    Return initialization parameters for the model.
    """
    return []
