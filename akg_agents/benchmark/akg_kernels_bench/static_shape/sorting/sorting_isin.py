import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, elements, test_elements):
        # torch.isin(elements, test_elements, *, assume_unique=False, invert=False)
        # Tests if each element of elements is in test_elements.
        # Returns a boolean tensor of the same shape as elements that is True for elements
        # in test_elements and False otherwise.
        # Isin operations are commonly used in neural networks for:
        # - Implementing masking operations
        # - Filtering elements based on a set of values
        # - Validating tensor contents
        return torch.isin(elements, test_elements)


def get_inputs():
    # Batch size: 1024
    # Hidden dimension: 4096
    elements = torch.randint(0, 1000, (1024, 4096), dtype=torch.int32)
    test_elements = torch.randint(0, 1000, (1024, 4096), dtype=torch.int32)
    return [elements, test_elements]


def get_init_inputs():
    # No parameters needed for isin
    return []