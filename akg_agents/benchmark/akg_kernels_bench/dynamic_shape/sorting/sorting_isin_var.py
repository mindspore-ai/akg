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


def get_inputs_dyn_list():
    # Case 1: Small batch, small hidden (non-aligned batch)
    elements1 = torch.randint(0, 1000, (15, 1344), dtype=torch.int32)
    test_elements1 = torch.randint(0, 1000, (15, 1344), dtype=torch.int32)
    
    # Case 2: Small batch, large hidden (aligned batch)
    elements2 = torch.randint(0, 1000, (16, 4096), dtype=torch.int32)
    test_elements2 = torch.randint(0, 1000, (16, 4096), dtype=torch.int32)
    
    # Case 3: Medium batch, medium hidden (non-aligned batch)
    elements3 = torch.randint(0, 1000, (127, 2688), dtype=torch.int32)
    test_elements3 = torch.randint(0, 1000, (127, 2688), dtype=torch.int32)
    
    # Case 4: Large batch, large hidden (aligned batch)
    elements4 = torch.randint(0, 1000, (512, 5120), dtype=torch.int32)
    test_elements4 = torch.randint(0, 1000, (512, 5120), dtype=torch.int32)
    
    # Case 5: Very large batch, very large hidden (non-aligned batch)
    elements5 = torch.randint(0, 1000, (1023, 8192), dtype=torch.int32)
    test_elements5 = torch.randint(0, 1000, (1023, 8192), dtype=torch.int32)
    
    return [
        [elements1, test_elements1],
        [elements2, test_elements2],
        [elements3, test_elements3],
        [elements4, test_elements4],
        [elements5, test_elements5]
    ]


def get_init_inputs():
    # No parameters needed for isin
    return []