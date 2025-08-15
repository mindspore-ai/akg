import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, tensor1, tensor2):
        # DeepNorm operation (simplified)
        added = tensor1 + tensor2
        result = torch.nn.functional.layer_norm(added, added.shape[1:])
        return result


def get_inputs():
    # Batch size: 32
    # Hidden dimension: 4096
    tensor1 = torch.randn(32, 4096, dtype=torch.float32)
    tensor2 = torch.randn(32, 4096, dtype=torch.float32)
    return [tensor1, tensor2]


def get_init_inputs():
    # No parameters required
    return []