import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Simple linear layer model for testing weight initialization alignment.
    Shape: 16x16x16 (batch_size=16, in_features=16, out_features=16)
    """
    def __init__(self, in_features, out_features):
        super(Model, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


batch_size = 16
in_features = 16
out_features = 16


def get_inputs():
    x = torch.randn(batch_size, in_features)
    return [x]


def get_init_inputs():
    return [in_features, out_features]

