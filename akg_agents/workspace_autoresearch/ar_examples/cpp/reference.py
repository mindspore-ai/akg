import torch


class Model(torch.nn.Module):
    def forward(self, x, y):
        return x + y


def get_inputs():
    x = torch.randn(4096, dtype=torch.float32)
    y = torch.randn(4096, dtype=torch.float32)
    return [x, y]


def get_init_inputs():
    return []
