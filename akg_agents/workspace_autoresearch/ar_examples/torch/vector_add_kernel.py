import torch


class ModelNew(torch.nn.Module):
    def forward(self, x, y):
        return x + y
