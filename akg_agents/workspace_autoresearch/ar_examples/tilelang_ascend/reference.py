import torch
import torch_npu  # noqa: F401


class Model(torch.nn.Module):
    def forward(self, x, y):
        return x + y


def get_inputs():
    device = "npu"
    x = torch.randn(128, 256, device=device, dtype=torch.float16)
    y = torch.randn(128, 256, device=device, dtype=torch.float16)
    return [x, y]


def get_init_inputs():
    return []
