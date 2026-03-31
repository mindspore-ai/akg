import torch

@torch.no_grad()
def run(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.relu(x)
