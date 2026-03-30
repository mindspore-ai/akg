import torch

def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict:
    batch_size = axes_and_scalars["batch_size"]
    seq_len = axes_and_scalars["seq_len"]
    hidden_size = axes_and_scalars["hidden_size"]
    
    x = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float32, device=device)
    
    return {
        "x": x
    }

@torch.no_grad()
def run(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.relu(x)
