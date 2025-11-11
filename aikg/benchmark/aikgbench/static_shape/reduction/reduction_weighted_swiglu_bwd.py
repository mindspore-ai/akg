import torch
import torch_npu

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, weight, grad):
        weighted_x = x * weight
        grad_weight = grad * weight
        grad_x = torch.sum(x * grad, dim=-1)
        return weighted_x, grad_weight, grad_x

def get_init_inputs():
    return []

def get_inputs():
    x = torch.randn(16, 1024, 2048)
    weight = torch.randn(16, 1024, 2048)
    grad = torch.randn(16, 1024, 2048)
    return [x, weight, grad]