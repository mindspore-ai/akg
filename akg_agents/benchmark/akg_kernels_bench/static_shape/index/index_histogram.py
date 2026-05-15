import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, indices, num_experts):
        splits = torch.zeros(num_experts, dtype=torch.int32, device=indices.device)
        indices = indices.flatten()
        splits.scatter_add_(0, indices, torch.ones_like(indices, dtype=torch.int32))
        return splits

topk = 8
token_num = 65536

def get_inputs():
    num_experts = 365
    indices = torch.randint(low=0, high=num_experts - 1, size=[token_num, topk], dtype=torch.int32)
    return [indices, num_experts]

def get_init_inputs():
    return []