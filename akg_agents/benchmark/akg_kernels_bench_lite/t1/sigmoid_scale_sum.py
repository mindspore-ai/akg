import torch
import torch_npu

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, bias):
        t1 = x * 2.0
        t2 = t1 + bias
        t3 = torch.sigmoid(t2)
        t4 = torch.sum(t3, dim=-1, keepdim=True)
        
        return t4

def get_init_inputs():
    return []

def get_inputs():
    x = torch.randn(1000, 8192)
    bias = torch.randn(8192)
    return [x, bias]