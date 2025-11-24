import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, alpha=1.0):
        super(Model, self).__init__()
        self.alpha = alpha

    def forward(self, input_tensor, mat1, mat2):
        # int8 x int8 -> int32 (accumulate in int32)
        mm_int32 = torch.matmul(mat1.to(torch.int32), mat2.to(torch.int32))
        # int32 + int32 -> int32
        sum_int32 = mm_int32 + input_tensor
        # int32 * fp32 -> fp32
        return sum_int32.to(torch.float32) * self.alpha


def get_inputs():
    input_tensor = torch.randint(-2**15, 2**15, (1024, 4096), dtype=torch.int32)
    mat1 = torch.randint(-128, 128, (1024, 1344), dtype=torch.int8)
    mat2 = torch.randint(-128, 128, (1344, 4096), dtype=torch.int8)
    return [input_tensor, mat1, mat2]


def get_init_inputs():
    alpha = 1.0
    return [alpha]