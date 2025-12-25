import torch
import torch.nn as nn
import triton
import triton.language as tl

# ============================================================================
# SGLang参考信息
# ============================================================================
# 源文件：sglang/python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py
# SGLang函数：_compute_pid
# 实现类型：辅助函数
# 功能：计算矩阵乘法中的进程ID (pid_m, pid_n)
# 测试文件：test/nightly/test_batch_invariant_ops.py
# 输入参考：根据源文件中的函数签名和test_batch_invariant_ops.py中的测试用例推断
# ============================================================================

# ============================================================================
# 以下是从SGLang直接复制的函数实现
# ============================================================================

def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


# ============================================================================
# AIKGBench标准接口
# ============================================================================
class Model(nn.Module):
    """直接使用复制的原始函数实现"""
    def __init__(self):
        super().__init__()
        # 不需要额外的初始化参数
    
    def forward(self, tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS):
        return _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS)

class ModelSGLang(nn.Module):
    """PyTorch实现"""

    def __init__(self):
        super().__init__()
        # 不需要额外的初始化参数
    
    def forward(self, tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS):
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (tile_id % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m
        return pid_m, pid_n

def get_inputs():
    """生成测试输入"""
    tile_id = 10
    num_pid_in_group = 16
    num_pid_m = 8
    GROUP_SIZE_M = 4
    NUM_SMS = 8
    
    # 将输入转换为张量以保持与其他模型的一致性
    tile_id = torch.tensor(tile_id, dtype=torch.int32)
    num_pid_in_group = torch.tensor(num_pid_in_group, dtype=torch.int32)
    num_pid_m = torch.tensor(num_pid_m, dtype=torch.int32)
    GROUP_SIZE_M = torch.tensor(GROUP_SIZE_M, dtype=torch.int32)
    NUM_SMS = torch.tensor(NUM_SMS, dtype=torch.int32)
    
    return [tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS]

def get_init_inputs():
    """生成初始化参数"""
    return []