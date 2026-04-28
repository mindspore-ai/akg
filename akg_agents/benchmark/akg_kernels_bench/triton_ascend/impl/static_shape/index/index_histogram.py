import torch
import triton
import triton.language as tl


@triton.jit
def aikg_histogram_kernel(
    sorted_indices_ptr,
    splits_ptr,
    total_elements,
):
    expert_idx = tl.program_id(0)
    expert_id = expert_idx.to(tl.float32)
    
    # 二分查找下界（第一个等于expert_id的位置）
    left = 0
    right = total_elements - 1
    start_pos = total_elements
    
    while left <= right:
        mid = (left + right) // 2
        mid_val = tl.load(sorted_indices_ptr + mid)
        
        if mid_val < expert_id:
            left = mid + 1
        else:
            if mid_val == expert_id:
                start_pos = tl.minimum(start_pos, mid)
            right = mid - 1
    
    # 二分查找上界（最后一个等于expert_id的位置）
    left = 0
    right = total_elements - 1
    end_pos = -1
    
    while left <= right:
        mid = (left + right) // 2
        mid_val = tl.load(sorted_indices_ptr + mid)
        
        if mid_val <= expert_id:
            if mid_val == expert_id:
                end_pos = tl.maximum(end_pos, mid)
            left = mid + 1
        else:
            right = mid - 1
    
    count = 0
    if start_pos <= end_pos:
        count = end_pos - start_pos + 1
    
    # 原子操作累加结果
    if count > 0:
        tl.atomic_add(splits_ptr + expert_idx, count)


def aikg_histogram_triton_torch(indices, num_experts):
    # 将输入展平并排序
    indices_flat = indices.flatten().to(torch.float32)
    sorted_indices, _ = torch.sort(indices_flat)
    total_elements = sorted_indices.numel()
    
    splits = torch.zeros(num_experts, dtype=torch.int32, device=indices.device)
    grid = (num_experts,)
    aikg_histogram_kernel[grid](
        sorted_indices,
        splits,
        total_elements,
    )
    
    return splits