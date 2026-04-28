import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
        triton.Config({'BLOCK_SIZE': 2048}),
    ],
    key=['total_elements'],
)
@triton.jit
def aikg_put_kernel(
    input_scores_ptr,
    unit_indices_ptr,
    position_map_ptr,
    group_buffer_ptr,
    sequence_length: tl.constexpr,
    select_count: tl.constexpr,
    total_elements: tl.constexpr,
    input_scores_stride_0: tl.constexpr,
    input_scores_stride_1: tl.constexpr,
    group_buffer_stride_0: tl.constexpr,
    group_buffer_stride_1: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    start_idx = pid * BLOCK_SIZE
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    unit_indices_tile = tl.load(unit_indices_ptr + offsets, mask=mask, other=0)
    position_map_tile = tl.load(position_map_ptr + offsets, mask=mask, other=0)
    
    for i in tl.range(0, BLOCK_SIZE):
        if start_idx + i < total_elements:
            row_idx = (start_idx + i) // select_count
            col_idx = (start_idx + i) % select_count
            score_offset = row_idx * input_scores_stride_0 + col_idx * input_scores_stride_1
            
            score_val = tl.load(input_scores_ptr + score_offset)
            
            unit_idx = tl.get_element(unit_indices_tile, [i])
            pos_idx = tl.get_element(position_map_tile, [i])
            
            output_offset = unit_idx * group_buffer_stride_0 + pos_idx * group_buffer_stride_1
            tl.store(group_buffer_ptr + output_offset, score_val)


def aikg_put_triton_torch(input_scores, unit_indices, position_map, group_buffer):
    sequence_length, select_count = input_scores.shape
    total_elements = unit_indices.shape[0]
    num_groups = group_buffer.shape[0]
    
    if not input_scores.is_contiguous():
        input_scores = input_scores.contiguous()
    if not unit_indices.is_contiguous():
        unit_indices = unit_indices.contiguous()
    if not position_map.is_contiguous():
        position_map = position_map.contiguous()
    if not group_buffer.is_contiguous():
        group_buffer = group_buffer.contiguous()
    
    grid = lambda meta: (triton.cdiv(total_elements, meta['BLOCK_SIZE']),)
    
    aikg_put_kernel[grid](
        input_scores,
        unit_indices,
        position_map,
        group_buffer,
        sequence_length,
        select_count,
        total_elements,
        input_scores.stride(0),
        input_scores.stride(1),
        group_buffer.stride(0),
        group_buffer.stride(1),
    )
    
    return group_buffer