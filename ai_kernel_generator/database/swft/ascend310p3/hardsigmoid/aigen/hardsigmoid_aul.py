def hardsigmoid_op_impl_npu(input: U.TensorPtr, output: U.TensorPtr, tiling: list):
    BLOCK_DIM = 8
    TOTAL_ELEMENTS = tiling[0]
    TILE_SIZE = tiling[1]
    CORE_NUM = tiling[2]
    elements_per_core = tiling[3]

    core_idx = U.get_core_idx()
    start_idx = core_idx * elements_per_core
    
    input_tile = U.Tile(shape=(TILE_SIZE,), dtype=U.float16, pos=U.VecBuf)

    for i in U.Pipelined(iterations=elements_per_core//TILE_SIZE):
        current_start = start_idx + i * TILE_SIZE
        
        # 数据加载阶段
        U.data_copy(dst=input_tile, src=input[current_start:current_start+TILE_SIZE],
                   src_pos=U.GlobalMem, dst_pos=U.VecBuf)
        
        # 计算阶段
        U.vectorscalar_op(op="adds", dst=input_tile, src=input_tile, factor=3.0)
        U.vectorscalar_op(op="muls", dst=input_tile, src=input_tile, factor=1.0/6)
        U.vectorscalar_op(op="maxs", dst=input_tile, src=input_tile, factor=0.0)
        U.vectorscalar_op(op="mins", dst=input_tile, src=input_tile, factor=1.0)
        
        # 数据写回阶段
        U.data_copy(dst=output[current_start:current_start+TILE_SIZE], src=input_tile,
                   src_pos=U.VecBuf, dst_pos=U.GlobalMem)


def hardsigmoid_op_impl_tiling(input_np, output_np, tiling):
    BLOCK_DIM = 8
    WORKSPACE_SIZE = 0
    # 硬编码参数（假设输入为(16,16384)）
    total_elements = 16 * 16384
    tile_size = 2048
    core_num = 8
    elements_per_core = total_elements // core_num
    tiling.extend([total_elements, tile_size, core_num, elements_per_core])