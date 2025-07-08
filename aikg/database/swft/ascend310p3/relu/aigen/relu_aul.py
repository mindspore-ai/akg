def relu_op_impl_npu(input: U.TensorPtr, output: U.TensorPtr, tiling: list):
    BLOCK_DIM = 8
    BATCH_PER_CORE = 2
    TILE_SIZE = 1024
    LOOP_COUNT = 16

    core_idx = U.get_core_idx()
    
    input_tile = U.Tile(shape=(BATCH_PER_CORE, TILE_SIZE), dtype=U.float16,
                      pos=U.VecBuf)
    output_tile = U.Tile(shape=(BATCH_PER_CORE, TILE_SIZE), dtype=U.float16,
                       pos=U.VecBuf)

    for iter_idx in U.Pipelined(iterations=LOOP_COUNT):
        batch_offset = core_idx * BATCH_PER_CORE
        elem_offset = iter_idx * TILE_SIZE
        
        U.data_copy(dst=input_tile, 
                   src=input[batch_offset:batch_offset+BATCH_PER_CORE, elem_offset:elem_offset+TILE_SIZE],
                   src_pos=U.GlobalMem,
                   dst_pos=U.VecBuf)
        
        U.vunary_op(op="relu", dst=output_tile, src=input_tile)
        
        U.data_copy(dst=output[batch_offset:batch_offset+BATCH_PER_CORE, elem_offset:elem_offset+TILE_SIZE],
                   src=output_tile,
                   src_pos=U.VecBuf,
                   dst_pos=U.GlobalMem)

def relu_op_impl_tiling(input_np, output_np, tiling):
    BLOCK_DIM = 8
    WORKSPACE_SIZE = 0
    tiling.extend([
        BLOCK_DIM,
        WORKSPACE_SIZE,
        2,
        1024
    ])