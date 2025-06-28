def leakyrelu_op_impl_npu(input: U.TensorPtr, output: U.TensorPtr, tiling: list):
    BLOCK_DIM = 8
    CORE_NUM = BLOCK_DIM
    total_batch = 16
    dim_per_core = 16384 // CORE_NUM
    
    core_idx = U.get_core_idx()
    start_dim = core_idx * dim_per_core
    end_dim = start_dim + dim_per_core
    
    input_tile = U.Tile(shape=(dim_per_core,), dtype=U.float16, pos=U.VecBuf)
    scaled_tile = U.Tile(shape=(dim_per_core,), dtype=U.float16, pos=U.VecBuf)
    output_tile = U.Tile(shape=(dim_per_core,), dtype=U.float16, pos=U.VecBuf)
    
    for batch in U.Pipelined(iterations=total_batch):
        U.data_copy(dst=input_tile, src=input[batch, start_dim:end_dim], 
                    src_pos=U.GlobalMem, dst_pos=U.VecBuf)
        U.vectorscalar_op(op="muls", dst=scaled_tile, src=input_tile, factor=0.01)
        U.vbinary_op(op="max", dst=output_tile, src1=input_tile, src2=scaled_tile)
        U.data_copy(dst=output[batch, start_dim:end_dim], src=output_tile,
                    src_pos=U.VecBuf, dst_pos=U.GlobalMem)


def leakyrelu_op_impl_tiling(input_np, output_np, tiling):
    BLOCK_DIM = 8
    WORKSPACE_SIZE = 0