def softplus_op_impl_npu(input: U.TensorPtr, output: U.TensorPtr, tiling: list):
    BLOCK_DIM = 8
    CORE_NUM = BLOCK_DIM
    batch_size = tiling[0]
    dim = tiling[1]
    dim_per_core = tiling[2]

    core_idx = U.get_core_idx()
    start_dim = core_idx * dim_per_core
    end_dim = start_dim + dim_per_core

    input_tile = U.Tile(shape=(batch_size, dim_per_core), dtype=U.float16, pos=U.VecBuf)
    temp_tile = U.Tile(shape=(batch_size, dim_per_core), dtype=U.float16, pos=U.VecBuf)

    U.data_copy(dst=input_tile, src=input[:, start_dim:end_dim], src_pos=U.GlobalMem, dst_pos=U.VecBuf)
    U.vunary_op(op="exp", dst=temp_tile, src=input_tile)
    U.vectorscalar_op(op="adds", dst=temp_tile, src=temp_tile, factor=1.0)
    U.vunary_op(op="log", dst=temp_tile, src=temp_tile)
    U.data_copy(dst=output[:, start_dim:end_dim], src=temp_tile, src_pos=U.VecBuf, dst_pos=U.GlobalMem)


def softplus_op_impl_tiling(input_np, output_np, tiling):
    BLOCK_DIM = 8
    WORKSPACE_SIZE = 0
    batch_size = 16
    dim = 16384
    tiling.append(batch_size)
    tiling.append(dim)
    tiling.append(dim // BLOCK_DIM)