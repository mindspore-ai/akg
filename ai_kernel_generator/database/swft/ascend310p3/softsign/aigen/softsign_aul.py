def softsign_op_impl_npu(input: U.TensorPtr, output: U.TensorPtr, tiling: list):
    BLOCK_DIM = 8
    WORKSPACE_SIZE = 0

    core_idx = U.get_core_idx()
    total_elements = 16 * 16384  # batch_size * dim
    elements_per_core = total_elements // BLOCK_DIM
    core_start = core_idx * elements_per_core

    TILE_SIZE = 4096
    LOOP_COUNT = elements_per_core // TILE_SIZE

    input_tile = U.Tile(shape=(TILE_SIZE,), dtype=U.float16, pos=U.VecBuf)
    abs_tile = U.Tile(shape=(TILE_SIZE,), dtype=U.float16, pos=U.VecBuf)
    denominator_tile = U.Tile(shape=(TILE_SIZE,), dtype=U.float16, pos=U.VecBuf)
    output_tile = U.Tile(shape=(TILE_SIZE,), dtype=U.float16, pos=U.VecBuf)

    for iter_idx in U.Pipelined(iterations=LOOP_COUNT):
        current_start = core_start + iter_idx * TILE_SIZE
        current_end = current_start + TILE_SIZE

        # Load input data
        U.data_copy(dst=input_tile, src=input[current_start:current_end], src_pos=U.GlobalMem, dst_pos=U.VecBuf)

        # Compute absolute value
        U.vunary_op(op="abs", dst=abs_tile, src=input_tile)

        # Compute denominator: abs + 1.0
        U.vectorscalar_op(op="adds", dst=denominator_tile, src=abs_tile, factor=1.0)

        # Compute division
        U.vbinary_op(op="div", dst=output_tile, src1=input_tile, src2=denominator_tile)

        # Store result
        U.data_copy(dst=output[current_start:current_end], src=output_tile, src_pos=U.VecBuf, dst_pos=U.GlobalMem)

def softsign_op_impl_tiling(input_np, output_np, tiling):
    BLOCK_DIM = 8
    WORKSPACE_SIZE = 0