def swish_op_impl_npu(input: U.TensorPtr, output: U.TensorPtr, tiling: list):
    BLOCK_DIM = 8
    CORE_TILE_ROWS = 16
    CORE_TILE_COLS = 2048
    LOOP_COUNT = 1

    core_idx = U.get_core_idx()
    
    # 创建Tile
    input_tile = U.Tile(
        shape=(CORE_TILE_ROWS, CORE_TILE_COLS),
        dtype=U.float16,
        pos=U.VecBuf
    )
    sigmoid_tile = U.Tile(
        shape=(CORE_TILE_ROWS, CORE_TILE_COLS),
        dtype=U.float16,
        pos=U.VecBuf
    )
    output_tile = U.Tile(
        shape=(CORE_TILE_ROWS, CORE_TILE_COLS),
        dtype=U.float16,
        pos=U.VecBuf
    )

    for iter_idx in U.Pipelined(iterations=LOOP_COUNT):
        # 计算当前核的数据偏移
        col_start = core_idx * CORE_TILE_COLS
        col_end = (core_idx + 1) * CORE_TILE_COLS

        # 加载输入数据
        U.data_copy(
            dst=input_tile,
            src=input[:, col_start:col_end],
            src_pos=U.GlobalMem,
            dst_pos=U.VecBuf
        )

        # 计算sigmoid部分
        U.vunary_op(op="neg", dst=sigmoid_tile, src=input_tile)
        U.vunary_op(op="exp", dst=sigmoid_tile, src=sigmoid_tile)
        U.vectorscalar_op(op="adds", dst=sigmoid_tile, src=sigmoid_tile, factor=1.0)
        U.vunary_op(op="reciprocal", dst=sigmoid_tile, src=sigmoid_tile)

        # 计算element-wise乘法
        U.vbinary_op(op="mul", dst=output_tile, src1=input_tile, src2=sigmoid_tile)

        # 写回结果
        U.data_copy(
            dst=output[:, col_start:col_end],
            src=output_tile,
            src_pos=U.VecBuf,
            dst_pos=U.GlobalMem
        )

def swish_op_impl_tiling(input_np, output_np, tiling):
    BLOCK_DIM = 8
    WORKSPACE_SIZE = 0