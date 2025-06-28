def selu__op_impl_npu(input: U.TensorPtr, output: U.TensorPtr, tiling: list):
    BLOCK_DIM = 8
    TILE_SHAPE = (16, 2048)  # batch_size=16, dim_per_core=16384//8
    ALPHA = 1.6732632423543772
    SCALE = 1.0507009873554805

    # 获取核心ID并计算数据范围
    core_idx = U.get_core_idx()
    start_dim = core_idx * TILE_SHAPE[1]
    end_dim = start_dim + TILE_SHAPE[1]

    # 创建Tile
    input_tile = U.Tile(shape=TILE_SHAPE, dtype=U.float16, pos=U.VecBuf)
    output_tile = U.Tile(shape=TILE_SHAPE, dtype=U.float16, pos=U.VecBuf)
    
    # 预定义填充Tile
    zero_tile = U.FilledTile(TILE_SHAPE, U.float16, U.VecBuf, value=0.0)
    alpha_tile = U.FilledTile(TILE_SHAPE, U.float16, U.VecBuf, value=ALPHA)
    one_tile = U.FilledTile(TILE_SHAPE, U.float16, U.VecBuf, value=1.0)

    for _ in U.Pipelined(iterations=1):
        # GlobalMem -> VecBuf数据加载
        U.data_copy(dst=input_tile, src=input[:, start_dim:end_dim],
                   src_pos=U.GlobalMem, dst_pos=U.VecBuf)

        # 计算条件分支
        condition_tile = U.Tile(shape=TILE_SHAPE, dtype=U.bool, pos=U.VecBuf)
        U.vbinary_op(op="gt", dst=condition_tile, src1=input_tile, src2=zero_tile)

        # 计算指数分支
        exp_tile = U.Tile(shape=TILE_SHAPE, dtype=U.float16, pos=U.VecBuf)
        U.vunary_op(op="exp", dst=exp_tile, src=input_tile)
        U.vbinary_op(op="sub", dst=exp_tile, src1=exp_tile, src2=one_tile)
        U.vbinary_op(op="mul", dst=exp_tile, src1=exp_tile, src2=alpha_tile)

        # 条件选择
        U.vbinary_op(op="select", dst=output_tile,
                   src1=input_tile, src2=exp_tile, mask=condition_tile)

        # 乘以scale系数
        U.vectorscalar_op(op="muls", dst=output_tile, src=output_tile, factor=SCALE)

        # VecBuf -> GlobalMem数据写回
        U.data_copy(dst=output[:, start_dim:end_dim], src=output_tile,
                   src_pos=U.VecBuf, dst_pos=U.GlobalMem)

def selu__tiling(input_np, output_np, tiling):
    BLOCK_DIM = 8
    WORKSPACE_SIZE = 0