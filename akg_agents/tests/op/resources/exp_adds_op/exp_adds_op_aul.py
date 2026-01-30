def exp_adds_op_impl_npu(input_np: U.TensorPtr, output_np: U.TensorPtr, static_config: list):
    # 多核并行配置
    BLOCK_DIM = 8
    core_idx = U.get_core_idx()

    # 硬编码shape参数
    ROWS_PER_CORE = 5  # 40//8=5
    VEC_LEN = 256

    # 创建Tile（循环外创建避免重复分配）
    input_tile = U.Tile(shape=(VEC_LEN,), dtype=U.float16, pos=U.VecBuf)
    tmp_tile = U.Tile(shape=(VEC_LEN,), dtype=U.float16, pos=U.VecBuf)
    output_tile = U.Tile(shape=(VEC_LEN,), dtype=U.float16, pos=U.VecBuf)

    # 流水线循环处理5行数据
    for iter_idx in U.Pipelined(iterations=ROWS_PER_CORE):
        # 计算当前行全局索引
        row_idx = core_idx * ROWS_PER_CORE + iter_idx

        # Stage1: 从GlobalMem加载数据到向量缓存
        U.data_copy(dst=input_tile, src=input_np[row_idx, 0:VEC_LEN],
                    src_pos=U.GlobalMem, dst_pos=U.VecBuf)

        # Stage2: 向量计算（指数+标量加）
        U.vunary_op(op="exp", dst=tmp_tile, src=input_tile)
        U.vectorscalar_op(op="adds", dst=output_tile, src=tmp_tile, factor=1.0)

        # Stage3: 写回结果到GlobalMem
        U.data_copy(dst=output_np[row_idx, 0:VEC_LEN], src=output_tile,
                    src_pos=U.VecBuf, dst_pos=U.GlobalMem)


def exp_adds_op_impl_tiling(input_np, output_np, tiling):
    BLOCK_DIM = 8
    WORKSPACE_SIZE = 0
    pass
