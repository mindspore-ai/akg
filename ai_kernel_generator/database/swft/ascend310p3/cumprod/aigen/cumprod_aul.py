def cumprod_op_impl_npu(input: U.TensorPtr, output: U.TensorPtr, tiling: list):
    BLOCK_DIM = 8
    TILE_SIZE = 256
    BATCH_SIZE = 128
    CORE_NUM = BLOCK_DIM

    core_idx = U.get_core_idx()
    batch_per_core = BATCH_SIZE // CORE_NUM
    start_batch = core_idx * batch_per_core
    end_batch = start_batch + batch_per_core

    # 创建Tile
    input_tile = U.Tile(shape=(TILE_SIZE,), dtype=U.float16, pos=U.VecBuf)
    cum_tile = U.Tile(shape=(TILE_SIZE,), dtype=U.float16, pos=U.VecBuf)
    prev_value = U.Tile(shape=(1,), dtype=U.float16, pos=U.VecBuf)

    # 初始化prev_value为1.0
    init_value = U.FilledTile((1,), U.float16, U.VecBuf, value=1.0)
    U.data_copy(dst=prev_value, src=init_value)

    for batch in range(start_batch, end_batch):
        # 重置prev_value
        U.data_copy(dst=prev_value, src=init_value)

        for tile_idx in U.Pipelined(iterations=16):  # 4000/256=15.625→16次循环
            start = tile_idx * TILE_SIZE
            end = start + TILE_SIZE

            # 加载数据到vector buffer
            U.data_copy(dst=input_tile, src=input[batch, start:end],
                       src_pos=U.GlobalMem, dst_pos=U.VecBuf)

            # 计算累积乘积
            U.vectorscalar_op(op="muls", dst=cum_tile, src=input_tile, factor=prev_value[0])
            # 递归倍增法实现前缀乘积
            s = 1
            while s < TILE_SIZE:
                U.vbinary_op(op="mul", dst=cum_tile[s:], src1=cum_tile[s:], src2=cum_tile[:-s])
                s *= 2

            # 写回结果
            U.data_copy(dst=output[batch, start:end], src=cum_tile,
                       src_pos=U.VecBuf, dst_pos=U.GlobalMem)

            # 更新prev_value
            last_value = U.Tile(shape=(1,), dtype=U.float16, pos=U.VecBuf)
            U.data_copy(dst=last_value, src=cum_tile[-1:])
            U.data_copy(dst=prev_value, src=last_value)

def cumprod_op_impl_tiling(input_np, output_np, tiling):
    BLOCK_DIM = 8
    WORKSPACE_SIZE = 0