import aul as U

def rmsnorm__op_impl_npu(input_np: U.TensorPtr, output_np: U.TensorPtr, tiling: list):
    # 硬编码shape值
    BATCH = 16
    FEATURES = 64
    DIM1 = 256
    DIM2 = 256
    BLOCK_DIM = 8
    TILE_DIM1 = DIM1 // BLOCK_DIM  # 32
    EPS = U.FilledTile((FEATURES, 1, 1), U.float32, U.VecBuf, value=1e-5)

    core_idx = U.get_core_idx()
    
    # 创建Tile（修改input_tile为float32类型）
    input_tile = U.Tile(shape=(FEATURES, TILE_DIM1, DIM2), dtype=U.float32, pos=U.VecBuf)
    squared_tile = U.Tile(shape=(FEATURES, TILE_DIM1, DIM2), dtype=U.float32, pos=U.VecBuf)
    mean_tile = U.Tile(shape=(FEATURES, 1, 1), dtype=U.float32, pos=U.VecBuf)
    rms_tile = U.Tile(shape=(FEATURES, 1, 1), dtype=U.float32, pos=U.VecBuf)

    # 流水线处理每个batch
    for batch_idx in U.Pipelined(iterations=BATCH):
        # 数据加载阶段（自动完成float16到float32类型转换）
        U.data_copy(
            dst=input_tile,
            src=input_np[batch_idx, :, core_idx*TILE_DIM1:(core_idx+1)*TILE_DIM1, :],
            src_pos=U.GlobalMem,
            dst_pos=U.VecBuf
        )

        # 计算平方
        U.vbinary_op(op="mul", dst=squared_tile, src1=input_tile, src2=input_tile)

        # 沿features维度求均值（修正axis参数为0）
        U.vreduce_op(op="sum", dst=mean_tile, src=squared_tile, axis=0)
        U.vectorscalar_op(op="muls", dst=mean_tile, src=mean_tile, factor=1.0/FEATURES)

        # 加epsilon并开平方（修正EPS形状匹配）
        U.vbinary_op(op="add", dst=rms_tile, src1=mean_tile, src2=EPS)
        U.vunary_op(op="sqrt", dst=rms_tile, src=rms_tile)

        # 除法计算（保持float32精度）
        U.vbinary_op(op="div", dst=input_tile, src1=input_tile, src2=rms_tile)

        # 结果写回（自动完成float32到float16类型转换）
        U.data_copy(
            dst=output_np[batch_idx, :, core_idx*TILE_DIM1:(core_idx+1)*TILE_DIM1, :],
            src=input_tile,
            src_pos=U.VecBuf,
            dst_pos=U.GlobalMem
        )

def rmsnorm__op_impl_tiling(input_np, output_np, tiling):
    BLOCK_DIM = 8
    WORKSPACE_SIZE = 0
    # 硬编码shape值
    BATCH = 16
    FEATURES = 64
    DIM1 = 256
    DIM2 = 256
    
    # 核间并行参数
    TILE_DIM1 = DIM1 // BLOCK_DIM  # 32
    
    # 内存对齐校验 (32字节对齐)
    assert (TILE_DIM1 * DIM2 * 2) % 32 == 0  # float16=2bytes
    
    # 设置必要的tiling参数
    tiling.append(BLOCK_DIM)
    tiling.append(WORKSPACE_SIZE)