import aul as U

def batchnorm_op_impl_npu(input_np: U.TensorPtr, output_np: U.TensorPtr, tiling: list):
    # 硬编码参数（来自Tiling函数）
    BLOCK_DIM = 8
    BATCH_SIZE = 16
    FEATURES = 64
    DIM1 = 256
    DIM2 = 256
    TILE_H = 16
    TILE_W = 16
    C_PER_CORE = FEATURES // BLOCK_DIM
    SPATIAL_TILES = (DIM1 // TILE_H) * (DIM2 // TILE_W)
    
    # 获取当前核心ID
    core_idx = U.get_core_idx()
    start_channel = core_idx * C_PER_CORE
    
    # 创建累加器Tile（每个通道的sum和sum_square）
    sum_tile = U.Tile(shape=(C_PER_CORE,), dtype=U.float32, pos=U.VecBuf)
    sum_sq_tile = U.Tile(shape=(C_PER_CORE,), dtype=U.float32, pos=U.VecBuf)
    
    # 初始化累加器为0
    zero_tile = U.FilledTile(shape=(C_PER_CORE,), dtype=U.float32, value=0.0)
    U.data_copy(dst=sum_tile, src=zero_tile)
    U.data_copy(dst=sum_sq_tile, src=zero_tile)
    
    # 第一遍循环：计算每个通道的sum和sum_square
    input_tile = U.Tile(shape=(BATCH_SIZE, C_PER_CORE, TILE_H, TILE_W), 
                       dtype=U.float16, pos=U.VecBuf)
    
    for block_idx in U.Pipelined(iterations=SPATIAL_TILES):
        # 计算空间块坐标
        block_h = block_idx // (DIM2 // TILE_W)
        block_w = block_idx % (DIM2 // TILE_W)
        h_start = block_h * TILE_H
        w_start = block_w * TILE_W
        
        # 加载输入数据块
        U.data_copy(
            dst=input_tile,
            src=input_np[0:BATCH_SIZE, start_channel:start_channel+C_PER_CORE, 
                         h_start:h_start+TILE_H, w_start:w_start+TILE_W],
            src_pos=U.GlobalMem,
            dst_pos=U.VecBuf
        )
        
        # 转换为float32并计算平方
        input_f32 = U.Tile(shape=input_tile.shape, dtype=U.float32, pos=U.VecBuf)
        U.vconv(dst=input_f32, src=input_tile)
        
        squared_tile = U.Tile(shape=input_f32.shape, dtype=U.float32, pos=U.VecBuf)
        U.vmul(dst=squared_tile, src1=input_f32, src2=input_f32)
        
        # 沿空间维度求和（保留通道维度）
        U.vreduce_op(op="sum", dst=sum_tile, src=input_f32, axis=(0,2,3))
        U.vreduce_op(op="sum", dst=sum_sq_tile, src=squared_tile, axis=(0,2,3))
    
    # 计算全局统计量（mean和var）
    total_pixels = BATCH_SIZE * DIM1 * DIM2
    total_pixels_tile = U.FilledTile(shape=(C_PER_CORE,), dtype=U.float32, value=float(total_pixels))
    
    mean_tile = U.Tile(shape=(C_PER_CORE,), dtype=U.float32, pos=U.VecBuf)
    var_tile = U.Tile(shape=(C_PER_CORE,), dtype=U.float32, pos=U.VecBuf)
    
    U.vdiv(dst=mean_tile, src1=sum_tile, src2=total_pixels_tile)
    
    mean_sq_tile = U.Tile(shape=(C_PER_CORE,), dtype=U.float32, pos=U.VecBuf)
    U.vmul(dst=mean_sq_tile, src1=mean_tile, src2=mean_tile)
    
    sum_sq_div_tile = U.Tile(shape=(C_PER_CORE,), dtype=U.float32, pos=U.VecBuf)
    U.vdiv(dst=sum_sq_div_tile, src1=sum_sq_tile, src2=total_pixels_tile)
    
    U.vsub(dst=var_tile, src1=sum_sq_div_tile, src2=mean_sq_tile)
    
    # 添加epsilon并计算标准差倒数
    eps_tile = U.FilledTile(shape=(C_PER_CORE,), dtype=U.float32, value=1e-5)
    var_eps_tile = U.Tile(shape=(C_PER_CORE,), dtype=U.float32, pos=U.VecBuf)
    U.vadd(dst=var_eps_tile, src1=var_tile, src2=eps_tile)
    
    inv_std_tile = U.Tile(shape=(C_PER_CORE,), dtype=U.float32, pos=U.VecBuf)
    U.vsqrt(dst=inv_std_tile, src=var_eps_tile)
    U.vrec(dst=inv_std_tile, src=inv_std_tile)
    
    # 第二遍循环：归一化计算
    norm_tile = U.Tile(shape=(BATCH_SIZE, C_PER_CORE, TILE_H, TILE_W), 
                      dtype=U.float16, pos=U.VecBuf)
    
    for block_idx in U.Pipelined(iterations=SPATIAL_TILES):
        # 计算空间块坐标
        block_h = block_idx // (DIM2 // TILE_W)
        block_w = block_idx % (DIM2 // TILE_W)
        h_start = block_h * TILE_H
        w_start = block_w * TILE_W
        
        # 加载输入数据块
        U.data_copy(
            dst=norm_tile,
            src=input_np[0:BATCH_SIZE, start_channel:start_channel+C_PER_CORE, 
                         h_start:h_start+TILE_H, w_start:w_start+TILE_W],
            src_pos=U.GlobalMem,
            dst_pos=U.VecBuf
        )
        
        # 转换为float32计算
        norm_f32 = U.Tile(shape=norm_tile.shape, dtype=U.float32, pos=U.VecBuf)
        U.vconv(dst=norm_f32, src=norm_tile)
        
        # 减去均值
        mean_broadcast = U.Tile(shape=norm_f32.shape, dtype=U.float32, pos=U.VecBuf)
        U.vbrcb(dst=mean_broadcast, src=mean_tile, axis=1)
        U.vsub(dst=norm_f32, src1=norm_f32, src2=mean_broadcast)
        
        # 乘以标准差倒数
        inv_std_broadcast = U.Tile(shape=norm_f32.shape, dtype=U.float32, pos=U.VecBuf)
        U.vbrcb(dst=inv_std_broadcast, src=inv_std_tile, axis=1)
        U.vmul(dst=norm_f32, src1=norm_f32, src2=inv_std_broadcast)
        
        # 转换回float16并存储
        U.vconv(dst=norm_tile, src=norm_f32)
        
        U.data_copy(
            dst=output_np[0:BATCH_SIZE, start_channel:start_channel+C_PER_CORE, 
                          h_start:h_start+TILE_H, w_start:w_start+TILE_W],
            src=norm_tile,
            src_pos=U.VecBuf,
            dst_pos=U.GlobalMem
        )

def batchnorm_op_impl_tiling(input_np, output_np, tiling):
    BLOCK_DIM = 8
    WORKSPACE_SIZE = 0
