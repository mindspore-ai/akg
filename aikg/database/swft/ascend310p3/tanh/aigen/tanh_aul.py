import aul as U

def tanh_op_impl_npu(input_np: U.TensorPtr, output_np: U.TensorPtr, tiling: list):
    # 硬编码参数（来自Tiling函数）
    BATCH_SIZE = 16
    DIM = 16384
    BLOCK_DIM = 8
    SAMPLES_PER_CORE = BATCH_SIZE // BLOCK_DIM
    TILE_DIM = 4096
    NUM_TILES_PER_SAMPLE = DIM // TILE_DIM
    TOTAL_TILES_PER_CORE = SAMPLES_PER_CORE * NUM_TILES_PER_SAMPLE
    
    # 获取当前核ID
    core_idx = U.get_core_idx()
    start_batch = core_idx * SAMPLES_PER_CORE
    
    # 创建Tile（输入和输出）
    input_tile = U.Tile(shape=(1, TILE_DIM), dtype=U.float16, pos=U.VecBuf)
    output_tile = U.Tile(shape=(1, TILE_DIM), dtype=U.float16, pos=U.VecBuf)
    
    # 使用流水线循环处理数据
    for tile_idx in U.Pipelined(iterations=TOTAL_TILES_PER_CORE):
        # 计算当前处理的样本索引和维度偏移
        sample_in_core = tile_idx // NUM_TILES_PER_SAMPLE
        tile_in_sample = tile_idx % NUM_TILES_PER_SAMPLE
        
        # 计算全局索引
        sample_idx = start_batch + sample_in_core
        start_dim = tile_in_sample * TILE_DIM
        end_dim = start_dim + TILE_DIM
        
        # 数据搬运：从全局内存加载到向量缓存
        U.data_copy(dst=input_tile, 
                    src=input_np[sample_idx:sample_idx+1, start_dim:end_dim],
                    src_pos=U.GlobalMem, dst_pos=U.VecBuf)
        
        # 执行tanh计算
        U.vunary_op(op="tanh", dst=output_tile, src=input_tile)
        
        # 数据搬运：从向量缓存写回全局内存
        U.data_copy(dst=output_np[sample_idx:sample_idx+1, start_dim:end_dim],
                    src=output_tile,
                    src_pos=U.VecBuf, dst_pos=U.GlobalMem)

def tanh_op_impl_tiling(input_np, output_np, tiling):
    # 设置核数和工作空间大小
    BLOCK_DIM = 8
    WORKSPACE_SIZE = 0
    # 硬编码tiling参数（实际应由编译器计算）
    tiling.extend([
        16,   # BATCH_SIZE
        16384, # DIM
        4096   # TILE_DIM
    ])