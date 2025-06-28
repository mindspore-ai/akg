import aul as U

def frobeniusnorm__op_impl_npu(input_np: U.TensorPtr, output_np: U.TensorPtr, tiling: list):
    # 获取当前核ID
    core_idx = U.get_core_idx()
    
    # 硬编码参数（来自Tiling函数）
    BATCH_SIZE = 16
    FEATURES = 64
    DIM1 = 256
    DIM2 = 256
    BLOCK_DIM = 8
    SAMPLES_PER_CORE = BATCH_SIZE * FEATURES // BLOCK_DIM
    
    # 计算当前核处理的数据范围
    start_idx = core_idx * SAMPLES_PER_CORE
    end_idx = start_idx + SAMPLES_PER_CORE
    
    # 创建累加器Tile（用于存储平方和）
    acc_tile = U.Tile(shape=(1,), dtype=U.float32, pos=U.VecBuf)
    # 创建临时Tile用于存储样本平方和
    sample_sum_tile = U.Tile(shape=(1,), dtype=U.float32, pos=U.VecBuf)
    # 初始化累加器
    U.vectorscalar_op(op="muls", dst=acc_tile, src=acc_tile, factor=0.0)
    
    # 创建输入数据Tile
    input_tile = U.Tile(shape=(1, DIM1, DIM2), dtype=U.float16, pos=U.VecBuf)
    
    # 第一阶段：计算平方和
    for i in U.Pipelined(iterations=SAMPLES_PER_CORE):
        # 计算当前样本索引
        sample_idx = start_idx + i
        
        # 计算全局偏移量
        offset = sample_idx * DIM1 * DIM2
        
        # 加载输入数据到VecBuf
        U.data_copy(dst=input_tile, 
                   src=input_np[offset:offset+DIM1*DIM2], 
                   src_pos=U.GlobalMem, 
                   dst_pos=U.VecBuf)
        
        # 计算平方
        U.vbinary_op(op="mul", dst=input_tile, src1=input_tile, src2=input_tile)
        
        # 计算当前样本的平方和
        U.vreduce_op(op="sum", dst=sample_sum_tile, src=input_tile, axis=-1)
        # 累加到总平方和
        U.vbinary_op(op="add", dst=acc_tile, src1=acc_tile, src2=sample_sum_tile)
    
    # 第二阶段：计算全局范数
    # 计算平方根
    U.vunary_op(op="sqrt", dst=acc_tile, src=acc_tile)
    
    # 第三阶段：归一化处理
    for i in U.Pipelined(iterations=SAMPLES_PER_CORE):
        # 计算当前样本索引
        sample_idx = start_idx + i
        
        # 计算全局偏移量
        offset = sample_idx * DIM1 * DIM2
        
        # 加载输入数据到VecBuf
        U.data_copy(dst=input_tile, 
                   src=input_np[offset:offset+DIM1*DIM2], 
                   src_pos=U.GlobalMem, 
                   dst_pos=U.VecBuf)
        
        # 归一化：除以范数
        U.vectorscalar_op(op="divs", dst=input_tile, src=input_tile, factor=acc_tile)
        
        # 写回结果
        U.data_copy(dst=output_np[offset:offset+DIM1*DIM2], 
                   src=input_tile, 
                   src_pos=U.VecBuf, 
                   dst_pos=U.GlobalMem)

def frobeniusnorm__op_impl_tiling(input_np, output_np, tiling):
    # 设置核数为8
    BLOCK_DIM = 8
    # 工作空间大小为0
    WORKSPACE_SIZE = 0
