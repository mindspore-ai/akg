def softmax_op_impl_npu(input: U.TensorPtr, output: U.TensorPtr, tiling: list):
    BATCH_SIZE = 16
    DIM = 16384
    BLOCK_DIM = 8
    SAMPLES_PER_CORE = BATCH_SIZE // BLOCK_DIM

    core_idx = U.get_core_idx()
    start_batch = core_idx * SAMPLES_PER_CORE
    
    # 创建Tile
    input_tile = U.Tile(shape=(1, DIM), dtype=U.float16, pos=U.VecBuf)
    max_tile = U.Tile(shape=(1,), dtype=U.float16, pos=U.VecBuf)
    exp_tile = U.Tile(shape=(1, DIM), dtype=U.float16, pos=U.VecBuf)
    sum_tile = U.Tile(shape=(1,), dtype=U.float16, pos=U.VecBuf)
    
    # 流水线处理每个样本
    for i in U.Pipelined(iterations=SAMPLES_PER_CORE):  # 2次迭代
        current_batch = start_batch + i
        
        # 加载输入数据到VecBuf
        U.data_copy(dst=input_tile, 
                    src=input_np[current_batch:current_batch+1, 0:DIM], 
                    src_pos=U.GlobalMem, dst_pos=U.VecBuf)
        
        # 1. 求最大值（沿dim轴）
        U.vreduce_op(op="max", dst=max_tile, src=input_tile, axis=1)
        
        # 2. 减去最大值
        U.vectorscalar_op(op="subs", dst=input_tile, src=input_tile, factor=max_tile)
        
        # 3. 计算指数
        U.vunary_op(op="exp", dst=exp_tile, src=input_tile)
        
        # 4. 求和
        U.vreduce_op(op="sum", dst=sum_tile, src=exp_tile, axis=1)
        
        # 5. 除法（每个元素除以和）
        U.vectorscalar_op(op="divs", dst=exp_tile, src=exp_tile, factor=sum_tile)
        
        # 写回结果到GlobalMem
        U.data_copy(dst=output_np[current_batch:current_batch+1, 0:DIM], 
                    src=exp_tile, 
                    src_pos=U.VecBuf, dst_pos=U.GlobalMem)

def softmax_op_impl_tiling(input_np, output_np, tiling):
    BLOCK_DIM = 8
    WORKSPACE_SIZE = 0