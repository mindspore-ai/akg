def logsoftmax_op_impl_npu(input_np: U.TensorPtr, output_np: U.TensorPtr, tiling: list):
    # 硬编码参数
    BATCH_SIZE = 16
    DIM = 16384
    BLOCK_DIM = 8
    SAMPLES_PER_CORE = BATCH_SIZE // BLOCK_DIM
    
    # 获取当前核ID
    core_idx = U.get_core_idx()
    start_batch = core_idx * SAMPLES_PER_CORE
    
    # 创建双缓冲Tile
    input_tile = U.Tile(shape=(1, DIM), dtype=U.float16, pos=U.VecBuf, buffer_num=2)
    shifted_tile = U.Tile(shape=(1, DIM), dtype=U.float16, pos=U.VecBuf, buffer_num=2)
    exp_tile = U.Tile(shape=(1, DIM), dtype=U.float16, pos=U.VecBuf, buffer_num=2)
    output_tile = U.Tile(shape=(1, DIM), dtype=U.float16, pos=U.VecBuf, buffer_num=2)
    max_val_tile = U.Tile(shape=(1, 1), dtype=U.float16, pos=U.VecBuf, buffer_num=2)
    sum_exp_tile = U.Tile(shape=(1, 1), dtype=U.float16, pos=U.VecBuf, buffer_num=2)
    log_sum_exp_tile = U.Tile(shape=(1, 1), dtype=U.float16, pos=U.VecBuf, buffer_num=2)
     

    # 流水线处理每个样本
    for sample_idx in U.Pipelined(iterations=SAMPLES_PER_CORE):
        global_idx = start_batch + sample_idx
         
        # 1. 加载输入数据
        U.data_copy(dst=input_tile, src=input_np[global_idx:global_idx+1, 0:DIM], 
                    src_pos=U.GlobalMem, dst_pos=U.VecBuf)
         
        # 2. 计算最大值
        U.vreduce_op(op="max", dst=max_val_tile, src=input_tile, axis=1)
         
        # 3. 计算 shifted = input - max_val
        U.vectorscalar_op(op="sub", dst=shifted_tile, src=input_tile, factor=max_val_tile)
         
        # 4. 计算指数
        U.vunary_op(op="exp", dst=exp_tile, src=shifted_tile)
         
        # 5. 计算指数和
        U.vreduce_op(op="sum", dst=sum_exp_tile, src=exp_tile, axis=1)

        # 6. 计算log(指数和)
        U.vunary_op(op="ln", dst=log_sum_exp_tile, src=sum_exp_tile)
         
        # 7. 计算log_softmax = shifted - log_sum_exp
        U.vectorscalar_op(op="sub", dst=output_tile, src=shifted_tile, factor=log_sum_exp_tile)
         
        # 8. 写回结果
        U.data_copy(dst=output_np[global_idx:global_idx+1, 0:DIM], src=output_tile, 
                    src_pos=U.VecBuf, dst_pos=U.GlobalMem)
 
def logsoftmax_op_impl_tiling(input_np, output_np, tiling):
    BLOCK_DIM = 8
    WORKSPACE_SIZE = 0