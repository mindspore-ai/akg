def cumsum_exclusive_op_impl_npu(input: U.TensorPtr, output: U.TensorPtr, tiling: list):
    BLOCK_DIM = 8
    BATCH_PER_CORE = 16
    INPUT_LEN = 4000
    
    core_idx = U.get_core_idx()
    start_batch = core_idx * BATCH_PER_CORE
    end_batch = start_batch + BATCH_PER_CORE
    
    # 预分配Tile
    input_tile = U.Tile(shape=(INPUT_LEN,), dtype=U.float16, pos=U.VecBuf)
    zero_tile = U.FilledTile((1,), U.float16, U.VecBuf, 0.0)
    concat_tile = U.Tile(shape=(INPUT_LEN+1,), dtype=U.float16, pos=U.VecBuf)
    sliced_tile = U.Tile(shape=(INPUT_LEN,), dtype=U.float16, pos=U.VecBuf)
    sum_tile = U.Tile(shape=(INPUT_LEN,), dtype=U.float16, pos=U.VecBuf)
    
    # 流水线处理批次维度
    for batch_idx in U.Pipelined(iterations=BATCH_PER_CORE):
        # 加载当前批次数据
        U.data_copy(dst=input_tile, src=input[start_batch+batch_idx, :], 
                   src_pos=U.GlobalMem, dst_pos=U.VecBuf)
        
        # 构造concat_tile [0|input]
        U.data_copy(dst=concat_tile[0:1], src=zero_tile)
        U.data_copy(dst=concat_tile[1:], src=input_tile)
        
        # 切片取前N-1元素
        U.data_copy(dst=sliced_tile, src=concat_tile[:-1])
        
        # 计算累积和
        U.data_copy(dst=sum_tile[0:1], src=sliced_tile[0:1])
        for i in range(1, INPUT_LEN):
            prev = sum_tile[i-1:i]
            current = sliced_tile[i:i+1]
            U.vbinary_op(op="add", dst=sum_tile[i:i+1], src1=prev, src2=current)
        
        # 存储结果
        U.data_copy(dst=output[start_batch+batch_idx, :], src=sum_tile,
                   src_pos=U.VecBuf, dst_pos=U.GlobalMem)

def cumsum_exclusive_op_impl_tiling(input_np, output_np, tiling):
    BLOCK_DIM = 8
    WORKSPACE_SIZE = 0