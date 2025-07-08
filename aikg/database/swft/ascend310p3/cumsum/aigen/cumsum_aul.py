def cumsum_op_impl_npu(input: U.TensorPtr, output: U.TensorPtr, tiling: list):
    core_idx = U.get_core_idx()
    BLOCK_DIM = 8
    BATCH_PER_CORE = 16  # 128/8=16
    ELEMENT_NUM = 4000

    # 创建Tile（放在循环外避免重复分配）
    input_tile = U.Tile(shape=(ELEMENT_NUM,), dtype=U.float16, pos=U.VecBuf)
    output_tile = U.Tile(shape=(ELEMENT_NUM,), dtype=U.float16, pos=U.VecBuf)

    # 流水线处理每个batch
    for batch_offset in U.Pipelined(iterations=BATCH_PER_CORE):
        # 计算全局batch索引
        global_batch = core_idx * BATCH_PER_CORE + batch_offset

        # 1. 加载输入数据到向量缓存
        U.data_copy(dst=input_tile, 
                   src=input[global_batch, 0:ELEMENT_NUM],
                   src_pos=U.GlobalMem,
                   dst_pos=U.VecBuf)

        # 2. 计算cumsum（向量化实现）
        # 2.1 初始化首元素
        U.data_copy(dst=output_tile[0:1], src=input_tile[0:1])
        
        # 2.2 前缀和计算
        for i in range(1, ELEMENT_NUM):
            prev = output_tile[i-1:i]
            current = input_tile[i:i+1]
            U.vbinary_op(op="add", 
                        dst=output_tile[i:i+1],
                        src1=prev,
                        src2=current)

        # 3. 结果写回全局内存
        U.data_copy(dst=output[global_batch, 0:ELEMENT_NUM],
                   src=output_tile,
                   src_pos=U.VecBuf,
                   dst_pos=U.GlobalMem)

def cumsum_op_impl_tiling(input_np, output_np, tiling):
    BLOCK_DIM = 8
    WORKSPACE_SIZE = 0