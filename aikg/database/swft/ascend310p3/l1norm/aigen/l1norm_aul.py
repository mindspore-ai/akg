def l1norm__op_impl_npu(input: U.TensorPtr, output: U.TensorPtr, tiling: list):
    BLOCK_DIM = 8
    batch_size = 16
    dim = 16384
    batch_per_core = batch_size // BLOCK_DIM

    core_idx = U.get_core_idx()
    start_idx = core_idx * batch_per_core

    input_tile = U.Tile(shape=(1, dim), dtype=U.float16, pos=U.VecBuf)
    abs_tile = U.Tile(shape=(1, dim), dtype=U.float16, pos=U.VecBuf)
    sum_tile = U.Tile(shape=(1, 1), dtype=U.float16, pos=U.VecBuf)
    output_tile = U.Tile(shape=(1, dim), dtype=U.float16, pos=U.VecBuf)

    for iter_idx in U.Pipelined(iterations=batch_per_core):
        current_batch = start_idx + iter_idx
        
        # 流水线阶段1: 加载数据
        U.data_copy(dst=input_tile, 
                   src=input[current_batch:current_batch+1, 0:dim],
                   src_pos=U.GlobalMem,
                   dst_pos=U.VecBuf)
        
        # 流水线阶段2: 计算逻辑
        U.vunary_op(op="abs", dst=abs_tile, src=input_tile)
        U.vreduce_op(op="sum", dst=sum_tile, src=abs_tile, axis=1)
        U.vbinary_op(op="div", dst=output_tile, src1=input_tile, src2=sum_tile)
        
        # 流水线阶段3: 存储结果
        U.data_copy(dst=output[current_batch:current_batch+1, 0:dim],
                   src=output_tile,
                   src_pos=U.VecBuf,
                   dst_pos=U.GlobalMem)

def l1norm__tiling(input_np, output_np, tiling):
    BLOCK_DIM = 8
    WORKSPACE_SIZE = 0