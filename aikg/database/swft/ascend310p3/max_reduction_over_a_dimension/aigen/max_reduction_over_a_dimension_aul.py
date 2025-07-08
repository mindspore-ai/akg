import aul as U

def max_reduction_over_a_dimension_op_impl_npu(input_np: U.TensorPtr, output_np: U.TensorPtr, tiling: list):
    # 硬编码参数（来自Tiling函数）
    BATCH_SIZE = 16
    DIM1 = 256
    DIM2 = 256
    BLOCK_DIM = 8  # 核数，由Tiling函数计算得出
    SAMPLES_PER_CORE = BATCH_SIZE // BLOCK_DIM  # 每个核处理2个batch
    
    # 获取当前核ID
    core_idx = U.get_core_idx()
    
    # 创建Tile（放在循环外避免重复分配）
    output_tile = U.Tile(shape=(1, DIM2), dtype=U.float16, pos=U.VecBuf)
    input_tile = U.Tile(shape=(1, 1, DIM2), dtype=U.float16, pos=U.VecBuf)
    
    # 外层流水线循环：处理每个batch
    for i in U.Pipelined(iterations=SAMPLES_PER_CORE):
        current_batch = core_idx * SAMPLES_PER_CORE + i
        
        # 初始化output_tile为最小值（float16的最小值）
        U.vunary_op(op="vector_dup", dst=output_tile, fill_shape=[1, DIM2], fill_value=-65504.0)
        
        # 内层流水线循环：遍历DIM1（规约轴）
        for j in U.Pipelined(iterations=DIM1):
            # 从GlobalMem加载数据到VecBuf
            U.data_copy(dst=input_tile, 
                        src=input_np[current_batch:current_batch+1, j:j+1, 0:DIM2],
                        src_pos=U.GlobalMem, 
                        dst_pos=U.VecBuf)
            
            # 计算最大值: output_tile = max(output_tile, input_tile)
            U.vbinary_op(op="max", dst=output_tile, src1=output_tile, src2=input_tile)
        
        # 将结果写回GlobalMem
        U.data_copy(dst=output_np[current_batch:current_batch+1, 0:DIM2], 
                    src=output_tile,
                    src_pos=U.VecBuf, 
                    dst_pos=U.GlobalMem)

# Tiling函数
def max_reduction_over_a_dimension_op_impl_tiling(input_np, output_np, tiling):
    # 设置核数（8核并行处理16个batch）
    BLOCK_DIM = 8
    WORKSPACE_SIZE = 0
