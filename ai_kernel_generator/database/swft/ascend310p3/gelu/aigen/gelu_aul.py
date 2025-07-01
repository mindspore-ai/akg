import aul as U

def gelu__op_impl_npu(input_np: U.TensorPtr, output_np: U.TensorPtr, tiling: list):
    # 硬编码参数（来自Tiling函数）
    BLOCK_DIM = 8
    BATCH_SIZE = 16
    DIM = 16384
    TILE_LEN = 256
    
    # 计算分块参数
    SAMPLES_PER_CORE = BATCH_SIZE // BLOCK_DIM  # 每个核处理的样本数
    BLOCKS_PER_SAMPLE = DIM // TILE_LEN       # 每个样本的分块数
    
    # 获取当前核ID
    core_idx = U.get_core_idx()
    start_batch = core_idx * SAMPLES_PER_CORE
    
    # 创建Tile
    input_tile = U.Tile(shape=(SAMPLES_PER_CORE, TILE_LEN), dtype=U.float16, pos=U.VecBuf)
    x3_tile = U.Tile(shape=(SAMPLES_PER_CORE, TILE_LEN), dtype=U.float16, pos=U.VecBuf)
    term1_tile = U.Tile(shape=(SAMPLES_PER_CORE, TILE_LEN), dtype=U.float16, pos=U.VecBuf)
    inner_tile = U.Tile(shape=(SAMPLES_PER_CORE, TILE_LEN), dtype=U.float16, pos=U.VecBuf)
    term2_tile = U.Tile(shape=(SAMPLES_PER_CORE, TILE_LEN), dtype=U.float16, pos=U.VecBuf)
    tanh_term_tile = U.Tile(shape=(SAMPLES_PER_CORE, TILE_LEN), dtype=U.float16, pos=U.VecBuf)
    one_plus_tanh_tile = U.Tile(shape=(SAMPLES_PER_CORE, TILE_LEN), dtype=U.float16, pos=U.VecBuf)
    half_x_tile = U.Tile(shape=(SAMPLES_PER_CORE, TILE_LEN), dtype=U.float16, pos=U.VecBuf)
    output_tile = U.Tile(shape=(SAMPLES_PER_CORE, TILE_LEN), dtype=U.float16, pos=U.VecBuf)
    
    # 预计算常数
    sqrt_2_over_pi = 0.7978845608028654  # sqrt(2/π)
    
    # 流水线处理
    for i in U.Pipelined(iterations=BLOCKS_PER_SAMPLE):
        start_idx = i * TILE_LEN
        end_idx = start_idx + TILE_LEN
        
        # 加载数据
        U.data_copy(dst=input_tile, src=input_np[start_batch:start_batch+SAMPLES_PER_CORE, start_idx:end_idx], 
                    src_pos=U.GlobalMem, dst_pos=U.VecBuf)
        
        # 计算步骤 (每个操作单独一行，无嵌套)
        # 1. 计算x^3
        U.vbinary_op(op="mul", dst=x3_tile, src1=input_tile, src2=input_tile)
        U.vbinary_op(op="mul", dst=x3_tile, src1=x3_tile, src2=input_tile)
        
        # 2. 计算0.044715 * x^3
        U.vectorscalar_op(op="muls", dst=term1_tile, src=x3_tile, factor=0.044715)
        
        # 3. 计算inner = x + term1
        U.vbinary_op(op="add", dst=inner_tile, src1=input_tile, src2=term1_tile)
        
        # 4. 计算sqrt_2_over_pi * inner
        U.vectorscalar_op(op="muls", dst=term2_tile, src=inner_tile, factor=sqrt_2_over_pi)
        
        # 5. 计算tanh(term2)
        U.vunary_op(op="tanh", dst=tanh_term_tile, src=term2_tile)
        
        # 6. 计算1 + tanh_term
        U.vectorscalar_op(op="adds", dst=one_plus_tanh_tile, src=tanh_term_tile, factor=1.0)
        
        # 7. 计算0.5 * x
        U.vectorscalar_op(op="muls", dst=half_x_tile, src=input_tile, factor=0.5)
        
        # 8. 计算最终结果
        U.vbinary_op(op="mul", dst=output_tile, src1=half_x_tile, src2=one_plus_tanh_tile)
        
        # 写回结果
        U.data_copy(dst=output_np[start_batch:start_batch+SAMPLES_PER_CORE, start_idx:end_idx], src=output_tile,
                    src_pos=U.VecBuf, dst_pos=U.GlobalMem)


def gelu__op_impl_tiling(input_np, output_np, tiling):
    # 设置核数和workspace
    BLOCK_DIM = 8
    WORKSPACE_SIZE = 0