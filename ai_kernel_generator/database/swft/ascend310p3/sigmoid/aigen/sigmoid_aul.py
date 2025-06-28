import aul as U

def sigmoid_op_impl_npu(input_np: U.TensorPtr, output_np: U.TensorPtr, tiling: list):
    # 硬编码参数（来自Tiling函数和任务定义）
    BATCH_SIZE = 16
    DIM = 16384
    BLOCK_DIM = 8
    SAMPLES_PER_CORE = BATCH_SIZE // BLOCK_DIM  # 每个核处理2个样本
    TILE_DIM = 8192  # 列方向分块大小
    LOOP_COUNT = DIM // TILE_DIM  # 循环次数

    # 获取当前核ID
    core_idx = U.get_core_idx()
    start_batch = core_idx * SAMPLES_PER_CORE
    end_batch = start_batch + SAMPLES_PER_CORE

    # 创建Tile（输入和输出）
    input_tile = U.Tile(shape=(SAMPLES_PER_CORE, TILE_DIM), dtype=U.float16, pos=U.VecBuf)
    output_tile = U.Tile(shape=(SAMPLES_PER_CORE, TILE_DIM), dtype=U.float16, pos=U.VecBuf)
    
    # 创建中间计算Tile
    neg_tile = U.Tile(shape=(SAMPLES_PER_CORE, TILE_DIM), dtype=U.float16, pos=U.VecBuf)
    exp_tile = U.Tile(shape=(SAMPLES_PER_CORE, TILE_DIM), dtype=U.float16, pos=U.VecBuf)
    add_tile = U.Tile(shape=(SAMPLES_PER_CORE, TILE_DIM), dtype=U.float16, pos=U.VecBuf)

    # 流水线处理
    for i in U.Pipelined(iterations=LOOP_COUNT):
        start_col = i * TILE_DIM
        end_col = start_col + TILE_DIM
        
        # 加载数据到向量缓存
        U.data_copy(dst=input_tile, 
                   src=input_np[start_batch:end_batch, start_col:end_col],
                   src_pos=U.GlobalMem, dst_pos=U.VecBuf)
        
        # 计算sigmoid: 1/(1+exp(-x))
        U.vectorscalar_op(op="muls", dst=neg_tile, src=input_tile, factor=-1.0)  # -x
        U.vunary_op(op="exp", dst=exp_tile, src=neg_tile)  # exp(-x)
        U.vectorscalar_op(op="adds", dst=add_tile, src=exp_tile, factor=1.0)  # 1 + exp(-x)
        U.vunary_op(op="rec", dst=output_tile, src=add_tile)  # 1/(1+exp(-x))
        
        # 存储结果到全局内存
        U.data_copy(dst=output_np[start_batch:end_batch, start_col:end_col],
                   src=output_tile,
                   src_pos=U.VecBuf, dst_pos=U.GlobalMem)

def sigmoid_op_impl_tiling(input_np, output_np, tiling):
    BLOCK_DIM = 8
    WORKSPACE_SIZE = 0