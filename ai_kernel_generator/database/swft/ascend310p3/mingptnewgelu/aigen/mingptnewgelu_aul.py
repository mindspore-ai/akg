def mingptnewgelu_op_impl_npu(input: U.TensorPtr, output: U.TensorPtr, tiling: list):
    BLOCK_DIM = 8
    TILE_ROWS = 250
    TILE_COLS = 2000
    core_idx = U.get_core_idx()
    
    total_rows = 2000
    rows_per_core = total_rows // BLOCK_DIM
    start_row = core_idx * rows_per_core
    end_row = start_row + rows_per_core
    
    input_tile = U.Tile(shape=(TILE_ROWS, TILE_COLS), dtype=U.float16, pos=U.VecBuf)
    output_tile = U.Tile(shape=(TILE_ROWS, TILE_COLS), dtype=U.float16, pos=U.VecBuf)
    
    factor_value = 0.7978845608
    cubic_factor = 0.044715
    
    cubic_tile = U.Tile(shape=(TILE_ROWS, TILE_COLS), dtype=U.float16, pos=U.VecBuf)
    sum_tile = U.Tile(shape=(TILE_ROWS, TILE_COLS), dtype=U.float16, pos=U.VecBuf)
    inner_tile = U.Tile(shape=(TILE_ROWS, TILE_COLS), dtype=U.float16, pos=U.VecBuf)
    
    for iter_idx in U.Pipelined(iterations=1):
        U.data_copy(dst=input_tile, src=input[start_row:end_row, :], 
                  src_pos=U.GlobalMem, dst_pos=U.VecBuf)
        
        U.vbinary_op(op="mul", dst=cubic_tile, src1=input_tile, src2=input_tile)
        U.vbinary_op(op="mul", dst=cubic_tile, src1=cubic_tile, src2=input_tile)
        U.vectorscalar_op(op="muls", dst=cubic_tile, src=cubic_tile, factor=cubic_factor)
        
        U.vbinary_op(op="add", dst=sum_tile, src1=input_tile, src2=cubic_tile)
        U.vectorscalar_op(op="muls", dst=inner_tile, src=sum_tile, factor=factor_value)
        
        tanh_tile = U.vunary_op(op="tanh", src=inner_tile)
        
        U.vectorscalar_op(op="adds", dst=sum_tile, src=tanh_tile, factor=1.0)
        U.vbinary_op(op="mul", dst=output_tile, src1=input_tile, src2=sum_tile)
        U.vectorscalar_op(op="muls", dst=output_tile, src=output_tile, factor=0.5)
        
        U.data_copy(dst=output[start_row:end_row, :], src=output_tile,
                  src_pos=U.VecBuf, dst_pos=U.GlobalMem)


def mingptnewgelu_op_impl_tiling(input_np, output_np, tiling):
    BLOCK_DIM = 8
    WORKSPACE_SIZE = 0