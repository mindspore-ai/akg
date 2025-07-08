import aul as U

def hardtanh_op_impl_npu(input: U.TensorPtr, output: U.TensorPtr, tiling: list):
    BLOCK_DIM = 8
    TOTAL_ELEMENTS = 16 * 16384
    ELEMENTS_PER_CORE = TOTAL_ELEMENTS // BLOCK_DIM
    
    core_idx = U.get_core_idx()
    start_idx = core_idx * ELEMENTS_PER_CORE
    end_idx = start_idx + ELEMENTS_PER_CORE
    
    input_tile = U.Tile(shape=(ELEMENTS_PER_CORE,), 
                      dtype=U.float16, 
                      pos=U.VecBuf)
    
    output_tile = U.Tile(shape=(ELEMENTS_PER_CORE,),
                       dtype=U.float16,
                       pos=U.VecBuf)
    
    for _ in U.Pipelined(iterations=1):
        U.data_copy(dst=input_tile, 
                  src=input[start_idx:end_idx],
                  src_pos=U.GlobalMem,
                  dst_pos=U.VecBuf)
        
        U.vectorscalar_op(op="maxs",
                        dst=output_tile,
                        src=input_tile,
                        factor=-1.0)
        
        U.vectorscalar_op(op="mins",
                        dst=output_tile,
                        src=output_tile,
                        factor=1.0)
        
        U.data_copy(dst=output[start_idx:end_idx],
                  src=output_tile,
                  src_pos=U.VecBuf,
                  dst_pos=U.GlobalMem)

def hardtanh_op_impl_tiling(input_np, output_np, tiling):
    BLOCK_DIM = 8
    WORKSPACE_SIZE = 0
    pass