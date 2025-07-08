def elu_op_impl(input: U.TensorPtr, output: U.TensorPtr):
    # Hardcoded tiling parameters
    BLOCK_DIM = 8
    TILE_LEN = 256
    LOOP_COUNT = 8
    BATCH_SIZE = 16
    
    core_idx = U.get_core_idx()
    dim_per_core = 2048  # 16384 / BLOCK_DIM
    
    # Create double buffered tiles
    input_tile = U.Tile(shape=(BATCH_SIZE, TILE_LEN), dtype=U.float16, pos=U.VecBuf)
    output_tile = U.Tile(shape=(BATCH_SIZE, TILE_LEN), dtype=U.float16, pos=U.VecBuf)
    
    # Intermediate tiles
    mask_tile = U.Tile(shape=(BATCH_SIZE, TILE_LEN), dtype=U.float16, pos=U.VecBuf)
    temp_tile = U.Tile(shape=(BATCH_SIZE, TILE_LEN), dtype=U.float16, pos=U.VecBuf)
    zero_tile = U.FilledTile((BATCH_SIZE, TILE_LEN), U.float16, U.VecBuf, value=0.0)
    one_tile = U.FilledTile((BATCH_SIZE, TILE_LEN), U.float16, U.VecBuf, value=1.0)
    inv_mask_tile = U.Tile(shape=(BATCH_SIZE, TILE_LEN), dtype=U.float16, pos=U.VecBuf)
    
    for i in U.Pipelined(iterations=LOOP_COUNT):
        current_dim = i * TILE_LEN
        
        # Load input with double buffering
        U.data_copy(dst=input_tile, 
                   src=input[:, core_idx*dim_per_core+current_dim:core_idx*dim_per_core+current_dim+TILE_LEN],
                   src_pos=U.GlobalMem, 
                   dst_pos=U.VecBuf)
        
        # Compute mask (input >= 0)
        U.vbinary_op(op="ge", dst=mask_tile, src1=input_tile, src2=zero_tile)
        
        # Compute inverse mask
        U.vbinary_op(op="sub", dst=inv_mask_tile, src1=one_tile, src2=mask_tile)
        
        # Compute exp(input) - 1
        U.vunary_op(op="exp", dst=temp_tile, src=input_tile)
        U.vectorscalar_op(op="adds", dst=temp_tile, src=temp_tile, factor=-1.0)
        
        # Apply alpha (1.0)
        U.vectorscalar_op(op="muls", dst=temp_tile, src=temp_tile, factor=1.0)
        
        # Calculate negative part
        U.vbinary_op(op="mul", dst=temp_tile, src1=temp_tile, src2=inv_mask_tile)
        
        # Calculate positive part
        U.vbinary_op(op="mul", dst=input_tile, src1=input_tile, src2=mask_tile)
        
        # Merge results
        U.vbinary_op(op="add", dst=output_tile, src1=input_tile, src2=temp_tile)
        
        # Store result with double buffering
        U.data_copy(dst=output[:, core_idx*dim_per_core+current_dim:core_idx*dim_per_core+current_dim+TILE_LEN],
                   src=output_tile,
                   src_pos=U.VecBuf,
                   dst_pos=U.GlobalMem)

def elu_op_impl_tiling(input_np, output_np, tiling):
    BLOCK_DIM = 8
    WORKSPACE_SIZE = 0