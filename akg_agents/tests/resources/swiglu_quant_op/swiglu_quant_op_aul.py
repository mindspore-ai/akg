def swiglu_quant_op_impl_npu(x: U.TensorPtr, smooth_scales: U.TensorPtr,
                             output: U.TensorPtr, scale: U.TensorPtr, swiglu_out: U.TensorPtr):
    BLOCK_DIM = 8
    M, N = 1024, 3584

    x0 = U.Tile(shape=(1, N), dtype=U.float16, pos=U.VecBuf)
    x1 = U.Tile(shape=(1, N), dtype=U.float16, pos=U.VecBuf)
    x0_f32 = U.Tile(shape=(1, N), dtype=U.float32, pos=U.VecBuf)
    x1_f32 = U.Tile(shape=(1, N), dtype=U.float32, pos=U.VecBuf)
    sigmoid_t = U.Tile(shape=(1, N), dtype=U.float32, pos=U.VecBuf)
    swiglu_t = U.Tile(shape=(1, N), dtype=U.float32, pos=U.VecBuf)
    swiglu_t_fp16 = U.Tile(shape=(1, N), dtype=U.float16, pos=U.VecBuf)
    q_tile = U.Tile(shape=(1, N), dtype=U.float32, pos=U.VecBuf)
    abs_tile = U.Tile(shape=(1, N), dtype=U.float32, pos=U.VecBuf)
    max_tile = U.Tile(shape=(1, 1), dtype=U.float32, pos=U.VecBuf)

    smooth_tile_f16 = U.Tile(shape=(N,), dtype=U.float16, pos=U.VecBuf)
    smooth_tile_f32 = U.Tile(shape=(N,), dtype=U.float32, pos=U.VecBuf)

    core_idx = U.get_core_idx()
    per_core_rows = M // BLOCK_DIM

    U.data_copy(dst=smooth_tile_f16, src=smooth_scales, src_pos=U.GlobalMem, dst_pos=U.VecBuf)
    U.vunary_op(op="cast_fp16_to_fp32", dst=smooth_tile_f32, src=smooth_tile_f16)

    for iter_idx in U.Pipelined(iterations=per_core_rows):
        row_start = core_idx * per_core_rows + iter_idx

        U.data_copy(dst=x0, src=x[row_start:row_start + 1, :N], src_pos=U.GlobalMem, dst_pos=U.VecBuf)
        U.data_copy(dst=x1, src=x[row_start:row_start + 1, N:2 * N], src_pos=U.GlobalMem, dst_pos=U.VecBuf)
        U.vunary_op(op="cast_fp16_to_fp32", dst=x0_f32, src=x0)
        U.vunary_op(op="cast_fp16_to_fp32", dst=x1_f32, src=x1)
        # sigmoid
        U.vectorscalar_op(op="muls", dst=sigmoid_t, src=x0_f32, factor=-1.0)
        U.vunary_op(op="exp", dst=sigmoid_t, src=sigmoid_t)
        U.vectorscalar_op(op="adds", dst=sigmoid_t, src=sigmoid_t, factor=1.0)
        U.vbinary_op(op="div", dst=sigmoid_t, src1=x0_f32, src2=sigmoid_t)
        # swiglu
        U.vbinary_op(op="mul", dst=swiglu_t, src1=sigmoid_t, src2=x1_f32)

        U.data_copy(dst=q_tile, src=swiglu_t, src_pos=U.VecBuf, dst_pos=U.VecBuf)
        # copy to swiglu_out
        U.vunary_op(op="cast_f32_to_f16", dst=swiglu_t_fp16, src=swiglu_t)
        U.data_copy(dst=swiglu_out[row_start:row_start + 1, 0:N],
                    src=swiglu_t_fp16, src_pos=U.VecBuf, dst_pos=U.GlobalMem)
        # mul smooth_scale
        U.vbinary_op(op="mul", dst=q_tile, src1=q_tile, src2=smooth_tile_f32)
        # copy to scale
        U.vunary_op(op="abs", dst=abs_tile, src=q_tile)
        U.vreduce_op(op="max", dst=max_tile, src=abs_tile, axis=-1)
        U.vectorscalar_op(op="divs", dst=max_tile, src=max_tile, factor=127.0)
        U.data_copy(dst=scale[row_start:row_start + 1], src=max_tile, src_pos=U.VecBuf, dst_pos=U.GlobalMem)
        # copy to output
        U.vbinary_op(op="div", dst=q_tile, src1=q_tile, src2=max_tile)
        U.vunary_op(op="round_fp32_to_int8", dst=q_tile, src=q_tile)
        U.data_copy(dst=output[row_start:row_start + 1, 0:N, ], src=q_tile, src_pos=U.VecBuf, dst_pos=U.GlobalMem)


def swiglu_quant_op_tiling(input_np, smooth_scales_np, output_np, scale_np, swiglu_out_np, tiling):
    BLOCK_DIM = 8
    WORKSPACE_SIZE = 0
