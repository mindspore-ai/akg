def swiglu_op(input: U.TensorPtr, output: U.TensorPtr):
    dim0 = 40
    dim1 = 256
    dim0_split = 5
    dim1_split = 128

    input_half = U.Tile((dim0_split, dim1), U.float16, U.VecBuf)
    input0_float = U.Tile((dim0_split, dim1_split), U.float32, U.VecBuf)
    input1_float = U.Tile((dim0_split, dim1_split), U.float32, U.VecBuf)
    neg_tile = U.Tile((dim0_split, dim1_split), U.float32, U.VecBuf)
    exp_tile = U.Tile((dim0_split, dim1_split), U.float32, U.VecBuf)
    sigmoid_tile = U.Tile((dim0_split, dim1_split), U.float32, U.VecBuf)
    result_tile = U.Tile((dim0_split, dim1_split), U.float32, U.VecBuf)
    output_half = U.Tile((dim0_split, dim1_split), U.float16, U.VecBuf)

    core_idx = get_core_idx()

    start_offset = core_idx * dim0_split
    end_offset = (core_idx + 1) * dim0_split
    U.data_copy(dst=input_half, src=input[start_offset:end_offset, 0:dim1], src_pos=U.GlobalMem, dst_pos=U.VecBuf)

    U.vunary_op(op="cast_fp16_to_fp32", dst=input0_float, src=input_half[0:dim1_split])
    U.vunary_op(op="cast_fp16_to_fp32", dst=input1_float, src=input_half[dim1_split:dim1])

    U.vunary_op(op="mul", dst=neg_tile, src1=input0_float, src2=-1.0)
    U.vunary_op(op="exp", dst=exp_tile, src=neg_tile)
    U.vbinary_op(op="add", dst=exp_tile, src1=exp_tile, src2=1.0)
    U.vbinary_op(op="div", dst=sigmoid_tile, src1=input0_float, src2=exp_tile)
    U.vbinary_op(op="mul", dst=result_tile, src1=sigmoid_tile, src2=input1_float)

    U.vunary_op(op="cast_fp32_to_fp16", dst=output_half, src=result_tile)
    U.data_copy(dst=output[start_offset:end_offset, 0:dim1_split],
                src=output_half, src_pos=U.VecBuf, dst_pos=U.GlobalMem)
