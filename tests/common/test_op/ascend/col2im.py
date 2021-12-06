import akg.tvm


def intrin_col2im(input_shape, output_shape, kernel, stride, pad, dtype):
    '''
    Compute col2im via cce col2im intrin function call directly

    Args:
        input_shape: the shape of the image
        output_shape: the shape of the result of im2col given the input image
        kernel: kernel sizes for im2col
        stride: stride sizes for im2col
        pad: padding sizes for im2col, including padding top, bottom, left, and right
        dtype: type of the data

    Return:
        cce intrin function call for col2im
    '''

    _, _, _, _, WINDOW_H, WINDOW_W, _ = input_shape
    _, _, H, W, _ = output_shape
    kernel_h, kernel_w = kernel
    stride_h, stride_w = stride
    pad_t, pad_b, pad_l, pad_r = pad

    assert (WINDOW_H * WINDOW_W) % 16 == 0, "Number of windows over the input must be divisible by 16 (col2im repeat)"
    assert (H * W * 16) % 64 == 0, "Input size must be divisible by 64 (vector_dup repeat)"

    # FCOL2IMG -------------------------------------------
    INPUT_W = W
    INPUT_H = H

    PAD_LEFT = pad_l
    PAD_RIGHT = pad_r
    PAD_TOP = pad_t
    PAD_BOTTOM = pad_b
    # ---------------------------------------------------
    # Xm ------------------------------------------------
    W_IDX_KERNEL = 0
    H_IDX_KERNEL = 0

    H_IDX = (-pad_l) & 0xFFFF  # fix negative numbers
    W_IDX = (-pad_t) & 0xFFFF

    C1_IDX = 0
    # ---------------------------------------------------
    # Xt ------------------------------------------------
    STRIDE_H = stride_h
    STRIDE_W = stride_w

    KERNEL_H = kernel_h
    KERNEL_W = kernel_w

    DILATION_H = 1
    DILATION_W = 1

    JUMP_OFFSET = 0
    REPEAT_MODE = 1
    REPEAT_TIME = (WINDOW_H * WINDOW_W) // 16
    # ---------------------------------------------------

    INPUT_B = 1
    INPUT_C1 = 1
    INPUT_C0 = 16

    input_data = akg.tvm.placeholder((INPUT_B, INPUT_C1, KERNEL_H, KERNEL_W, WINDOW_H, WINDOW_W, INPUT_C0), dtype=dtype)

    result = akg.tvm.compute(
        (INPUT_B, INPUT_C1, INPUT_H, INPUT_W, INPUT_C0),
        lambda b, c1, h, w, c0: input_data[b, c1, h % KERNEL_H, w % KERNEL_W, h % WINDOW_H, w % WINDOW_W, c0],
        name="col2im_intrinsic",
    )

    input_data_buff = akg.tvm.decl_buffer(
        input_data.shape, input_data.dtype, name="input_data_buff", offset_factor=1, scope="local.UB"
    )

    result_buff = akg.tvm.decl_buffer(result.shape, result.dtype, name="result_buff", offset_factor=1, scope="local.UB")

    def pack_args(sp):
        assert len(sp) == 20
        fcol2img = (
            akg.akg.tvm.const(sp[0], "uint64")
            + akg.akg.tvm.const(sp[1] * 2 ** 16, "uint64")
            + akg.akg.tvm.const(sp[2] * 2 ** 32, "uint64")
            + akg.akg.tvm.const(sp[3] * 2 ** 40, "uint64")
            + akg.akg.tvm.const(sp[4] * 2 ** 48, "uint64")
            + akg.akg.tvm.const(sp[5] * 2 ** 56, "uint64")
        )

        Xm = (
            akg.akg.tvm.const(sp[6] * 2 ** 16, "uint64")
            + akg.akg.tvm.const(sp[7] * 2 ** 24, "uint64")
            + akg.akg.tvm.const(sp[8] * 2 ** 32, "uint64")
            + akg.akg.tvm.const(sp[9] * 2 ** 48, "uint64")
            + akg.akg.tvm.const(sp[10], "uint64")
        )

        Xt = (
            akg.akg.tvm.const(sp[11], "uint64")
            + akg.akg.tvm.const(sp[12] * 2 ** 6, "uint64")
            + akg.akg.tvm.const(sp[13] * 2 ** 12, "uint64")
            + akg.akg.tvm.const(sp[14] * 2 ** 20, "uint64")
            + akg.akg.tvm.const(sp[15] * 2 ** 28, "uint64")
            + akg.akg.tvm.const(sp[16] * 2 ** 36, "uint64")
            + akg.akg.tvm.const(sp[17] * 2 ** 44, "uint64")
            + akg.akg.tvm.const(sp[18] * 2 ** 52, "uint64")
            + akg.akg.tvm.const(sp[19] * 2 ** 56, "uint64")
        )

        return (fcol2img, Xm, Xt)

    def intrin_func(ins, outs):
        sp = [
            INPUT_W, INPUT_H, PAD_LEFT, PAD_RIGHT, PAD_TOP, PAD_BOTTOM,  # FMATRIX
            W_IDX_KERNEL, H_IDX_KERNEL, W_IDX, H_IDX, C1_IDX,  # Xm
            STRIDE_W, STRIDE_H, KERNEL_W, KERNEL_H, DILATION_W, DILATION_H, JUMP_OFFSET, REPEAT_MODE, REPEAT_TIME,  # Xt
        ]
        aa = ins[0]
        bb = outs[0]
        ib = akg.tvm.ir_builder.create()
        fcol2img, Xm, Xt = pack_args(sp)
        ib.emit(akg.tvm.call_extern(dtype, "set_fcol2img", fcol2img))
        ib.emit(akg.tvm.call_extern(dtype, "vector_dup", bb.access_ptr("w"), 0, (INPUT_H * INPUT_W * 16) // 64, 1, 1, 8, 8))
        c = 0
        for kh in range(KERNEL_H):
            for kw in range(KERNEL_W):
                sp[6] = kw
                sp[7] = kh
                fcol2img, Xm, Xt = pack_args(sp)
                ib.emit(
                    akg.tvm.call_extern(
                        dtype,
                        "col2img",
                        bb.access_ptr("rw"),
                        aa.access_ptr("r", offset=c * 16 * INPUT_C0 * REPEAT_TIME),
                        Xm,
                        Xt,
                    )
                )
                c += 1
        return ib.get()

    with akg.tvm.build_config(offset_factor=1):
        return akg.tvm.decl_tensor_intrin(result.op, intrin_func, binds={input_data: input_data_buff, result: result_buff})
