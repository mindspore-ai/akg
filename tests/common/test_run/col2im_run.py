import numpy as np
from akg.utils import kernel_exec as utils
from tests.common.tensorio import compare_tensor
from tests.common.test_op.col2im_compute import col2im_manual_schedule
from tests.common.gen_random import random_gaussian


def col2im_benchmark(data, kernel, pad, stride, output_H_W):

    N, C1, KH, KW, OH, OW, C0 = data.shape
    H, W = output_H_W
    stride_h, stride_w = stride
    pad_t, pad_b, pad_l, pad_r = pad

    expect_pad_shape = (N, C1, H + pad_b + pad_t, W + pad_r + pad_l, C0)
    expect_pad = np.zeros(expect_pad_shape, dtype=data.dtype)

    for n in range(N):
        for c1 in range(C1):
            for kh in range(KH):
                for kw in range(KW):
                    for ho in range(OH):
                        for wo in range(OW):
                            for c0 in range(C0):
                                expect_pad[n, c1, ho * stride_h + kh, wo * stride_w + kw, c0] += data[
                                    n, c1, kh, kw, ho, wo, c0
                                ]

    return expect_pad[:, :, pad_t: (pad_t + H), pad_l: (pad_l + W), :]


def col2im_run(shape, kernel, stride, pad, output_H_W, dtype, polyhedral=False, attrs=None):
    expect, data, res = gen_data(dtype, kernel, pad, shape, stride, output_H_W)

    if polyhedral:
        raise Exception("ERROR: no DSL with poly support for col2im, please select manual schedule version")
    else:
        mod = col2im_manual_schedule(shape, kernel, stride, pad, dtype, output_H_W, attrs=attrs, polyhedral=polyhedral)
    output = utils.mod_launch(mod, [data, res])

    return data, output, expect, compare_tensor(output, expect, rtol=5e-03, equal_nan=True)


def gen_data(dtype, kernel, pad, shape, stride, output_H_W):

    N, C1, KH, KW, OH, OW, C0 = shape
    H, W = output_H_W
    pad_t, pad_b, pad_l, pad_r = pad
    stride_h, stride_w = stride
    kernel_h, kernel_w = kernel

    assert H == (OH - 1) * stride_h + kernel_h - (pad_t + pad_b), "Height of input and output do not match"
    assert W == (OW - 1) * stride_w + kernel_w - (pad_l + pad_r), "Width of input and output do not match"

    original_shape = (N, C1, H, W, C0)
    original_shape_pad = (N, C1, H + pad_t + pad_b, W + pad_l + pad_r, C0)

    input_for_im2col = np.zeros(original_shape_pad, dtype=dtype)
    input_for_im2col[:, :, pad_t: (pad_t + H), pad_l: (pad_l + W), :] = random_gaussian(
        original_shape, miu=1, sigma=0.1
    ).astype(dtype)

    data = np.zeros(shape, dtype=dtype)

    for n in range(N):
        for c1 in range(C1):
            for kh in range(KH):
                for kw in range(KW):
                    for ho in range(OH):
                        for wo in range(OW):
                            for c0 in range(C0):
                                data[n, c1, kh, kw, ho, wo, c0] = input_for_im2col[
                                    n, c1, ho * stride_h + kh, wo * stride_w + kw, c0
                                ]

    expect = col2im_benchmark(data, kernel, pad, stride, output_H_W).astype(dtype)
    res = np.full(original_shape, np.nan, dtype)
    return expect, data, res
