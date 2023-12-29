# Copyright 2019-2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import akg
import math
import akg.tvm
import akg.utils as utils
from akg.utils import custom_tiling as ct_util

conv_dtype = 'float16'
block_size = 16


def manual_im2col_1repeat(indices, A, fp_w, fp_h, fm_w, fm_h, fp_c1, pad_t, pad_b, l1_h_fmatrix, stride_h, stride_w,
                          kernel_h, kernel_w):
    window_index, c0_index = indices

    kh_index = fp_h
    kw_index = fp_w

    # just randomly use pad_t and pad_b to make sure they're used
    h_index = fm_h + kh_index + pad_t + pad_b + l1_h_fmatrix
    w_index = fm_w + window_index * stride_w + kw_index

    # A always has dimension (1, 1, l1_h, l1_w, 16)
    return A[0, 0, h_index, w_index, c0_index]


def intrin_load_im2col(A_shape, strides, kernel_size, padding):
    N, C1, H, W, C0 = A_shape
    stride_h, stride_w = strides
    kernel_h, kernel_w = kernel_size
    pad_t, pad_b, pad_l, pad_r = padding
    # compute output Ho and Wo
    Ho = (H + pad_t + pad_b - kernel_h) // (stride_h) + 1
    Wo = (W + pad_l + pad_r - kernel_w) // (stride_w) + 1

    l1_h = akg.tvm.var("l1_h", dtype='int32')
    l1_w = akg.tvm.var("l1_w", dtype='int32')
    # we know that the n-batch and C1 are fixed. The H and W of the piece of A are unknown.
    a = akg.tvm.placeholder((1, 1, l1_h, l1_w, C0), dtype=conv_dtype)
    fp_w = akg.tvm.var("fp_w")
    fp_h = akg.tvm.var("fp_h")
    fm_w = akg.tvm.var("fm_w")
    fm_h = akg.tvm.var("fm_h")
    fp_c1 = akg.tvm.var("fp_c1")
    pad_t = akg.tvm.var("pad_t")
    pad_b = akg.tvm.var("pad_b")
    l1_h_fmatrix = akg.tvm.var("l1_h_fmatrix")

    # Output will be of shape (block_size (window positions), C0)  = (16x16)
    c = akg.tvm.compute((block_size, C0),
                        lambda *indices: manual_im2col_1repeat(indices, a, fp_w, fp_h, fm_w, fm_h, fp_c1, pad_t, pad_b,
                                                               l1_h_fmatrix, stride_h, stride_w, kernel_h, kernel_w),
                        name='im2col_manual')

    # Ab_scope = "local.L1"
    # Cb_scope = "local.L0A"
    Ab_scope = ""
    Cb_scope = ""

    Ab = akg.tvm.decl_buffer(a.shape, a.dtype,
                             name="Abuf",
                             offset_factor=1,
                             scope=Ab_scope)  # , strides=[akg.tvm.var("s1"), akg.tvm.var("s2"), akg.tvm.var("s3"), akg.tvm.var("s4"), akg.tvm.var("s5")])

    Cb = akg.tvm.decl_buffer(c.shape, c.dtype, name="Cbuf", offset_factor=1, scope=Cb_scope)

    mode = 0
    repeat = 1
    if has_pad(padding) and not large_fmap(A_shape):
        mode = 1
        repeat = Ho.value * Wo.value // 16
        if (Ho.value * Wo.value) % 16 > 0:
            repeat = repeat + 1
    elif large_fmap(A_shape):
        mode = 1
        h_cut = 2
        repeat = h_cut * Wo.value // 16

    def intrin_func(ins, outs, sp):
        aa = ins[0]
        dd = outs[0]

        def _body():
            ib = akg.tvm.ir_builder.create()
            attrs = {"pragma_out_h": Ho, "pragma_out_w": Wo}
            ib.scope_attr(attrs, "im2colKey", "im2colValue")
            ib.emit(akg.tvm.call_extern("int32", "cce_img2col_ub",
                                        dd.access_ptr("w"),
                                        aa.access_ptr("r"),
                                        # the constant params are dilation, jump offset, repeat-mode, # repeats, c0 mode
                                        sp[0], sp[1], sp[2], sp[3], sp[4], stride_w, stride_h, kernel_w, kernel_h, 1, 1,
                                        1, mode, repeat, 0, sp[5], sp[6], pad_l, pad_r, l1_h, l1_w))

            return ib.get()

        return _body()

    with akg.build_config(offset_factor=1):
        return akg.tvm.decl_tensor_intrin(c.op, intrin_func, binds={a: Ab, c: Cb},
                                          scalar_params=[fp_w, fp_h, fm_w, fm_h, fp_c1, pad_t, pad_b, l1_h_fmatrix])


def im2col_fractal(A_im2col_shape, A, kernel_h, kernel_w, stride, padding):
    load_im2col = intrin_load_im2col(A.shape, stride, [kernel_h, kernel_w], padding)

    N, C1, H, W, C0 = A.shape
    stride_h, stride_w = stride
    pad_t, pad_b, pad_l, pad_r = padding
    Wo = (W + pad_l + pad_r - kernel_w) // stride_w + 1
    if len(A_im2col_shape) == 5:
        n, windowBatches, ComboSize, windowsPerBatch, channelsPerBatch = A_im2col_shape
        if has_pad(padding) and not large_tensor(A):
            # assume we need the whole width of A
            # choose a section of the rows of A that encompasses all of the windows in the current window-batch
            compute = akg.tvm.compute(A_im2col_shape, lambda i, j, k: load_im2col(A[0, 0, 0:H, 0:W, 0:C0],
                                                                                  # fetch position w, fetch position h
                                                                                  (k % kernel_w),
                                                                                  ((k // kernel_w) % kernel_h),
                                                                                  # top-left corner w,
                                                                                  (((
                                                                                                j * windowsPerBatch) % Wo) * stride_w) - pad_l,
                                                                                  # top-left corner h, (when this is < 0, the slice will start from row 0 so there is no redundancy
                                                                                  # between base address and this param)
                                                                                  ((
                                                                                               j * windowsPerBatch) // Wo) * stride_h - pad_t,
                                                                                  # initial C1
                                                                                  (k // kernel_w) // kernel_h,
                                                                                  # top padding
                                                                                  pad_t,
                                                                                  # bottom padding
                                                                                  pad_b,
                                                                                  kernel_h - akg.tvm.max(((((((
                                                                                                                          j * windowsPerBatch) // Wo) * stride_h) - pad_t) + kernel_h) - H),
                                                                                                         0),
                                                                                  name='im2col_fractal'
                                                                                  ))
        else:
            if large_tensor(A):
                compute = akg.tvm.compute(A_im2col_shape, lambda i, j, k: load_im2col(A[0, 0, 0:H, 0:W, 0:C0],
                                                                                      # fetch position w, fetch position h
                                                                                      (k % kernel_w),
                                                                                      ((k // kernel_w) % kernel_h),
                                                                                      # top-left corner w,
                                                                                      (((
                                                                                                    j * windowsPerBatch) % Wo) * stride_w) - pad_l,
                                                                                      ((
                                                                                                   j * windowsPerBatch) // Wo) * stride_h - pad_t,
                                                                                      # initial C1
                                                                                      (k // kernel_w) // kernel_h,
                                                                                      pad_t, pad_b,
                                                                                      # num rows in L1 for fmatrix is discounted by the amount of bottom padding
                                                                                      kernel_h - akg.tvm.max(((((((
                                                                                                                              j * windowsPerBatch) // Wo) * stride_h) - pad_t) + kernel_h) - H),
                                                                                                             0),
                                                                                      name='im2col_fractal'
                                                                                      ))
            else:
                # assume we need the whole width of A
                # choose a section of the rows of A that encompasses all of the windows in the current window-batch
                compute = akg.tvm.compute(A_im2col_shape, lambda i, j, k: load_im2col(A[0, 0,
                                                                                      # A[i, (k // kernel_w) // kernel_h,
                                                                                      # assume padding < kernel size
                                                                                      0:H,
                                                                                      # tvm.max((((j*windowsPerBatch)//Wo)*stride_h)-pad_t, 0):
                                                                                      # tvm.min(H, ((((j*windowsPerBatch)//Wo)*stride_h)-pad_t)+kernel_h),
                                                                                      0:W, 0:C0],
                                                                                      # fetch position w, fetch position h
                                                                                      (k % kernel_w),
                                                                                      ((k // kernel_w) % kernel_h),
                                                                                      # top-left corner w,
                                                                                      (((
                                                                                                    j * windowsPerBatch) % Wo) * stride_w) - pad_l,
                                                                                      # top-left corner h, (when this is < 0, the slice will start from row 0 so there is no redundancy
                                                                                      # between base address and this param)
                                                                                      # tvm.min(((j*windowsPerBatch)//Wo)*stride_h - pad_t, 0),
                                                                                      ((
                                                                                                   j * windowsPerBatch) // Wo) * stride_h - pad_t,
                                                                                      # initial C1
                                                                                      (k // kernel_w) // kernel_h,
                                                                                      # top padding
                                                                                      akg.tvm.max(pad_t - (((
                                                                                                                        j * windowsPerBatch) // Wo) * stride_h),
                                                                                                  0),
                                                                                      # bottom padding
                                                                                      akg.tvm.max(((((((
                                                                                                                   j * windowsPerBatch) // Wo) * stride_h) - pad_t) + kernel_h) - H),
                                                                                                  0),
                                                                                      # num rows in L1 for fmatrix is discounted by the amount of bottom padding
                                                                                      kernel_h - akg.tvm.max(((((((
                                                                                                                              j * windowsPerBatch) // Wo) * stride_h) - pad_t) + kernel_h) - H),
                                                                                                             0),
                                                                                      name='im2col_fractal'
                                                                                      ))
    elif len(A_im2col_shape) == 4:
        windowBatches, ComboSize, windowsPerBatch, channelsPerBatch = A_im2col_shape
        if has_pad(padding):
            compute = akg.tvm.compute(A_im2col_shape, lambda j, k: load_im2col(A[0, 0,
                                                                               0:H,
                                                                               0:W, 0:C0],
                                                                               # fetch position w, fetch position h
                                                                               (k % kernel_w),
                                                                               ((k // kernel_w) % kernel_h),
                                                                               # top-left corner w,
                                                                               -pad_l,
                                                                               # top-left corner h, (when this is < 0, the slice will start from row 0 so there is no redundancy
                                                                               # between base address and this param)
                                                                               - pad_t,
                                                                               # initial C1
                                                                               (k // kernel_w) // kernel_h,
                                                                               # top padding
                                                                               pad_t,
                                                                               # bottom padding
                                                                               pad_b,
                                                                               # num rows in L1 for fmatrix is discounted by the amount of bottom padding
                                                                               kernel_h - akg.tvm.max(((((((
                                                                                                                       j * windowsPerBatch) // Wo) * stride_h) - pad_t) + kernel_h) - H),
                                                                                                      0),
                                                                               name='im2col_fractal'
                                                                               ))
        else:
            compute = akg.tvm.compute(A_im2col_shape, lambda j, k: load_im2col(A[0, 0,
                                                                               0:H,
                                                                               0:W, 0:C0],
                                                                               # fetch position w, fetch position h
                                                                               (k % kernel_w),
                                                                               ((k // kernel_w) % kernel_h),
                                                                               # top-left corner w,
                                                                               (((
                                                                                             j * windowsPerBatch) % Wo) * stride_w) - pad_l,
                                                                               # top-left corner h, (when this is < 0, the slice will start from row 0 so there is no redundancy
                                                                               # between base address and this param)
                                                                               ((
                                                                                            j * windowsPerBatch) // Wo) * stride_h - pad_t,
                                                                               # initial C1
                                                                               (k // kernel_w) // kernel_h,
                                                                               # top padding
                                                                               akg.tvm.max(pad_t - (((
                                                                                                                 j * windowsPerBatch) // Wo) * stride_h),
                                                                                           0),
                                                                               # bottom padding
                                                                               # akg.tvm.max(((((((j*windowsPerBatch)//Wo)*stride_h)-pad_t)+kernel_h) - H), 0),
                                                                               0,
                                                                               # num rows in L1 for fmatrix is discounted by the amount of bottom padding
                                                                               kernel_h - akg.tvm.max(((((((
                                                                                                                       j * windowsPerBatch) // Wo) * stride_h) - pad_t) + kernel_h) - H),
                                                                                                      0),
                                                                               name='im2col_fractal'
                                                                               ))

    return compute


def load_im2col_dsl_no_padding(placeholders, input_shape, kernel, padding, stride):
    # A is input
    A, = placeholders

    batch_size, C1, Hi, Wi, C0 = input_shape
    kernel_h, kernel_w = kernel
    padding_top, padding_bottom, padding_left, padding_right = padding
    stride_h, stride_w = stride

    # compute output Ho and Wo
    Ho = (Hi + padding_top + padding_bottom - kernel_h) // (stride_h) + 1
    Wo = (Wi + padding_left + padding_right - kernel_w) // (stride_w) + 1

    M_size = batch_size * Ho * Wo
    assert M_size % block_size != 0, "M size must be the multiple of block_size"
    HoWo_mad = M_size // block_size

    # im2col
    # small-z-big-Z
    A_im2col_fractal_shape = (HoWo_mad,
                              C1 * kernel_h * kernel_w,  # * (C0 // block_size) = * 1,
                              block_size,
                              block_size)

    A_im2col_fractal_res = im2col_fractal(A_im2col_fractal_shape, A, kernel_h, kernel_w, stride, padding)
    return A_im2col_fractal_res


def load_im2col_dsl(placeholders, input_shape, kernel, padding, stride):
    # A is input
    A, = placeholders

    batch_size, C1, Hi, Wi, C0 = input_shape
    kernel_h, kernel_w = kernel
    padding_top, padding_bottom, padding_left, padding_right = padding
    stride_h, stride_w = stride

    # compute output Ho and Wo
    Ho = (Hi + padding_top + padding_bottom - kernel_h) // (stride_h) + 1
    Wo = (Wi + padding_left + padding_right - kernel_w) // (stride_w) + 1

    # im2col
    # small-z-big-Z
    HoWo_mad = (Ho * Wo + block_size - 1) // block_size * block_size
    A_im2col_fractal_shape = (batch_size,
                              HoWo_mad // block_size,
                              C1 * kernel_h * kernel_w,  # * (C0 // block_size) = * 1,
                              block_size,
                              block_size)

    A_im2col_fractal_res = im2col_fractal(A_im2col_fractal_shape, A, kernel_h, kernel_w, stride, padding)

    return A_im2col_fractal_res


load_im2col_set_dim_map = {
    # 2D
    str(((32, 1, 224, 224, 16), (7, 7), (2, 2), (3, 3, 3, 3))): ((1, 0), (14, 0), (49, 0)),
    str(((32, 1, 224, 224, 16), (7, 7), (2, 2), (2, 3, 2, 3))): ((1, 0), (14, 0), (49, 0)),
}


def has_pad(pad):
    if pad[0] > 0 or pad[1] > 0 or pad[2] > 0 or pad[3] > 0:
        return True
    return False


def large_tensor(tensor):
    return large_fmap(tensor.shape)


def large_fmap(fmap):
    fmap_n, fmap_c1, fmap_h, fmap_w, fmap_c0 = fmap
    map_size = fmap_h.value * fmap_w.value * fmap_c0.value * 2
    if map_size > math.pow(2, 20):
        return True
    return False


def get_attrs():
    """
    get attrs config
    """
    attr_map = dict()
    attr_map["loop_partition_unroll"] = False
    attr_map["enable_multicore"] = False
    attr_map["coarsenImg2Col"] = True
    attr_map["enable_double_buffer"] = False
    attr_map["pragma_sink_last_axis"] = False
    attr_map["enable_hoist_insn"] = False
    attr_map["pragma_enable_reschedule"] = False
    return attr_map


def load_im2col_set_dim(tensor_fmap, kernel, stride, pad):
    fmap_n, fmap_c1, fmap_h, fmap_w, fmap_c0 = tensor_fmap.shape
    filter_h, filter_w = kernel
    # calculate the tiling factor.
    ho = (fmap_h + pad[0] + pad[1] - filter_h) // (stride[0]) + 1
    wo = (fmap_w + pad[2] + pad[3] - filter_w) // (stride[1]) + 1
    mode = True
    if (ho.value * wo.value) % 16 == 0:
        h_cut = (ho * wo) // 16
        if has_pad(pad):
            mode = False
    else:
        h_cut = (fmap_n * ho * wo) // 16
        mode = False
    co_cut = filter_h * filter_w

    key = ((fmap_n, fmap_c1, fmap_h, fmap_w, fmap_c0), tuple(kernel), tuple(stride), tuple(pad))

    set_dims = ct_util.set_dims_by_key(str(key), load_im2col_set_dim_map)
    if set_dims == '':
        dims = ()
        if mode and fmap_n.value > 1:
            dims += ((1, 0),)
        dims += ((h_cut, 0), (co_cut, 0))
        return ct_util.set_dims(dims), str(key)
    return set_dims, str(key)


@ct_util.reg_set_dim_func(load_im2col_set_dim)
def LoadIm2col(tensor_fmap, kernel, stride, pad, target=utils.CCE):
    """
    Supported Platforms:
        'Ascend'
    """
    # adjust to TilingApi
    # feature map (NCHW -> NC1HWC0)
    fmap_n, fmap_c1, fmap_h, fmap_w, fmap_c0 = tensor_fmap.shape
    fmap_shape_NC1HWCO = (fmap_n, fmap_c1, fmap_h, fmap_w, fmap_c0)

    # stride (stride_h, stride_w)
    filter_h, filter_w = kernel

    # fmap_placeholder (NC1HWCO)
    load_im2col_dsl_input = (tensor_fmap,)

    load_im2col_dsl_output = load_im2col_dsl(load_im2col_dsl_input, fmap_shape_NC1HWCO, kernel, pad, stride)

    # calculate the tiling factor.
    ho = (fmap_h + pad[0] + pad[1] - filter_h) // (stride[0]) + 1
    wo = (fmap_w + pad[2] + pad[3] - filter_w) // (stride[1]) + 1

    if not large_tensor(tensor_fmap) and ((ho.value * wo.value) % block_size > 0 or has_pad(pad)):
        load_im2col_dsl_output = load_im2col_dsl_no_padding(load_im2col_dsl_input, fmap_shape_NC1HWCO, kernel, pad,
                                                            stride)
    attrs = get_attrs()
    attrs['dim'] = load_im2col_set_dim(tensor_fmap, kernel, stride, pad)[0]
    return load_im2col_dsl_output, attrs
