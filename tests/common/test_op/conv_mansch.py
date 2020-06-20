# Copyright 2019 Huawei Technologies Co., Ltd
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

"""operator dsl function: conv_mansch"""

import akg.tvm
import akg.topi
import akg
from akg import backend as cce
conv_dtype = 'float16'
block_size = 16



def manual_im2col_1repeat(indices, A, fp_w, fp_h, fm_w, fm_h, pad_t, pad_b, l1_h_fmatrix, stride_w):

    window_index, c0_index = indices


    kh_index = fp_h
    kw_index = fp_w

    # just randomly use pad_t and pad_b to make sure they're used
    h_index = fm_h + kh_index + pad_t + pad_b + l1_h_fmatrix
    w_index = fm_w + window_index*stride_w + kw_index

    # A always has dimension (1, 1, l1_h, l1_w, 16)
    return A[0, 0, h_index, w_index, c0_index]


def intrin_load3d(A_shape, strides, kernel_size, padding):


    _, _, _, _, c0_value = A_shape
    stride_h, stride_w = strides
    kernel_h, kernel_w = kernel_size
    pad_t, pad_b, pad_l, pad_r = padding

    l1_h = akg.tvm.var("l1_h", dtype='int32')
    l1_w = akg.tvm.var("l1_w", dtype='int32')
    # we know that the n-batch and C1 are fixed. The H and W of the piece of A are unknown.
    a = akg.tvm.placeholder((1, 1, l1_h, l1_w, c0_value), dtype=conv_dtype)
    fp_w = akg.tvm.var("fp_w")
    fp_h = akg.tvm.var("fp_h")
    fm_w = akg.tvm.var("fm_w")
    fm_h = akg.tvm.var("fm_h")
    fp_c1 = akg.tvm.var("fp_c1")
    pad_t = akg.tvm.var("pad_t")
    pad_b = akg.tvm.var("pad_b")
    l1_h_fmatrix = akg.tvm.var("l1_h_fmatrix")

    # Output will be of shape (block_size (window positions), C0)  = (16x16)

    c = akg.tvm.compute((block_size, c0_value), lambda *indices : manual_im2col_1repeat(indices, a, fp_w, fp_h, fm_w, fm_h,  pad_t, pad_b, l1_h_fmatrix, stride_w), name='im2col_manual')

    Ab_scope = "local.L1"
    Cb_scope = "local.L0A"

    Ab = akg.tvm.decl_buffer(a.shape, a.dtype,
                         name="Abuf",
                         offset_factor=1, scope=Ab_scope) #, strides=[akg.tvm.var("s1"), akg.tvm.var("s2"), akg.tvm.var("s3"), akg.tvm.var("s4"), akg.tvm.var("s5")])

    Cb = akg.tvm.decl_buffer(c.shape, c.dtype, name="Cbuf", offset_factor=1, scope=Cb_scope)

    def intrin_func(ins, outs, sp):
        aa = ins[0]
        dd = outs[0]
        def _body():
            ib = akg.tvm.ir_builder.create()
            ib.emit(akg.tvm.call_extern("int32", "cce_img2col_",
                                    dd.access_ptr("w"),
                                    aa.access_ptr("r"),
                                    # the constant params are dilation, jump offset, repeat-mode, # repeats, c0 mode
                                    sp[0], sp[1], sp[2], sp[3], sp[4], stride_w, stride_h, kernel_w, kernel_h, 1, 1, 1, 0, 1, 0, sp[5], sp[6], pad_l, pad_r, sp[7], l1_w))

            return ib.get()
        return _body()

    with akg.build_config(offset_factor=1):
        return akg.tvm.decl_tensor_intrin(c.op, intrin_func, binds={a: Ab, c: Cb}, scalar_params=[fp_w, fp_h, fm_w, fm_h, fp_c1, pad_t, pad_b, l1_h_fmatrix])



def im2col_fractal(A_im2col_shape, A, kernel_h, kernel_w, stride, padding):

    load3D = intrin_load3d(A.shape, stride, [kernel_h, kernel_w], padding)


    _, _, H, W, C0 = A.shape
    stride_h, stride_w = stride
    pad_t, _, pad_l, pad_r = padding
    _, _, _, windowsPerBatch, _ = A_im2col_shape
    Wo = (W + pad_l + pad_r - kernel_w)//stride_w + 1

    # assume we need the whole width of A
    # choose a section of the rows of A that encompasses all of the windows in the current window-batch
    return akg.tvm.compute(A_im2col_shape, lambda i, j, k : load3D(A[i, (k // kernel_w) // kernel_h,
                                                               # assume padding < kernel size
                                                               akg.tvm.max((((j*windowsPerBatch)//Wo)*stride_h)-pad_t, 0):
                                                               akg.tvm.min(H, ((((j*windowsPerBatch)//Wo)*stride_h)-pad_t)+kernel_h), 0:W, 0:C0],
                                                               # fetch position w, fetch position h
                                                               (k % kernel_w), ((k // kernel_w) % kernel_h),
                                                               # top-left corner w,
                                                               (((j*windowsPerBatch)%Wo)*stride_w)-pad_l,
                                                               # top-left corner h, (when this is < 0, the slice will start from row 0 so there is no redundancy
                                                               # between base address and this param)
                                                               akg.tvm.min(((j*windowsPerBatch)//Wo)*stride_h - pad_t, 0),
                                                               # initial C1
                                                               (k // kernel_w) // kernel_h,
                                                               # top padding
                                                               akg.tvm.max(pad_t - (((j*windowsPerBatch)//Wo)*stride_h), 0),
                                                               # bottom padding
                                                               akg.tvm.max(((((((j*windowsPerBatch)//Wo)*stride_h)-pad_t)+kernel_h) - H), 0),
                                                               # num rows in L1 for fmatrix is discounted by the amount of bottom padding
                                                               kernel_h - akg.tvm.max(((((((j*windowsPerBatch)//Wo)*stride_h)-pad_t)+kernel_h) - H), 0), name='im2col_fractal'))



def mad(mad_shape, A, B, fp32_mad):
    k1 = akg.tvm.reduce_axis((0, B.shape[0]), name='k1')
    k0 = akg.tvm.reduce_axis((0, block_size), name='k0')

    # If tag set to 'gemv', computeOp return tensor of specific layout.
    # e.g. gemv of 1x32, tensor C is 1x32 but occupy 16x32 fractal matrix size. gemv of 2x32 also occupy 16x32.
    if fp32_mad:
        mmad_dtype  ="float32"
        mmad_mode = 'f162f32'
    else:
        mmad_dtype = "float16"
        mmad_mode = 'f162f16'
    C = akg.tvm.compute(mad_shape,
                       lambda n, j1, i, j0: akg.lang.cce.mmad((A[n, i // block_size, k1, i % block_size, k0] * B[k1, j1, j0, k0]).astype(mmad_dtype),
                                                    axis=[k1, k0]),

                       name='mad',
                       tag='gemm',
                       attrs={'mode': mmad_mode})
    return C



def conv_dsl(placeholders, input_shape, filter_shape, padding, stride, use_bias=False, fp32_mad = True):
    """
    Calculate conv by manual schedule.

    Note:
        The number of channels must be at least block_size
        The number of filters must be at least block_size
        The number of window positions must be evenly divisible by block_size
        Wo = block_size

    Args:
        placeholders (tvm.tensor.Tensor): Tensor of type float16.
        input_shape (list): a list has 4 nums.
        filter_shape (list): a list has 4 nums.
        padding (list): a list has 2 nums.
        stride (list): a list has 2 nums.
        use_bias (bool): If True, need add bias, else bias equal to zero.
        fp32_mad (bool): If True, need cast to float16.

    Returns:
            tvm.tensor.Tensor.

    """
    # A is input, B is filter, C is bias
    if use_bias:
        A, B, C = placeholders
    else:
        A, B = placeholders

    batch_size, C1, Hi, Wi, _ = input_shape
    _, kernel_h, kernel_w, kernel_co, _ = filter_shape
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

    # mad
    # small-z-big-N
    # Here, the mad_res has 3 axes, it is guarantees linear mapping for codegen,
    # otherwise we need to do fuse.
    # If not, H axis should calculate by Zh*block_size+zh, the linearity is not guaranteed.
    mad_shape = (batch_size,
                 kernel_co // block_size,
                 HoWo_mad,
                 block_size)
    mad_res = mad(mad_shape, A_im2col_fractal_res, B, fp32_mad)
    conv_shape = (batch_size, kernel_co // block_size, Ho * Wo, block_size)
    conv_res = akg.tvm.compute(conv_shape,
                           lambda n, i, j, k: mad_res(n, i, j, k),
                           name='conv',
                           attrs={'kernel_h': kernel_h, 'kernel_w': kernel_w, 'padding': padding, 'stride': stride}, tag = "conv2d")
    if fp32_mad:
        conv_res = akg.topi.cast(conv_res, "float16")

    if use_bias:
        conv_res = akg.tvm.compute(conv_shape,
                               lambda n, i, j, k: conv_res(n, i, j, k) + C[0, i, 0, 0, k],
                               name='conv_bias_add', tag = "broadcast")
    return [conv_res]



def conv_sch(s, placeholders, tiling_factor_h=0, tiling_factor_m=0, tiling_factor_k=0, tiling_factor_n=0):

    out = placeholders[1][-1]
    scheduled_ops = []

    inputs_consumers = {}
    compute_axes = {}

    def traverse(op):
        """inline all one-to-one-mapping operators except the last stage (output)"""
        if akg.topi.tag.is_broadcast(op.tag):
            _schedule_broadcast(s, op, tiling_factor_h=tiling_factor_h,
                                tiling_factor_m=tiling_factor_m,
                                inputs_consumers = inputs_consumers, compute_axes = compute_axes)
            for tensor in op.input_tensors:
                if tensor.op.input_tensors and tensor.op not in scheduled_ops:
                    traverse(tensor.op)
        else:
            _schedule_conv2d(s, op, tiling_factor_h=tiling_factor_h,
                             tiling_factor_m=tiling_factor_m, tiling_factor_k=tiling_factor_k,
                             tiling_factor_n=tiling_factor_n,  compute_axes = compute_axes)

        scheduled_ops.append(op)

    def cache_inputs():
        for inputs in inputs_consumers.keys():
            input_ub = s.cache_read(inputs, cce.scope_ubuf, inputs_consumers[inputs])
            out = s.outputs[0] # this is temporary
            s[input_ub].compute_at(s[out], compute_axes["e_Ncut_o"])


    traverse(out.op)
    cache_inputs()

    return s



def _schedule_broadcast(s, op,  tiling_factor_h=0, tiling_factor_m=0,
                     inputs_consumers = None, compute_axes = None):
    if inputs_consumers is None:
        inputs_consumers = {}
    if compute_axes is None:
        compute_axes = {}
    if akg.topi.tag.is_broadcast(op.tag):
        C = op.output(0)
        CC = s.cache_write(C, cce.scope_ubuf)  # this is necessary for all tensors if using compute_inline
        if C.op in s.outputs:
            # tile, reorder
            compute_axes["e_Ncut_o"], compute_axes["e_Ncut_i"] = s[C].split(C.op.axis[0], factor=1)
            compute_axes["e_hcut_o"], compute_axes["e_hcut_i"] = s[C].split(C.op.axis[2], factor=tiling_factor_h) # ->A row major, A
            compute_axes["e_mcut_o"], compute_axes["e_mcut_i"] = s[C].split(compute_axes["e_hcut_i"], factor=tiling_factor_m) # ->mad_ubuf
            s[C].reorder(compute_axes["e_Ncut_o"],compute_axes["e_hcut_o"], compute_axes["e_mcut_o"],
                         compute_axes["e_Ncut_i"], C.op.axis[1], compute_axes["e_mcut_i"], C.op.axis[3])
        else:
            s[C].compute_inline()
        out = s.outputs[0] # this is temporary
        s[CC].compute_at(s[out], compute_axes["e_Ncut_i"]) # corresponds to N_cut_i axis

        # collect information about consumers of input tensors
        # to be used later in cache_inputs() method
        for inputs in op.input_tensors:
            # only cache input tensors, i.e., tensors that don't have their own inputs
            if  (len(inputs.op.input_tensors) == 0):
                if inputs in inputs_consumers:
                    inputs_consumers[inputs].append(CC)
                else:
                    inputs_consumers[inputs] = [CC]

def _schedule_conv2d(s, op, tiling_factor_h=0, tiling_factor_m=0, tiling_factor_k=0,
                         tiling_factor_n=0, compute_axes = dict()):
    def __template_Cut1_ConvH_MKN():
        # tiling M K N, from L1 to L0A
        # in case test_CCE_Conv((1, 96, 36, 36), (256, 96, 5, 5), (0, 0), (1, 1), Tile_h=20, Tile_co=0, Tile_m=128,
        # Tile_k=128, Tile_n=128), we want to get the 128 * 128 mad, expressed as 8 * 8 * 16 * 16.
        # So we will tile the L0ABC in cube calculation unit, and compute_at it to mad_ubuf which should be tiled
        # by same M & N (it does not have k)
        Mad_cc_Ncut_o, Mad_cc_Ncut_i = s[Mad_cc].split(Mad_cc.op.axis[0], factor=1)
        Mad_cc_mcut_o, Mad_cc_mcut_i = s[Mad_cc].split(Mad_cc.op.axis[2], factor=tiling_factor_m)
        Mad_cc_kcut_o, Mad_cc_kcut_i = s[Mad_cc].split(Mad_cc.op.reduce_axis[0], factor=tiling_factor_k)
        Mad_cc_ncut_o, Mad_cc_ncut_i = s[Mad_cc].split(Mad_cc.op.axis[1], factor=tiling_factor_n)
        s[Mad_cc].reorder(Mad_cc_Ncut_o, Mad_cc_ncut_o, Mad_cc_mcut_o, Mad_cc_kcut_o, Mad_cc_Ncut_i, Mad_cc_ncut_i,
                          Mad_cc_mcut_i, Mad_cc.op.axis[3], Mad_cc_kcut_i, Mad_cc.op.reduce_axis[1])

        compute_axes["Mad_cc_kcut_o"] = Mad_cc_kcut_o

        # we tile Mad_ubuf.op.axis[0] by factor 1 because we want to remain the n axis in for loop for codegen.
        # we should make sure that every axis is in the for loop for an intrinsic.
        Mad_ubuf_Ncut_o, Mad_ubuf_Ncut_i = s[Mad_ubuf].split(Mad_ubuf.op.axis[0], factor=1)
        Mad_ubuf_mcut_o, Mad_ubuf_mcut_i = s[Mad_ubuf].split(Mad_ubuf.op.axis[2], factor=tiling_factor_m)
        Mad_ubuf_ncut_o, Mad_ubuf_ncut_i = s[Mad_ubuf].split(Mad_ubuf.op.axis[1], factor=tiling_factor_n)
        s[Mad_ubuf].reorder(Mad_ubuf_Ncut_o, Mad_ubuf_ncut_o, Mad_ubuf_mcut_o, Mad_ubuf_Ncut_i, Mad_ubuf_ncut_i,
                            Mad_ubuf_mcut_i, Mad_ubuf.op.axis[3])
        s[Mad_cc].compute_at(s[Mad_ubuf], Mad_ubuf_mcut_o)

        s[Mad_ubuf].compute_at(se, compute_axes["e_mcut_o"]) # corresponds to m_cut_o axis

        s[Mad_cc].pragma(Mad_cc_Ncut_i, 'mad_pattern', 1)
        s[Mad_cc].emit_insn(Mad_cc_Ncut_i, 'mad')

        # emit convolution params.
        s[Mad_cc].pragma(Mad_cc_kcut_o, 'is_reduce_k_outer', 1)

    se = s.stage_map[s.outputs[0]]

    conv_res = op.output(0)
    mad_res = s[conv_res].op.input_tensors[0]
    A_im2col_fractal_res = s[mad_res].op.input_tensors[0]
    # dsl inputs
    A = s[A_im2col_fractal_res].op.input_tensors[0]  # two levels?
    B = s[mad_res].op.input_tensors[1]

    if conv_res.op in s.outputs: # need to tile, reorder, dma_copy, no more pragmas
        compute_axes["e_Ncut_o"], compute_axes["e_Ncut_i"] = s[conv_res].split(conv_res.op.axis[0], factor=1)
        compute_axes["e_hcut_o"], compute_axes["e_hcut_i"] = s[conv_res].split(conv_res.op.axis[2], factor=tiling_factor_h) # ->A row major, A
        compute_axes["e_mcut_o"], compute_axes["e_mcut_i"] = s[conv_res].split(compute_axes["e_hcut_i"], factor=tiling_factor_m) # ->mad_ubuf
        s[conv_res].reorder(compute_axes["e_Ncut_o"],compute_axes["e_hcut_o"], compute_axes["e_mcut_o"],
                            compute_axes["e_Ncut_i"],  conv_res.op.axis[1],  compute_axes["e_mcut_i"], conv_res.op.axis[3])
        # conv copy

    else:
        s[conv_res].compute_inline()

    Mad_ubuf = s.cache_write(mad_res, cce.scope_ubuf)
    Mad_cc = s.cache_write(Mad_ubuf, cce.scope_cc)

    __template_Cut1_ConvH_MKN()

    A_cbuf = s.cache_read(A, cce.scope_cbuf, [A_im2col_fractal_res])
    A_ca = s.cache_write(A_im2col_fractal_res, cce.scope_ca)
    s[A_ca].compute_at(s[Mad_cc], compute_axes["Mad_cc_kcut_o"])
    s[A_im2col_fractal_res].compute_inline()
    B_cbuf = s.cache_read(B, cce.scope_cbuf, [Mad_cc])
    B_cb = s.cache_read(B_cbuf, cce.scope_cb, [Mad_cc])
    s[B_cb].compute_at(s[Mad_cc], compute_axes["Mad_cc_kcut_o"])
    s[B_cbuf].compute_at(s[Mad_cc], compute_axes["Mad_cc_kcut_o"])
    s[mad_res].compute_inline()

    s[A_cbuf].compute_at(se, compute_axes["e_hcut_o"]) # corresponds to h_cut_o axis
    # emit insn





def test_CCE_Conv(FMap_shape, Filter_shape, Pad, Stride,
                  Tile_h=0, Tile_co=0, Tile_m=0, Tile_k=0, Tile_n=0,
                  use_bias=False, fp32_mad = True, kernel_name="conv"):

    # adjust to TilingApi
    # feature map (NCHW -> NC1HWC0)
    fmap_n, fmap_c, fmap_h, fmap_w = FMap_shape
    fmap_shape_NC1HWCO = (fmap_n, fmap_c // block_size, fmap_h, fmap_w, block_size)

    # filter (NCHW -> C1HWNC0)
    filter_n, filter_c, filter_h, filter_w = Filter_shape
    filter_shape_C1HWNC0 = (filter_c // block_size, filter_h, filter_w, filter_n, block_size)
    # filter (C1HWNC0 -> filter_fractal)
    filter_shape_fractal = (
        filter_c * filter_h * filter_w // block_size, filter_n // block_size, block_size, block_size)

    # stride (stride_h, stride_w)
    stride = Stride

    # fmap_placeholder (NC1HWCO)
    fmap_placeholder = akg.tvm.placeholder(fmap_shape_NC1HWCO, dtype=conv_dtype, name='fmap')
    # filter_placeholder (fractal)
    filter_placeholder = akg.tvm.placeholder(filter_shape_fractal, dtype=conv_dtype, name='filter')

    if use_bias:
        bias_shape = (1, filter_n // block_size, 1, 1, block_size)
        bias_placeholder = akg.tvm.placeholder(bias_shape, dtype= conv_dtype, name='bias')
        conv_dsl_input = (fmap_placeholder, filter_placeholder, bias_placeholder)
    else:
        conv_dsl_input = (fmap_placeholder, filter_placeholder)

    conv_dsl_outputs = conv_dsl(conv_dsl_input, fmap_shape_NC1HWCO, filter_shape_C1HWNC0, Pad, stride, use_bias, fp32_mad)

    # calculate the tiling factor.
    Wo = (fmap_w + Pad[2] + Pad[3] - filter_w) // (stride[1]) + 1
    H_tiling = (Tile_h - filter_h) // (stride[0]) + 1

    # For adjusting to TilingApi, here are some tiling factor changes.
    # tiling_factor_h occurs in L1, and Tile_n is means the n in 'nchw', so we need translate it to H_tiling
    # used as Ho in A_im2col_row_major_shape
    # others are similar, they need to be changed to format where them are used.
    tiling_factor_h = H_tiling * Wo // block_size * block_size
    tiling_factor_co = Tile_co // block_size
    tiling_factor_m = Tile_m // block_size * block_size
    tiling_factor_n = Tile_n // block_size
    tiling_factor_k = Tile_k // block_size

    # schedule
    # pick the last one as the final result
    s = akg.tvm.create_schedule(conv_dsl_outputs[-1].op)


    conv_sch(s, (conv_dsl_input, conv_dsl_outputs), tiling_factor_h=tiling_factor_h,
             tiling_factor_m=tiling_factor_m, tiling_factor_k=tiling_factor_k, tiling_factor_n=tiling_factor_n)

    args = list(conv_dsl_input) + [conv_dsl_outputs[-1]]
    with akg.build_config(add_lower_pass = cce.debug_mode(0), dump_pass_ir = True):
        mod = akg.build(s, args, "cce", name=kernel_name, attrs= {"loop_partition_unroll": True})
        return mod
