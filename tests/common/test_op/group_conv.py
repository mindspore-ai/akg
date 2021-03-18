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

"""operator dsl function: group_conv"""
import akg.topi
import akg.tvm
import akg
import akg.lang.cce
from akg.utils import kernel_exec as utils
from akg import dim


def group_conv(N, H, W, CI, CO, group, KH, KW, PAD_H, PAD_W, SH, SW, cutH, cutCo, cutM, cutK, cutN, block_size, use_bias=False, kernel_name='conv'):
    """
    split channels of FeatureMap to some groups,every group has its filter-kernel

    Args:
        args1:a list,the size is 3 if use_bias else the size is 2;
              data[0] akg.tvm.Tensor of type float16 ,shape 5D(N, CI//C0, C0, H, W)
              data[1] akg.tvm.Tensor of type float16 ,shape 6D(CI//(CI//C0)//C0, KH, KW, k_ch*CI//C0, C0, C0)
              data[2] akg.tvm.Tensor of type float16 ,shape 5D(N, CI*k_ch//C0, OH, OW, C0)
        N:batchsize
        H:height of featureMap
        W:width of featureMap
        CI:channel of featureMap
        C0:num of Filters
        group:num of spliting channels of FeatureMap
        KH:height of Filter
        KW:width of Filter
        PAD_H:padding pixels in vertical direction
        PAD_W:padding pixels in horizontal direction
        SH:stride in vertical direction
        SW:stride in horizontal direction
        block_size:a int var
        use_bias:a bool value
    Returns:
        akg.tvm.Tensor of same type as data, shape is 5D(N, C0//block_size, block_size, OH, OW)
    """

    conv_dtype = "float16"

    if cutH == H:
        cutH += PAD_H + PAD_H

    assert CO % group == 0 and CI % group == 0
    assert CO % block_size == 0 and (CI // group) % block_size == 0

    # (N, CI, H, W) -> (N, C0, H, W, C1)
    A = akg.tvm.placeholder((N, CI // block_size, H, W, block_size), dtype=conv_dtype, name="A")
    # (CO, CI // group, KH, KW) -> (CI // group // block * KH * KW, CO // block, block, block)
    B = akg.tvm.placeholder((CI // group // block_size * KH * KW, CO // block_size, block_size, block_size), dtype=conv_dtype, name="B")

    bias = akg.tvm.placeholder((1, CO // block_size, 1, 1, block_size), dtype=conv_dtype, name="bias")

    OH = (H + 2 * PAD_H - KH) // SH + 1
    OW = (W + 2 * PAD_W - KW) // SW + 1

    kc1 = akg.tvm.reduce_axis((0, CI // block_size // group), name="kc1")
    kh = akg.tvm.reduce_axis((0, KH), name="kh")
    kw = akg.tvm.reduce_axis((0, KW), name="kw")
    kc0 = akg.tvm.reduce_axis((0, block_size), name="kc0")

    p_top, p_bottom, p_left, p_right = PAD_H, PAD_H, PAD_W, PAD_W
    output_name = "output"
    output_bias_name = "output_bias"

    C = akg.tvm.compute((N, CO // block_size, OH, OW, block_size),
                    lambda n, c1, h, w, c0: akg.lang.cce.mmad(
                        akg.tvm.if_then_else(akg.tvm.any((h * SH + kh) < p_top, (h * SH + kh) > (H + p_top - 1),
                                                 (w * SW + kw) < p_left, (w * SW + kw) > (W + p_left - 1)), akg.tvm.const(0.0, conv_dtype),
                                         A[n, c1 // ((CO // block_size) // group) * ((CI // block_size) // group) + kc1, (h * SH + kh - p_top), (w * SW + kw - p_left), kc0])
        * B[(kc1 * KH + kh) * KW + kw, c1, c0, kc0], axis=[kc1, kh, kw, kc0]),
        attrs={
                        "pragma_conv_kernel_n": CO,
                        "pragma_conv_kernel_h": KH,
                        "pragma_conv_kernel_w": KW,
                        "pragma_conv_padding_top": p_top,
                        "pragma_conv_padding_bottom": p_bottom,
                        "pragma_conv_padding_left": p_left,
                        "pragma_conv_padding_right": p_right,
                        "pragma_conv_bypass_l1": 1,
                        "pragma_conv_stride_h": SH,
                        "pragma_conv_stride_w": SW,
                        "pragma_conv_fm_n": N,
                        "pragma_conv_fm_c": CI,
                        "pragma_conv_fm_h": H,
                        "pragma_conv_fm_w": W,
                        "pragma_conv_dilation_h": 1,
                        "pragma_conv_dilation_w": 1,
                        "pragma_conv_h_cut": cutH,
                        "pragma_conv_w_cut": W + 2 * PAD_W,
                        "pragma_conv_co_cut": cutCo,
                        "pragma_conv_m_cut": cutM,
                        "pragma_conv_k_cut": cutK,
                        "pragma_conv_n_cut": cutN,
                        "feature": A.op.name,
                        "filter": B.op.name,
                        "bias": bias.op.name,
                        "res": output_name,
                        "res_bias": output_bias_name},
        name=output_name)

    if use_bias:
        out = akg.tvm.compute(C.shape, lambda n, c1, h, w, c0: C[n, c1, h, w, c0] + bias[0, c1, 0, 0, c0], name=output_bias_name)
        bufs = [A, B, bias, out]
    else:
        out = C
        bufs = [A, B, out]

    # create schedule for cce
    s = akg.tvm.create_schedule([out.op])

    # set cut / tiling
    out_n, out_c1, out_h, out_w, out_c0 = akg.topi.util.get_const_tuple(out.shape)

    # set dim
    tile_out_h = (cutH - KH) // SH + 1

    info = dim.Dim()
    if (out_n > 1):
        info.setdim(index=0, axis=0, tilel1=1, tilel0=0)       # n
    if (out_c1 > 1):
        info.setdim(index=0, axis=0, tilel1=cutCo // block_size, tilel0=0)  # c1
    if (out_h > 1):
        info.setdim(index=0, axis='H', tilel1=tile_out_h, tilel0=0)  # h
    if (out_w > 1):
        info.setdim(index=0, axis=3, tilel1=out_w, tilel0=0)   # w
    if (out_c0 > 1):
        info.setdim(index=0, axis=4, tilel1=out_c0, tilel0=0)  # c0
    assert CI // block_size // group == 1
    if (CI // block_size // group > 1):
        info.setdim(index=0, axis=5, tilel1=CI // block_size // group, tilel0=0)      # kc1
    if (KH > 1):
        info.setdim(index=0, axis=5, tilel1=KH, tilel0=0)      # kh
    if (KW > 1):
        info.setdim(index=0, axis=5, tilel1=KW, tilel0=0)      # kw

    # build
    with akg.build_config(add_lower_pass=utils.debug_mode(0), dump_pass_ir=True):
        mod = akg.build(s, bufs, "cce", name=kernel_name, attrs={"dim": str(info)}, polyhedral=True)

    return OH, OW, A, B, C, mod
