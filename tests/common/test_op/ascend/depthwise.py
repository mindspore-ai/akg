# Copyright 2019-2021 Huawei Technologies Co., Ltd
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

"""operator dsl function: depthwise"""
import akg.tvm
import akg.lang.ascend
from akg import dim
import akg.utils as utils
from akg.utils import custom_tiling as ct_util

depthwise_set_dim_map = {
    str((16, 112, 112, 32, 1, 3, 3, 1, 1, 1, 1)): (15, 16, 256, 3 * 16, 16),
    str((16, 112, 112, 64, 1, 3, 3, 1, 1, 2, 2)): (15, 16, 512, 3 * 16, 16),
    str((1, 56, 56, 128, 1, 3, 3, 1, 1, 1, 1)): (15, 16, 512, 3 * 16, 16),
    str((1, 56, 56, 128, 1, 3, 3, 1, 1, 2, 2)): (15, 16, 512, 3 * 16, 16),
    str((1, 28, 28, 256, 1, 3, 3, 1, 1, 1, 1)): (7, 16, 512, 3 * 16, 16),
    str((1, 28, 28, 256, 1, 3, 3, 1, 1, 2, 2)): (30, 16, 512, 3 * 16, 16),
    str((1, 14, 14, 512, 1, 3, 3, 1, 1, 1, 1)): (7, 16, 512, 3 * 16, 16),
    str((1, 14, 14, 512, 1, 3, 3, 1, 1, 2, 2)): (30, 16, 512, 3 * 16, 16),
    str((1, 7, 7, 1024, 1, 3, 3, 1, 1, 1, 1)): (15, 16, 512, 3 * 16, 16),
    # mobilenet V2
    str((1, 112, 112, 32, 1, 3, 3, 1, 1, 1, 1)): (114, 16, 256, 48, 16),
    str((1, 112, 112, 96, 1, 3, 3, 1, 1, 2, 2)): (114, 16, 512, 48, 16),
    str((1, 7, 7, 960, 1, 3, 3, 1, 1, 1, 1)): (9, 16, 512, 3 * 16, 16),

}


def depthwise_set_dim_func(data, N, H, W, CI, k_ch, KH, KW, PAD_H, PAD_W, SH, SW, block_size, use_bias=False):
    key = [N, H, W, CI, k_ch, KH, KW, PAD_H, PAD_W, SH, SW]
    hash_key = str((tuple(key)))
    clear = True
    if hash_key in depthwise_set_dim_map:
        cutH, cutCo, _, _, _ = depthwise_set_dim_map[hash_key]
        clear = False
    else:
        # raise RuntimeError("other can not find cutH, cutCo, cutM, cutK, cutN")
        cutH = (KH - 1) * KH + 1
        cutCo = 16
    group = CI // block_size
    CO = CI * k_ch

    OH = (H + 2 * PAD_H - KH) // SH + 1
    OW = (W + 2 * PAD_W - KW) // SW + 1

    out_n, out_c1, out_h, out_w, out_c0 = [N, CO // block_size, OH, OW, block_size]
    # set dim
    tile_out_h = (cutH - KH) // SH + 1

    info = dim.Dim()
    if (out_n > 1):
        info.setdim(index=0, axis=0, tilel1=1, tilel0=0)  # n
    if (out_c1 > 1):
        info.setdim(index=0, axis=0, tilel1=cutCo // block_size, tilel0=0)  # c1
    if (out_h > 1):
        info.setdim(index=0, axis='H', tilel1=tile_out_h, tilel0=0)  # h
    if (out_w > 1):
        info.setdim(index=0, axis=3, tilel1=out_w, tilel0=0)  # w
    if (out_c0 > 1):
        info.setdim(index=0, axis=4, tilel1=out_c0, tilel0=0)  # c0
    assert CI // block_size // group == 1
    if (CI // block_size // group > 1):
        info.setdim(index=0, axis=5, tilel1=CI // block_size // group, tilel0=0)  # kc1
    if (KH > 1):
        info.setdim(index=0, axis=5, tilel1=KH, tilel0=0)  # kh
    if (KW > 1):
        info.setdim(index=0, axis=5, tilel1=KW, tilel0=0)  # kw
    if clear:
        info = ""
    return str(info)


@ct_util.reg_set_dim_func(depthwise_set_dim_func)
def depthwise(data, N, H, W, CI, k_ch, KH, KW, PAD_H, PAD_W, SH, SW, block_size, use_bias=False):
    """
    Depthwise 5-D convolutions,every channel has its filter-kernel

    Args:
        data (list):a list,the size is 3 if use_bias else the size is 2;
              data[0] tvm.tensor.Tensor of type float16 ,shape 5D(N, CI//C0, C0, H, W)
              data[1] tvm.tensor.Tensor of type float16 ,shape 6D(CI//(CI//C0)//C0, KH, KW, k_ch*CI//C0, C0, C0)
              data[2] tvm.tensor.Tensor of type float16 ,shape 5D(N, CI*k_ch//C0, OH, OW, C0)
        N (int): batchsize
        H (int): height of featureMap
        W (int): width of featureMap
        CI (int): channel of featureMap
        k_ch (int): channel of Filter
        KH (int): height of Filter
        KW (int): width of Filter
        PAD_H (int): padding pixels in vertical direction
        PAD_W (int): padding pixels in horizontal direction
        SH (int): stride in vertical direction
        SW (int): stride in horizontal direction
        block_size (int): a int var also called "C0"
        use_bias (bool ): If True need add bias, else bias equal to zero.
    Returns:
        akg.tvm.Tensor of same type as data, shape is 5D(N, CI*k_ch//C0, OH, OW, C0)
    """

    check_list = ["float16"]
    dtype = data[0].dtype
    if not (dtype in check_list):
        raise RuntimeError("depthwise only support %s while dtype is %s" % (",".join(check_list), dtype))

    for i in range(len(data)):
        shape = data[i].shape
        utils.check_shape(shape)
    conv_dtype = 'float16'
    group = CI // block_size
    CO = CI * k_ch

    assert k_ch == 1
    assert CO % group == 0 and CI % group == 0
    assert CO % block_size == 0 and (CI // group) % block_size == 0
    clear = False  # if clear, use auto tiling
    # (N, CI, H, W) -> (N, C0, H, W, C1)
    A = data[0]
    # (CO, CI // group, KH, KW) -> (CI // group // block * KH * KW, CO // block, block, block)
    B = data[1]
    if use_bias:
        bias = data[2]
        bias_name = bias.op.name
    else:
        bias = None
        bias_name = "bias_name"

    key = [N, H, W, CI, k_ch, KH, KW, PAD_H, PAD_W, SH, SW]
    hash_key = str((tuple(key)))
    if hash_key in depthwise_set_dim_map:
        cutH, cutCo, cutM, cutK, cutN = depthwise_set_dim_map[hash_key]
    else:
        # raise RuntimeError("other can not find cutH, cutCo, cutM, cutK, cutN")
        cutH = (KH - 1) * KH + 1
        cutCo = 16
        cutM = 16
        cutK = 16 * KH * KW
        cutN = 16
        clear = True  # use auto tiling
    OH = (H + 2 * PAD_H - KH) // SH + 1
    OW = (W + 2 * PAD_W - KW) // SW + 1

    kc1 = akg.tvm.reduce_axis((0, CI // block_size // group), name="kc1")
    kh = akg.tvm.reduce_axis((0, KH), name="kh")
    kw = akg.tvm.reduce_axis((0, KW), name="kw")
    kc0 = akg.tvm.reduce_axis((0, block_size), name="kc0")

    p_top, p_bottom, p_left, p_right = PAD_H, PAD_H, PAD_W, PAD_W
    output_name = "output"
    output_bias_name = "output_bias"

    attr = {
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
        "feature": A.op.name,
        "filter": B.op.name,
        "bias": bias_name,
        "res": output_name,
        "res_bias": output_bias_name
    }

    if not clear:
        attr["pragma_conv_h_cut"] = cutH
        attr["pragma_conv_w_cut"] = W + 2 * PAD_W
        attr["pragma_conv_co_cut"] = cutCo
        attr["pragma_conv_m_cut"] = cutM
        attr["pragma_conv_k_cut"] = cutK
        attr["pragma_conv_n_cut"] = cutN
    C = akg.tvm.compute((N, CO // block_size, OH, OW, block_size),
                        lambda n, c1, h, w, c0: akg.lang.ascend.mmad(
                            akg.tvm.if_then_else(akg.tvm.any((h * SH + kh) < p_top, (h * SH + kh) > (H + p_top - 1),
                                                             (w * SW + kw) < p_left, (w * SW + kw) > (W + p_left - 1)),
                                                 akg.tvm.const(0.0, conv_dtype),
                                                 A[n, c1 // ((CO // block_size) // group) * (
                                                             (CI // block_size) // group) + kc1, (
                                                               h * SH + kh - p_top), (w * SW + kw - p_left), kc0])
                            # A[n, kc1, (h * SH + kh - p_top), (w * SW + kw - p_left), kc0])
                            * B[(kc1 * KH + kh) * KW + kw, c1, c0, kc0], axis=[kc1, kh, kw, kc0]),
                        attrs=attr, name=output_name)

    if use_bias:
        out = akg.tvm.compute(C.shape, lambda n, c1, h, w, c0: C[n, c1, h, w, c0] + bias[0, c1, 0, 0, c0],
                              name=output_bias_name)

    else:
        out = C

    return out
