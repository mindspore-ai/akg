import akg.topi as topi
from akg.topi.util import get_const_tuple
import akg.tvm as tvm


def get_channel_inners(ic_in, oc_in, in_channel, out_channel, target_str):
    """A simple tiling to definite input/output channels' inner axes"""

    if ic_in != -1 and oc_in != -1:
        return ic_in, oc_in

    simd_size = 8
    if "skylake-avx512" in target_str:
        simd_size = 16
    elif "core-avx2" in target_str:
        simd_size = 8

    oc_in = 1
    for inner in range(simd_size, 0, -1):
        if out_channel % inner == 0:
            oc_in = inner
            break

    ic_in = 1
    for inner in range(simd_size, 0, -1):
        if in_channel % inner == 0:
            ic_in = inner
            break

    return ic_in, oc_in


def pack_data(data, weight, ic_inner, oc_inner):
    """Pack data form NCHW to NCHWc"""
    n, _, ih, iw = get_const_tuple(data.shape)
    oc, ic, kh, kw = get_const_tuple(weight.shape)

    ic_outer = ic // ic_inner
    oc_outer = oc // oc_inner

    data = tvm.compute(
        (n, ic_outer, ih, iw, ic_inner),
        lambda bs, c, h, w, vc: data[bs, c * ic_inner + vc, h, w],
        name="packed_data",
    )

    weight = tvm.compute(
        (oc_outer, ic_outer, kh, kw, ic_inner, oc_inner),
        lambda occ, icc, k_h, k_w, icb, ocb: weight[occ *
                                                    oc_inner + ocb, icc * ic_inner + icb, k_h, k_w],
        name="packed_weight",
    )

    return data, weight


def unpack_nchwc_to_nchw(packed_out, out_dtype):
    """Unpack the data from layout NCHWc to NCHW"""

    n, oc_outer, oh, ow, oc_inner = get_const_tuple(packed_out.shape)

    idxmod = tvm.indexmod
    idxdiv = tvm.indexdiv

    oshape = (n, oc_outer * oc_inner, oh, ow)
    unpacked_out = tvm.compute(
        oshape,
        lambda n, c, h, w: packed_out[n, idxdiv(c, oc_inner), h, w, idxmod(c, oc_inner)].astype(
            out_dtype
        ),
        name="output_nchw",
    )
    return unpacked_out
