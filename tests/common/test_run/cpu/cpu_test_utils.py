import numpy as np


def unpack_nchwc_to_nchw_python(data, dtype):
    """Unpack data(NCHW[x]c) to data(NCHW)"""

    n, c_outer, h, w, c_inner = data.shape
    target_shape = (n, c_outer*c_inner, h, w)
    unpacked_res = np.zeros(target_shape).astype(dtype)
    for nn in range(target_shape[0]):
        for cc in range(target_shape[1]):
            for hh in range(target_shape[2]):
                for ww in range(target_shape[3]):
                    unpacked_res[nn][cc][hh][ww] = data[nn][cc //
                                                            c_inner][hh][ww][cc % c_inner]

    return unpacked_res


def unpack_kcrsxy_to_kcrs_python(weight, dtype):
    """Unpack weight(KCRS[x]c[y]k) to weight(KCRS)"""

    oc_outer, ic_outer, kh, kw, ic_inner, oc_inner = weight.shape
    target_shape = (oc_outer*oc_inner, ic_outer*ic_inner, kh, kw)
    unpacked_res = np.zeros(target_shape).astype(dtype)
    for oc in range(target_shape[0]):
        for ic in range(target_shape[1]):
            for khh in range(target_shape[2]):
                for kww in range(target_shape[3]):
                    unpacked_res[oc][ic][khh][kww] = weight[oc // oc_inner][ic //
                                                                            ic_inner][khh][kww][ic % ic_inner][oc % oc_inner]

    return unpacked_res


def pack_nchw_to_nchwc_python(data, c_inner, dtype):
    """Pack data(NCHW) to data(NCHW[x]c)"""
    n, c, h, w = data.shape
    target_shape = (n, c // c_inner, h, w, c_inner)
    packed_res = np.zeros(target_shape).astype(dtype)
    for nn in range(target_shape[0]):
        for cc_out in range(target_shape[1]):
            for hh in range(target_shape[2]):
                for ww in range(target_shape[3]):
                    for cc_in in range(target_shape[4]):
                        packed_res[nn][cc_out][hh][ww][cc_in] = data[nn][cc_out * c_inner + cc_in][hh][ww]

    return packed_res