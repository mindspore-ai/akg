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

import tvm
import akg
from akg.utils import kernel_exec as utils
from tests.common.test_op.im2col import intrin_load_im2col

def im2col_manual_schedule(shape, kernel, stride, pad, dtype, polyhedral=True, attrs=None):
    '''
    Compute im2col via cce im2col intrin function call directly

    Args:
        shape: shape of the data
        kernel: kernel sizes for im2col
        stride: stride sizes for im2col
        pad: padding sizes for im2col, including padding top, bottom, left, and right
        dtype: type of the data

    Return:
        cce intrin function call for im2col
    '''

    load_im2col = intrin_load_im2col(dtype)

    b, c1, h, w, c0 = shape
    stride_h, stride_w = stride
    kernel_h, kernel_w = kernel
    pad_t, pad_b, pad_l, pad_r = pad
    dilation_w, dilation_h = 1, 1
    jump_offset = 1
    repeat_mode = 0
    repeat_time = 1
    csize = 0
    block_size = 16

    # output size <=> number of windows
    ho = (h + pad_b + pad_t - kernel_h) // stride_h + 1
    wo = (w + pad_r + pad_l - kernel_w) // stride_w + 1

    im2col_shape = (b,
                    (ho * wo + block_size - 1) // block_size,
                    c1 * kernel_h * kernel_w,
                    block_size,
                    c0)

    def _im2col_compute(i, j, k, data):

        j_h = (((j*block_size) // wo)*stride_h)-pad_t
        j_w = (((j*block_size) %  wo)*stride_w)-pad_l

        # num rows in l1 for fmatrix is discounted by the amount of bottom padding
        h_3d         = kernel_h - tvm.max(((j_h+kernel_h) - h), 0)
        pad_t_3d     = tvm.max(-j_h, 0)
        pad_b_3d     = tvm.max(((j_h+kernel_h) - h), 0)
        w_idx_kernel = (k % kernel_w)
        h_idx_kernel = ((k // kernel_w) % kernel_h)
        w_idx        = j_w
        # when this is < 0, the slice will start from row 0 so there is no redundancy between base address and this param
        h_idx        = tvm.min(j_h, 0)
        c1_idx = (k // kernel_w) // kernel_h

        load_im2col_input = data[i,
                            c1_idx,
                            # assume padding < kernel size
                            tvm.max(0, j_h):tvm.min(h, j_h+kernel_h),
                            0:w,
                            0:c0]

        return load_im2col(load_im2col_input,
                      w, h_3d, pad_l, pad_r, pad_t_3d, pad_b_3d,
                      w_idx_kernel, h_idx_kernel, w_idx, h_idx, 0,
                      stride_w, stride_h, kernel_w, kernel_h, dilation_w, dilation_h, jump_offset, repeat_mode, repeat_time,
                      csize)

    # tensor for the input data
    data = tvm.placeholder(shape, dtype, name="input_data")

    # assume we need the whole width of a
    # choose a section of the rows of a that encompasses all of the windows in the current window-batch
    res = tvm.compute(im2col_shape,
                      lambda i, j, k:
                          _im2col_compute(i, j, k, data),
                      name='im2col_fractal')

    # schedule for differentiation operation
    s = tvm.create_schedule([res.op])

    data_ub = s.cache_read(data, "local.L1", [res])
    res_ub = s.cache_write(res, "local.UB")

    s[data_ub].compute_at(s[res], res.op.axis[0])
    s[res_ub].compute_at(s[res], res.op.axis[2])

    with akg.build_config(add_lower_pass=utils.debug_mode(0), dump_pass_ir=True):
        mod = akg.build(s, [data, res], "cce", name="im2col_manual_schedule",
                       attrs=attrs, polyhedral=polyhedral)
        source_code = mod.imported_modules[0].get_source()
        kernel_name = "im2col_manual_schedule"
        utils.create_code(kernel_name, './', source_code)
    return mod
