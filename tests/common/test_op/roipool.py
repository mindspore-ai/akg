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

"""operator dsl function: roipool"""
import akg.tvm
import akg
from akg import backend as cce
from akg.utils import kernel_exec as utils
from akg.utils import validation_check as vc_util


def roipool(shape, roibox, pooled_shape, dtype, kernel_name="roipool_forward_output", attrs=None):
    check_list = ["float16"]
    if not (dtype.lower() in check_list):
        raise RuntimeError("tile_cce only support %s while dtype is %s" % (",".join(check_list), dtype))
    vc_util.check_shape(shape)
    assert (len(shape) == 4)
    assert (len(roibox) == 4)
    assert (len(pooled_shape) == 2)

    a_n, a_c, a_h, a_w = shape
    roi_t, roi_b, roi_l, roi_r = roibox
    assert (roi_t >= 0 and roi_t < roi_b and roi_b < a_h)
    assert (roi_l >= 0 and roi_l < roi_r and roi_r < a_w)

    a = akg.tvm.placeholder(shape, name="a", dtype=dtype)
    Crop = akg.tvm.compute([a_n, a_c, roi_b - roi_t, roi_r - roi_l],
                           lambda n, c, h, w: a[n, c, roi_t + h, roi_l + w])

    p_h, p_w = pooled_shape
    win_h = (roi_b - roi_t) // p_h + (1 if (roi_b - roi_t) % p_h > 0 else 0)
    win_w = (roi_r - roi_l) // p_w + (1 if (roi_r - roi_l) % p_w > 0 else 0)

    assert p_h <= (roi_b - roi_t) and p_w <= (roi_r - roi_l)

    Unpooled = akg.tvm.compute([a_n, a_c, p_h, p_w, win_h, win_w],
                               lambda n, c, h, w, wh, ww: akg.tvm.expr.Select(
                                   akg.tvm.all(
                                       h * win_h + wh < roi_b - roi_t,
                                       w * win_w + ww < roi_r - roi_l
                                   ),
                                   Crop[n, c, h * win_h + wh, w * win_w + ww],
                                   akg.tvm.const(0, a.dtype)
                               ))

    rh = akg.tvm.reduce_axis((0, win_h))
    rw = akg.tvm.reduce_axis((0, win_w))
    output_shape = [a_n, a_c, p_h, p_w]
    res = akg.tvm.compute(output_shape,
                          lambda n, c, h, w: akg.tvm.max(Unpooled[n, c, h, w, rh, rw], axis=[rh, rw]))
    s = akg.tvm.create_schedule(res.op)
    s[Crop].compute_inline()
    s[Unpooled].compute_inline()
    kernel_name = utils.gen_name_kernel(kernel_name, dtype, shape)
    with akg.build_config(add_lower_pass=cce.debug_mode(0), dump_pass_ir=True):
        mod = akg.build(s, [a, res], "cce", name=kernel_name, attrs=attrs, polyhedral=True)
        return mod, output_shape
