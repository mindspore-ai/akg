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

"""operator dsl function: fc"""
import akg
import akg.tvm
from akg import backend as cce


def fc(fMapBatch, weight, fc_dtype, block_size, attrs, kernel_name="Fully_Connected"):
    """
    Computes full connection.

    Args:
        fMapBatch(akg.tvm.Tensor): Should be a 4D tensor.
        weight(akg.tvm.Tensor): Should be a 4D tensor of same type as fMapBatch.
        fc_dtype(str): Specifies data type of input tensors.
        block_size(int): Block size.
        attrs(dicts): Attributes.
        kernel_name(str): Kernel name.

    Returns:
        akg.tvm.Tensor of same type as input tensors.
    """
    # NCHW
    f_n, f_c, f_h, f_w = fMapBatch.shape
    w_n, w_c, w_h, w_w = weight.shape

    if f_c != w_c or f_h != w_h or f_w != w_w or w_n < 32:
        raise RuntimeError("invalid input shape")
    f_shape_nc1hwc0 = (f_n, f_c // block_size, f_h, f_w, block_size)

    w_shape_fractal = (w_c // block_size * w_h * w_w, w_n // block_size, block_size, block_size)

    A = akg.tvm.placeholder(f_shape_nc1hwc0, dtype=fc_dtype, name='fmap')
    B = akg.tvm.placeholder(w_shape_fractal, dtype=fc_dtype, name='weight')

    out_shape_nc1hwc0 = (f_n, w_n // block_size, 1, 1, block_size)

    weight_shape_nc1hwc0 = (w_n, w_c // block_size, w_h, w_w, block_size)

    _, k_c1, k_h, k_w, k_c0 = weight_shape_nc1hwc0

    kc1 = akg.tvm.reduce_axis((0, k_c1), name='kc1')
    kh = akg.tvm.reduce_axis((0, k_h), name='kh')
    kw = akg.tvm.reduce_axis((0, k_w), name='kw')
    kc0 = akg.tvm.reduce_axis((0, k_c0), name='kc0')

    res = akg.tvm.compute(out_shape_nc1hwc0,
                      lambda n, c1, h, w, c0: akg.lang.cce.mmad(
                          A[n, kc1, (h + kh), (w + kw), kc0]
                          * B[(kc1 * k_h + kh) * k_w + kw, c1, c0, kc0],
                          axis=[kc1, kh, kw, kc0]), name="res")

    s = akg.tvm.create_schedule(res.op)
    with akg.build_config(add_lower_pass=cce.debug_mode(0), dump_pass_ir=True):
        mod = akg.build(s, [A, B, res], "cce", name=kernel_name, attrs=attrs, polyhedral=True)

    return mod
