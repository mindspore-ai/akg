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

"""operator dsl function:encode"""

import akg
import akg.lang.cce

from akg.tvm.hybrid import script
import akg.tvm
import akg.topi


COORDINATES_LEN = 4
COORDINATES_PAD_LEN = 8


def bouding_box_encode(anchor_box, groundtruth_box, anchor_samples, scale_factors, epsilon=1e-5):
    """
    Calculate bounding box encode.

    Args:
        anchor_box: akg.tvm.Tensor.
        groundtruth_box: akg.tvm.Tensor.
        anchor_samples: akg.tvm.Tensor.
        scale_factors: Tuple or list.
        epsilon: Default to be 1e-5.

    Returns:
        Tensor.
    """
    # check shapes
    anchor_box_shape = [x.value for x in anchor_box.shape]
    groundtruth_box_shape = [x.value for x in groundtruth_box.shape]
    anchor_samples_shape = [x.value for x in anchor_samples.shape]
    for shape in (anchor_box_shape, groundtruth_box_shape, anchor_samples_shape):
        check_shape(shape)
    # num archors
    assert anchor_box_shape[0] == anchor_samples_shape[1]
    # batch size
    assert groundtruth_box_shape[0] == anchor_samples_shape[0]
    assert (not scale_factors or len(scale_factors) == COORDINATES_LEN) and \
        (anchor_box_shape[-1] == COORDINATES_PAD_LEN) and \
        (groundtruth_box_shape[-1] == COORDINATES_PAD_LEN)

    # check dtypes; (vextract instruction only support for float16)
    check_list = ["float16"]
    assert anchor_box.dtype == groundtruth_box.dtype
    dtype = anchor_box.dtype
    if not dtype.lower() in check_list:
        raise RuntimeError("concat_cce only support %s while dtype is %s" % (",".join(check_list), dtype))
    assert anchor_samples.dtype == "int32"

    # extract coordinate for anchor
    reducer = akg.tvm.comm_reducer(lambda x, y: y, lambda t: akg.tvm.const(0, dtype=t), name="reducer")
    anchor_coordinate_shape = (anchor_box_shape[0],)
    k0 = akg.tvm.reduce_axis((0, 8), name='k0')
    ymin_a = akg.tvm.compute(anchor_coordinate_shape, lambda j0: reducer(
        akg.lang.cce.extract0(anchor_box[j0, k0]), axis=k0), name="ymin_a")
    k1 = akg.tvm.reduce_axis((0, 8), name='k1')
    xmin_a = akg.tvm.compute(anchor_coordinate_shape, lambda j1: reducer(
        akg.lang.cce.extract1(anchor_box[j1, k1]), axis=k1), name="xmin_a")
    k2 = akg.tvm.reduce_axis((0, 8), name='k2')
    ymax_a = akg.tvm.compute(anchor_coordinate_shape, lambda j2: reducer(
        akg.lang.cce.extract2(anchor_box[j2, k2]), axis=k2), name="ymax_a")
    k3 = akg.tvm.reduce_axis((0, 8), name='k3')
    xmax_a = akg.tvm.compute(anchor_coordinate_shape, lambda j3: reducer(
        akg.lang.cce.extract3(anchor_box[j3, k3]), axis=k3), name="xmax_a")
    # get center coordinates and sizes for anchor
    width_a_raw = akg.lang.cce.vsub(xmax_a, xmin_a)
    height_a_raw = akg.lang.cce.vsub(ymax_a, ymin_a)
    height_a_half = akg.lang.cce.vmuls(height_a_raw, akg.tvm.const(0.5, dtype))
    ycenter_a_raw = akg.lang.cce.vadd(ymin_a, height_a_half)
    width_a_half = akg.lang.cce.vmuls(width_a_raw, akg.tvm.const(0.5, dtype))
    xcenter_a_raw = akg.lang.cce.vadd(xmin_a, width_a_half)

    # extract coordinate for anchor_sample
    @script
    def hy_func_extract_sample(anchor_samples_hy, groundtruth_box_hy):
        batch_size, num_anchor = anchor_samples_hy.shape
        _, num_groundtruth, _ = groundtruth_box_hy.shape
        output = output_tensor((COORDINATES_PAD_LEN, batch_size, num_anchor), groundtruth_box_hy.dtype)
        for i in range(batch_size):
            for j in range(num_anchor):
                # COORDINATES_PAD_LEN should be replace by COORDINATES_LEN and need to valid
                for m in range(COORDINATES_PAD_LEN):
                    # loop for k should be replace by anchor_samples_hy[i, j], but now have some problem with anchor_samples_hy[i, j]
                    for k in range(num_groundtruth):
                        if k == anchor_samples_hy[i, j]:
                            output[m, i, j] = groundtruth_box_hy[i, k, m]
        return output
    anchor_samples_box_extract = hy_func_extract_sample(anchor_samples, groundtruth_box)
    ymin = akg.tvm.compute(anchor_samples_shape, lambda *indice: anchor_samples_box_extract[0, indice[0], indice[1]], name="ymin")
    xmin = akg.tvm.compute(anchor_samples_shape, lambda *indice: anchor_samples_box_extract[1, indice[0], indice[1]], name="xmin")
    ymax = akg.tvm.compute(anchor_samples_shape, lambda *indice: anchor_samples_box_extract[2, indice[0], indice[1]], name="ymax")
    xmax = akg.tvm.compute(anchor_samples_shape, lambda *indice: anchor_samples_box_extract[3, indice[0], indice[1]], name="xmax")
    # get center coordinates and sizes for anchor_sample
    width = akg.lang.cce.vsub(xmax, xmin)
    height = akg.lang.cce.vsub(ymax, ymin)
    height_half = akg.lang.cce.vmuls(height, akg.tvm.const(0.5, dtype))
    ycenter = akg.lang.cce.vadd(ymin, height_half)
    width_half = akg.lang.cce.vmuls(width, akg.tvm.const(0.5, dtype))
    xcenter = akg.lang.cce.vadd(xmin, width_half)

    # encode
    height_a = akg.topi.broadcast_to(height_a_raw, anchor_samples_shape)
    width_a = akg.topi.broadcast_to(width_a_raw, anchor_samples_shape)
    ycenter_a = akg.topi.broadcast_to(ycenter_a_raw, anchor_samples_shape)
    xcenter_a = akg.topi.broadcast_to(xcenter_a_raw, anchor_samples_shape)
    epsilon_ = akg.lang.cce.broadcast(akg.tvm.const(epsilon, dtype), anchor_samples_shape)
    h_a = akg.lang.cce.vadd(height_a, epsilon_)
    w_a = akg.lang.cce.vadd(width_a, epsilon_)
    h = akg.lang.cce.vadd(height, epsilon_)
    w = akg.lang.cce.vadd(width, epsilon_)
    xc_sub_xc_a = akg.lang.cce.vsub(xcenter, xcenter_a)
    w_a_rec = akg.lang.cce.vrec(w_a)
    h_a_rec = akg.lang.cce.vrec(h_a)
    tx = akg.lang.cce.vmul(xc_sub_xc_a, w_a_rec)
    yc_sub_yc_a = akg.lang.cce.vsub(ycenter, ycenter_a)
    ty = akg.lang.cce.vmul(yc_sub_yc_a, h_a_rec)
    h_div_h_a = akg.lang.cce.vmul(h, h_a_rec)
    th = akg.lang.cce.vlog(h_div_h_a)
    w_div_w_a = akg.lang.cce.vmul(w, w_a_rec)
    tw = akg.lang.cce.vlog(w_div_w_a)
    if scale_factors:
        ty = akg.lang.cce.vmuls(ty, akg.tvm.const(scale_factors[0], dtype))
        tx = akg.lang.cce.vmuls(tx, akg.tvm.const(scale_factors[1], dtype))
        th = akg.lang.cce.vmuls(th, akg.tvm.const(scale_factors[2], dtype))
        tw = akg.lang.cce.vmuls(tw, akg.tvm.const(scale_factors[3], dtype))
    output = akg.topi.stack([ty, tx, th, tw], axis=-1)
    return output
