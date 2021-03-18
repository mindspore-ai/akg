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

import numpy as np
import akg.topi
import akg
from akg.utils import kernel_exec as utils
from tests.common.gen_random import random_gaussian

def bilinear_sample_nchw(inputss, indices, max_y, max_x):
    in_y = indices[2]
    yf = np.floor(in_y)
    yc = np.int32(np.ceil(in_y))

    y0 = np.int32(np.floor(in_y))
    if (yc > max_y) == True:
        y1 = max_y
    else:
        y1 = yc
    y_lerp = in_y - yf

    in_x = indices[3]
    xf = np.floor(in_x)
    xc = np.int32(np.ceil(in_x))

    x0 = np.int32(np.floor(in_x))
    if (xc > max_x) == True:
        x1 = max_x
    else:
        x1 = xc
    y_lerp = in_y - yf
    x_lerp = in_x - xf

    A = inputss[indices[0], indices[1], y0, x0]
    B = inputss[indices[0], indices[1], y0, x1]
    C = inputss[indices[0], indices[1], y1, x0]
    D = inputss[indices[0], indices[1], y1, x1]

    return (A * (1 - x_lerp) * (1 - y_lerp) + B * x_lerp * (1 - y_lerp) + C * (1 - x_lerp) * y_lerp + D * x_lerp * y_lerp)


def roi_align_run(data_shape, num_rois, dtype, pooled_size, spatial_scale, sample_ratio, attrs=None):
    rois_shape = (num_rois, 5)

    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(akg.topi.vision.rcnn.roi_align.roi_align_nchw, [data_shape, rois_shape], [dtype, dtype],
                                  kernel_name=kernel_name, op_attrs=[pooled_size, spatial_scale, sample_ratio],
                                  attrs=attrs, tuning=t)
        if t:
            e_shape, expect, inputs, output, pooled_size_h, pooled_size_w, rois = gen_data(data_shape, dtype, num_rois,
                                                                                           pooled_size, rois_shape)
            return mod, expect, (inputs, rois, output)
        else:
            return mod
    else:
        e_shape, expect, inputs, output, pooled_size_h, pooled_size_w, rois = gen_data(data_shape, dtype, num_rois,
                                                                                       pooled_size, rois_shape)
        mod = utils.op_build_test(akg.topi.vision.rcnn.roi_align.roi_align_nchw, [data_shape, rois_shape], [dtype, dtype],
                                  kernel_name="roi_align_run", op_attrs=[pooled_size, spatial_scale, sample_ratio],
                                  attrs=attrs)
        # print(mod.imported_modules[0].get_source())
        output = utils.mod_launch(mod, [inputs, rois, output], expect=expect)

        for i in range(e_shape[0]):
            for c in range(e_shape[1]):
                for ph in range(e_shape[2]):
                    for pw in range(e_shape[3]):
                        roi = rois[i]
                        batch_index = roi[0].astype("int32")
                        roi_start_w, roi_start_h, roi_end_w, roi_end_h = roi[1], roi[2], roi[3], roi[4]
                        roi_start_h *= spatial_scale
                        roi_end_h *= spatial_scale
                        roi_start_w *= spatial_scale
                        roi_end_w *= spatial_scale
                        # force malformed ROIs to be 1x1
                        roi_h = np.maximum(roi_end_h - roi_start_h, 1.0)
                        roi_w = np.maximum(roi_end_w - roi_start_w, 1.0)
                        bin_h = roi_h / pooled_size_h
                        bin_w = roi_w / pooled_size_w
                        if sample_ratio > 0:
                            roi_bin_grid_h = np.int32(sample_ratio)
                            roi_bin_grid_w = np.int32(sample_ratio)
                        else:
                            roi_bin_grid_h = np.ceil(roi_h / pooled_size_h).astype("int32")
                            roi_bin_grid_w = np.ceil(roi_w / pooled_size_w).astype("int32")
                        count = roi_bin_grid_h * roi_bin_grid_w

                        roi_start_h += ph * bin_h
                        roi_start_w += pw * bin_w
                        summ = 0
                        for rh in range(0, roi_bin_grid_h):
                            for rw in range(0, roi_bin_grid_w):
                                outside = (((roi_start_h + (rh + 0.5) * bin_h / roi_bin_grid_h) < -1.0) or (
                                    (roi_start_w + (rw + 0.5) * bin_w / roi_bin_grid_w) < -1.0)
                                    or ((roi_start_h + (rh + 0.5) * bin_h / roi_bin_grid_h) > data_shape[2]) or (
                                    (roi_start_w + (rw + 0.5) * bin_w / roi_bin_grid_w) > data_shape[3]))
                                y = np.maximum((roi_start_h + (rh + 0.5) * bin_h / roi_bin_grid_h), 0.0)
                                x = np.maximum((roi_start_w + (rw + 0.5) * bin_w / roi_bin_grid_w), 0.0)
                                val = bilinear_sample_nchw(inputs, (batch_index, c, y, x), data_shape[2] - 1,
                                                           data_shape[3] - 1)
                                if outside == True:
                                    summ += 0.0 / count
                                else:
                                    summ += val / count
                        expect[i][c][ph][pw] = summ

        return [inputs, rois], output, expect, np.allclose(output, expect, rtol=5e-03, atol=0.1, equal_nan=True)


def gen_data(data_shape, dtype, num_rois, pooled_size, rois_shape):
    inputs = random_gaussian(data_shape, miu=1, sigma=0.1).astype(dtype)
    # rois = np.array([[0.0,2.0,2.0,8.0,8.0],[0.0,4.0,4.0,12.0,12.0]]).astype(dtype)
    rois = np.random.uniform(0.0, 0.1, size=rois_shape)
    for x in range(num_rois):
        rois[x][0] = np.random.randint(data_shape[0], size=1)
        rois[x][1] = np.random.uniform(low=0.0, high=data_shape[3], size=1)
        rois[x][2] = np.random.uniform(low=0.0, high=data_shape[2], size=1)
        rois[x][3] = np.random.uniform(low=rois[x][1], high=data_shape[3], size=1)
        rois[x][4] = np.random.uniform(low=rois[x][2], high=data_shape[2], size=1)
    rois = rois.astype(dtype)
    pooled_size_h = 0.0
    pooled_size_w = 0.0
    if isinstance(pooled_size, int):
        pooled_size_h = pooled_size
        pooled_size_w = pooled_size
        e_shape = (num_rois, data_shape[1], pooled_size_h, pooled_size_w)
    else:
        pooled_size_h, pooled_size_w = pooled_size
        e_shape = (num_rois, data_shape[1], pooled_size_h, pooled_size_w)
    output = np.full(e_shape, 1.0, dtype)
    expect = random_gaussian(e_shape, miu=1, sigma=0.1)
    return e_shape, expect, inputs, output, pooled_size_h, pooled_size_w, rois
