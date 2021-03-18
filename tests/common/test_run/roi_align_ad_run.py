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
from akg.utils import kernel_exec as utils
from tests.common.test_op.roi_align_ad import roi_align_ad
from tests.common.gen_random import random_gaussian

def roi_align_ad_run(data_shape, num_rois, dtype, pooled_size, spatial_scale, sample_ratio, attrs=None):
    rois_shape = (num_rois, 5)
    pooled_size_h = 0.0
    pooled_size_w = 0.0
    if isinstance(pooled_size, int):
        pooled_size_h = pooled_size
        pooled_size_w = pooled_size
        e_shape = (num_rois, data_shape[1], pooled_size_h, pooled_size_w)
    else:
        pooled_size_h, pooled_size_w = pooled_size
        e_shape = (num_rois, data_shape[1], pooled_size_h, pooled_size_w)

    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(roi_align_ad, [e_shape, data_shape, rois_shape], [dtype, dtype, dtype],
                                  kernel_name=kernel_name, op_attrs=[pooled_size, spatial_scale, sample_ratio],
                                  attrs=attrs, tuning=t)
        if t:
            expect, head, inputs, output, rois = gen_data(data_shape, dtype, e_shape, num_rois, rois_shape)
            return mod, expect, (head, inputs, rois, output)
        else:
            return mod
    else:
        expect, head, inputs, output, rois = gen_data(data_shape, dtype, e_shape, num_rois, rois_shape)
        mod = utils.op_build_test(roi_align_ad, [e_shape, data_shape, rois_shape], [dtype, dtype, dtype],
                                  kernel_name="roi_align_ad_run", op_attrs=[pooled_size, spatial_scale, sample_ratio],
                                  attrs=attrs)
        output = utils.mod_launch(mod, [head, inputs, rois, output], expect=expect)

        return [head, inputs, rois], output, expect, np.allclose(output, expect, rtol=5e-03, atol=0.1, equal_nan=True)


def gen_data(data_shape, dtype, e_shape, num_rois, rois_shape):
    rois = np.random.uniform(0.0, 0.1, size=rois_shape)
    for x in range(num_rois):
        rois[x][0] = np.random.randint(data_shape[0], size=1)
        rois[x][1] = np.random.uniform(low=0.0, high=data_shape[3], size=1)
        rois[x][2] = np.random.uniform(low=0.0, high=data_shape[2], size=1)
        rois[x][3] = np.random.uniform(low=rois[x][1], high=data_shape[3], size=1)
        rois[x][4] = np.random.uniform(low=rois[x][2], high=data_shape[2], size=1)
    rois = rois.astype(dtype)
    inputs = random_gaussian(data_shape, miu=1, sigma=0.1).astype(dtype)
    head = random_gaussian(e_shape, miu=1, sigma=0.1).astype(dtype)
    output = np.full(data_shape, 1.0, dtype)
    expect = output
    return expect, head, inputs, output, rois
