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
from test_op import roipool
from tensorio import compare_tensor
from gen_random import random_gaussian

def roipool_run(shape, roibox, pooled_shape, dtype, attrs, cce_path="./"):
    mod, output_shape = roipool.roipool(shape, roibox, pooled_shape, dtype, attrs=attrs)
    input1 = random_gaussian(shape1, miu=1, sigma=0.1)

    if (dtype == "int32"):
        input1 = input1.astype(np.int32)
    elif (dtype == "float16"):
        input1 = input1.astype(np.float16)
    elif (dtype == "float32"):
        input1 = input1.astype(np.float32)

    expect = roipool_expect(input1, shape, roibox, pooled_shape)

    # source_code = mod.imported_modules[0].get_source()
    # utils.create_cce(kernel_name, cce_path, source_code)

    output = np.full(output_shape, np.nan, dtype)
    output = utils.mod_launch(mod, (input1, output), expect=expect)

    return (input1,), output, expect, compare_tensor(output, expect, rtol=5e-03, equal_nan=True)


def roipool_expect(input_, shape, roibox, pooled_shape):
    output_shape = [shape[0], shape[1], pooled_shape[0], pooled_shape[1]]
    crop_shape = [shape[0], shape[1], roibox[1] - roibox[0], roibox[3] - roibox[2]]
    crop = np.zeros(crop_shape, dtype)
    for (n, c, h, w), _ in np.ndenumerate(crop):
        crop[n, c, h, w] = input_[n, c, roibox[0] + h, roibox[2] + w]

    p_h, p_w = pooled_shape
    win_h = (roibox[1] - roibox[0]) // p_h + (1 if (roibox[1] - roibox[0]) % p_h > 0 else 0)
    win_w = (roibox[3] - roibox[2]) // p_w + (1 if (roibox[3] - roibox[2]) % p_w > 0 else 0)
    unpooled_shape = list(shape[:2]) + list(pooled_shape) + [win_h, win_w]
    unpooled = np.zeros(unpooled_shape, dtype)
    for (n, c, h, w, wh, ww), _ in np.ndenumerate(unpooled):
        if h * win_h + wh < (roibox[1] - roibox[0]) and w * win_w + ww < (roibox[3] - roibox[2]):
            unpooled[n, c, h, w, wh, ww] = crop[n, c, h * win_h + wh, w * win_w + ww]
        else:
            unpooled[n, c, h, w, wh, ww] = 0

    expect = np.amax(unpooled, axis=(-1, -2))
    return expect
