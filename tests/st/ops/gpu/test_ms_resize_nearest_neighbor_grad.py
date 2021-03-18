# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
# limitations under the License
import numpy as np
from akg.utils import kernel_exec as utils
from akg.ops.nn.resize_nearest_neighbor_grad import resize_nearest_neighbor_grad
from tests.common.gen_random import random_gaussian
from tests.common.tensorio import compare_tensor


def resize_nearest_grad(grad, size, align_corners, dtype):
    inshape = grad.shape

    if align_corners:
        scale_h = (inshape[2] - 1) / (size[0] - 1)
        scale_w = (inshape[3] - 1) / (size[1] - 1)

    else:
        scale_h = inshape[2] / size[0]
        scale_w = inshape[3] / size[1]

    oshape = (inshape[0], inshape[1], size[0], size[1])
    output = np.full(oshape, np.nan, dtype)

    for n in range(oshape[0]):
        for c in range(oshape[1]):
            for h in range(oshape[2]):
                for w in range(oshape[3]):
                    if align_corners:
                        in_h = int(round(scale_h * h));
                        in_w = int(round(scale_w * w));
                    else:
                        epsilon = 1e-5
                        in_h = int(floor(scale_h * h));
                        in_w = int(floor(scale_w * w));

                    in_h = max(min(in_h, inshape[2] - 1), 0)
                    in_w = max(min(in_w, inshape[3] - 1), 0)
                    output[n, c, h, w] = grad[n, c, in_h, in_w]
    return output


def gen_data(shape, size, align_corners, dtype):
    support_list = {"float16": np.float16, "float32": np.float32}
    grad = random_gaussian(shape, miu=1, sigma=0.1).astype(support_list[dtype])
    expect = resize_nearest_grad(grad, size, align_corners, dtype)
    outshape = [shape[0], shape[1], size[0], size[1]]
    output = np.full(outshape, np.nan, dtype)
    return grad, output, expect
