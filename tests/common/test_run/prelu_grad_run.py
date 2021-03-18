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
from tests.common.test_op import prelu_grad
from tests.common.tensorio import compare_tensor
from tests.common.gen_random import random_gaussian

def prelu_grad_run(shape, w_shape, dtype, rtol, attrs):
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(prelu_grad.prelu_grad, [shape, shape, w_shape], [dtype, dtype, dtype],
                                  kernel_name=kernel_name, attrs=attrs, tuning=t)
        if t:
            dy, expect_dA, expect_dw, input_data, output_dA, output_dw, w_data = gen_data(dtype, shape, w_shape)
            return mod, (expect_dA, expect_dw), {"args": (dy, input_data, w_data, output_dA, output_dw),
                                                 'outputs': (-2, -1), 'tuning': False}
        else:
            return mod
    else:
        mod = utils.op_build_test(prelu_grad.prelu_grad, [shape, shape, w_shape], [dtype, dtype, dtype],
                                  kernel_name='prelu_grad', attrs=attrs)
        dy, expect_dA, expect_dw, input_data, output_dA, output_dw, w_data = gen_data(dtype, shape, w_shape)
        output_dA, output_dw = utils.mod_launch(mod, (dy, input_data, w_data, output_dA, output_dw), outputs=(-2, -1)
                                                , expect=(expect_dw, expect_dA))
        return (input_data, dy, w_data), (output_dw, output_dA), (expect_dw, expect_dA), \
            compare_tensor(output_dw, expect_dw, rtol=rtol) and compare_tensor(output_dA, expect_dA, rtol=rtol)


def gen_data(dtype, shape, w_shape):
    # input_data = -0.01 * np.ones(shape).astype(dtype)
    # dy = -0.01 * np.ones(shape).astype(dtype)
    input_data = random_gaussian(shape, miu=0, sigma=0.001).astype(dtype.lower())
    dy = random_gaussian(shape, miu=-0.01, sigma=0.001).astype(dtype.lower())
    # w_data = random_gaussian(w_shape, miu=0, sigma=1).astype(dtype.lower())
    # input_data = np.random.uniform(low=-1.0, high=1.0, size=shape).astype(dtype)
    # dy = np.random.uniform(low=-1.0, high=1.0, size=shape).astype(dtype)
    w_data = np.random.uniform(low=0, high=1.0, size=w_shape).astype(dtype)
    w_reshape = w_data.reshape(1, w_shape[0], 1, 1)
    w_broadcast = np.broadcast_to(w_reshape, shape)
    expect_dA = dy * (input_data >= 0) + dy * (input_data < 0) * w_broadcast
    # expect_dA = (dy.astype("float32") * (input_data >= 0) + dy.astype("float32") * (input_data < 0) * w_broadcast.astype("float32")).astype(dtype.lower())
    dw_intermediate = dy * (input_data < 0) * input_data
    if w_shape[0] == 1:
        # expect_dw = np.sum(dw_intermediate, keepdims = True, dtype=dtype)
        expect_dw = (np.sum(dw_intermediate, dtype="float32")).astype(dtype.lower())
        # expect_dw = np.sum(dw_intermediate, dtype=dtype)
        expect_dw = expect_dw.reshape(1)
    else:
        expect_dw = (np.sum(dw_intermediate, axis=(0, 2, 3), dtype="float32")).astype(dtype.lower())
        # expect_dw = np.sum(dw_intermediate, axis=(0,2,3), dtype=dtype)
        # expect_dw = np.sum(dw_intermediate.astype("float32"), axis=(0,2,3))
        # expect_dw = expect_dw.astype(dtype)
    output_dA = np.full(expect_dA.shape, np.nan, dtype)
    output_dw = np.full(expect_dw.shape, np.nan, dtype)
    return dy, expect_dA, expect_dw, input_data, output_dA, output_dw, w_data
