# Copyright 2020 Huawei Technologies Co., Ltd
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

"""run function for xlogy_grad"""

import numpy as np
from tests.common.tensorio import compare_tensor
from akg.utils import kernel_exec as utils
from tests.common.test_op import xlogy_grad
from tests.common.gen_random import random_gaussian
from tests.common.base import get_rtol_atol
from akg.utils.dsl_create import produce_shapes


def xlogy_grad_run(shape1, shape2, dtype, attrs):
    _, _, grad_shape = produce_shapes(shape1, shape2)
    mod = utils.op_build_test(xlogy_grad.xlogy_grad,
                              [shape1, shape2, grad_shape],
                              [dtype, dtype, dtype],
                              kernel_name="xlogy_grad", attrs=attrs)
    expects, inputs, outputs = gen_data(shape1, shape2, dtype)
    reses = utils.mod_launch(
        mod, (*inputs, *outputs), expect=expects,
        outputs=(-2, -1))

    rtol, atol = get_rtol_atol("xlogy_grad", dtype)
    TestCase_Results = list(map(lambda x, y: compare_tensor(
        x, y, rtol=rtol, atol=atol, equal_nan=True), reses, expects))

    return inputs, reses, expects, all(TestCase_Results)


def gen_data(shape1, shape2, dtype):
    x1 = random_gaussian(shape1, miu=1, sigma=0.3).astype(dtype)
    x2 = np.abs(random_gaussian(shape2, miu=1, sigma=0.3)).astype(dtype)
    shape1, shape2, fout_shape = produce_shapes(shape1, shape2)
    dy = random_gaussian(fout_shape, miu=1, sigma=0.3).astype(dtype)
    rx, ry = xlogy_grad.broadcast_gradient_args(shape1, shape2)
    dx1_bc = np.where(np.equal(x1, 0.),
                      np.zeros_like(np.multiply(x1, x2)),
                      np.log(x2)) * dy
    dx2_bc = np.where(np.equal(x1, 0.),
                      np.zeros_like(np.multiply(x1, x2)),
                      np.divide(x1, x2)) * dy
    dx1 = (np.sum(
        dx1_bc.astype("float64"),
        axis=tuple(rx),
        keepdims=True) if len(rx) > 0 else dx1_bc).astype(dtype)
    dx2 = (np.sum(
        dx2_bc.astype("float64"),
        axis=tuple(ry),
        keepdims=True) if len(ry) > 0 else dx2_bc).astype(dtype)
    output1 = np.full(shape1, np.nan, dtype)
    output2 = np.full(shape2, np.nan, dtype)
    return (dx1, dx2), (x1, x2, dy), (output1, output2)
