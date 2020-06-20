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
from gen_random import random_gaussian
from tensorio import compare_tensor
from test_op.matrix_diag import matrix_diag


def matrix_diag_run(shape, out_shape, dtype, attrs=None):
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(matrix_diag, [shape], [dtype], op_attrs=[out_shape],
                                  kernel_name=kernel_name, attrs=attrs, tuning=t)
        if t:
            expect, input, output = gen_data(dtype, out_shape, shape)
            return mod, expect, (input, output)
        else:
            return mod
    else:
        mod = utils.op_build_test(matrix_diag, [shape], [dtype], op_attrs=[out_shape],
                                  kernel_name="matrix_diag", attrs=attrs)
        expect, input, output = gen_data(dtype, out_shape, shape)
        output = utils.mod_launch(mod, (input, output), expect=expect)
        result = compare_tensor(output, expect)
        return input, output, expect, result


def gen_data(dtype, out_shape, shape):
    input = random_gaussian(size=shape, miu=0, sigma=10).astype(dtype)
    expect = get_matrix_diag(input, out_shape)
    output = np.full(out_shape, np.nan, dtype)
    return expect, input, output


def get_matrix_diag(data, out_shape):
    res = np.zeros(out_shape, dtype=data.dtype)
    n = np.prod(res.shape[:-2]) if len(out_shape) > 2 else 1
    res = res.reshape(n, res.shape[-2], res.shape[-1])
    data = data.reshape(n, data.shape[-1])
    for i in range(res.shape[0]):
        for j in range(min(res.shape[1], res.shape[2], data.shape[1])):
            res[i][j][j] = data[i][j]
    return res.reshape(out_shape)
