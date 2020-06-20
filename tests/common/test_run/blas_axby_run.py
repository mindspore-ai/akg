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
from tensorio import compare_tensor
from akg.utils import kernel_exec as utils
from test_op import blas_axby


def blas_axby_run(shape, dtype, kernel_name, attrs):
    alpha = 2.0
    beta = 3.0

    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(blas_axby.blas_axby, [shape, shape], [dtype, dtype], [alpha, beta],
                                  kernel_name=kernel_name, attrs=attrs, tuning=t)
        if t:
            args, expect, input = gen_data(alpha, beta, dtype, shape)
            return mod, expect, args
        else:
            return mod
    else:
        mod = utils.op_build_test(blas_axby.blas_axby, [shape, shape], [dtype, dtype], [alpha, beta],
                                  kernel_name=kernel_name, attrs=attrs)
        args, expect, input = gen_data(alpha, beta, dtype, shape)
        output = utils.mod_launch(mod, args, expect=expect)
        result = compare_tensor(expect, output, rtol=1e-03, atol=1e-03, equal_nan=True)

        return (input, input), expect, output, result


def gen_data(alpha, beta, dtype, shape):
    support_list = {"float16": np.float16, "float32": np.float32}
    input = np.random.normal(size=shape).astype(support_list[dtype])
    x = np.multiply(input, alpha)
    y = np.multiply(input, beta)
    expect = np.add(x, y)
    output = np.full(x.shape, np.nan, dtype)
    args = [input, input, output]
    return args, expect, input
