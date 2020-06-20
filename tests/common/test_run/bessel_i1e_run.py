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

import numpy as np
from scipy import special as sp

from akg.utils import kernel_exec as utils
from test_op.bessel_i1e import bessel_i1e
from gen_random import random_gaussian
from base import get_rtol_atol
from tensorio import compare_tensor

def bessel_i1e_run(x_shape, x_dtype, attrs):
    shapes = [x_shape]
    dtypes = [x_dtype]
    mod = utils.op_build_test(bessel_i1e, shapes, dtypes,
                              kernel_name="bessel_i1e", attrs=attrs)
    bench_mark, inputs, output = gen_data(dtypes, shapes)
    output = utils.mod_launch(mod, inputs + [output], expect=bench_mark)
    rtol, atol = get_rtol_atol("bessel_i1e", dtypes[0])
    compare_res = compare_tensor(output, bench_mark, rtol=rtol, atol=atol)
    return inputs, output, bench_mark, compare_res


def gen_data(dtypes, shapes):
    inputs = []
    for dtype, shape in zip(dtypes, shapes):
        input = random_gaussian(shape, miu=3.75).astype(dtype)
        inputs.append(input)
    expect = sp.i1e(input)
    output = np.full(expect.shape, np.nan, dtypes[0])
    return expect, inputs, output
