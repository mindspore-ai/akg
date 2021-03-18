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

"""
mul run define
"""

import numpy as np
from akg.utils import kernel_exec as utils
from akg.ops.math import mul
from tests.common.tensorio import compare_tensor
from tests.common.gen_random import random_gaussian
from tests.common.base import get_rtol_atol


def mul_run(shapes, dtype, attrs):
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(mul.mul, shapes, [dtype, dtype], kernel_name=kernel_name, attrs=attrs, tuning=t)
        if t:
            expect, lhd, output, rhd = gen_data(dtype, shapes)
            return mod, expect, (lhd, rhd, output)
        else:
            return mod
    else:
        mod = utils.op_build_test(mul.mul, shapes, [dtype, dtype], kernel_name='mul', attrs=attrs)
        expect, lhd, output, rhd = gen_data(dtype, shapes)
        output = utils.mod_launch(mod, (lhd, rhd, output), expect=expect)
        rtol, atol = get_rtol_atol("mul", dtype)
        return (lhd, rhd), output, expect, compare_tensor(output, expect, rtol=rtol, atol=atol, equal_nan=True)


def gen_data(dtype, shapes):
    support_list = {"float16": np.float16, "float32": np.float32}
    if not (dtype.lower() in support_list):
        raise RuntimeError("Will not gen data because tile_cce only support %s while dtype is %s" % (
            ",".join(support_list.keys()), dtype))

    # Generate data for testing the op
    lhd = random_gaussian(shapes[0], miu=1, sigma=0.1).astype(support_list[dtype])
    rhd = random_gaussian(shapes[1], miu=1, sigma=0.1).astype(support_list[dtype])
    expect = np.multiply(lhd, rhd)
    out_shape = expect.shape
    output = np.full(out_shape, np.nan, dtype)
    return expect, lhd, output, rhd
