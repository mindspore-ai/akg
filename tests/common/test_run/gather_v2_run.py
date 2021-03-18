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

from tests.common.base import get_rtol_atol
from tests.common.tensorio import compare_tensor
from akg.utils import kernel_exec as utils
from akg.ops.array import gather_v2
from tests.common.gen_random import random_gaussian
import numpy as np


def gen_data(shape1, dtype1, shape2, dtype2, axis):
    params = random_gaussian(shape1).astype(dtype1)
    indices = np.random.randint(
        low=0, high=shape1[axis], size=shape2).astype(dtype2)
    expect = np.take(params, indices, axis=axis)
    return params, indices, expect


def gather_v2_run(shape1, dtype1, shape2, dtype2, axis, attrs):
    op_attrs = [axis]
    mod = utils.op_build_test(gather_v2.gather_v2,
                              [shape1, shape2], [
                                  dtype1, dtype2], op_attrs=op_attrs,
                              kernel_name="gather_v2", attrs=attrs)
    params, indices, expect = gen_data(shape1, dtype1, shape2, dtype2, axis)
    output_shape = expect.shape
    if len(expect.shape) == 0:
        output_shape = (1,)
    output = np.full(output_shape, np.nan, expect.dtype)
    output = utils.mod_launch(mod, (params, indices, output), expect=expect)
    atol, rtol = get_rtol_atol("gather_v2", dtype1)
    compare_res = compare_tensor(output, expect, rtol=rtol, atol=atol)
    return (params, indices), output, expect, compare_res
