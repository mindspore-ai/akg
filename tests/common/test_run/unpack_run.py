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

"""run function for unpack"""

import numpy as np
from tests.common.tensorio import compare_tensor
from akg.utils import kernel_exec as utils
from tests.common.test_op import unpack
from tests.common.gen_random import random_gaussian
from tests.common.base import get_rtol_atol


def unpack_run(shape, dtype, tensor_format, num, axis, attrs):
    """run function for unpack"""
    mod = utils.op_build_test(unpack.unpack, [shape], [dtype],
                              op_attrs=[tensor_format, num, axis],
                              kernel_name="unpack", attrs=attrs)
    data, expects, out_bufs = gen_data(shape, dtype, axis)
    outputs = utils.mod_launch(mod, (data, *out_bufs), expect=expects,
                               outputs=list(range(-len(out_bufs), 0)))
    rtol, atol = get_rtol_atol("unpack", dtype)

    cmp_res = list(map(lambda x, y:
                       compare_tensor(x, y, rtol=rtol, atol=atol),
                       outputs, expects))

    return data, outputs, expects, all(cmp_res)


def gen_data(shape, dtype, axis):
    """generate valid data for unpack"""
    axis = axis if axis >= 0 else len(shape) + axis
    data = random_gaussian(shape, miu=1, sigma=0.3).astype(dtype)
    expects = np.split(data, list(range(data.shape[axis]))[1:], axis=axis)
    out_bufs = []
    out_shape = list(shape)
    out_shape[axis] = 1
    for _ in range(shape[axis]):
        out_bufs.append(np.full(out_shape, np.nan, dtype))
    return data, tuple(expects), tuple(out_bufs)
