# Copyright 2019-2021 Huawei Technologies Co., Ltd
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

"""reduce_min_run"""

import numpy as np
from tests.common.tensorio import compare_tensor
from akg.utils import kernel_exec as utils
from akg.ops.math import reduce_min
from akg.utils.dsl_create import get_reduce_out_shape
from tests.common.gen_random import random_gaussian
from tests.common.base import get_rtol_atol
from akg.utils.result_analysis import target_profiling
from akg.utils.format_transform import to_tvm_nd_array

def reduce_min_run(shape, dtype, axis, keepdims, kernel_name="reduce_min", attrs=None):
    """run function for dsl function reduce_min."""
    default_attrs = { "polytops_parameter_shifting": True }
    if attrs is None:
        attrs = {}
    attrs.update(default_attrs)

    op_attrs = [axis, keepdims]

    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(reduce_min, [shape], [dtype], 
                                  op_attrs=op_attrs, kernel_name=kernel_name, attrs=attrs, tuning=t)
        if t:
            expect, inputs, output = gen_data(axis, dtype, keepdims, shape)
            return mod, expect, (inputs, output)

        return mod

    mod = utils.op_build_test(reduce_min, [shape], [dtype], 
                              op_attrs=op_attrs, kernel_name=kernel_name, attrs=attrs)
    expect, inputs, output = gen_data(axis, dtype, keepdims, shape)
    output = utils.mod_launch(mod, (inputs, output), expect=expect)
    if attrs.get("profiling", False):
            import akg
            target_name = attrs["target"].split()[0]
            args_list = to_tvm_nd_array([inputs, output], akg.tvm.context(target_name, 0))
            target_profiling(mod, *args_list, target=target_name, repeat_time=attrs["repeat_times"])
    rtol, atol = get_rtol_atol("reduce_min", dtype)
    return inputs, output, expect, compare_tensor(output, expect, rtol=rtol, atol=atol, equal_nan=True)


def gen_data(axis, dtype, keepdims, shape):
    """Generates input, output and expect data."""
    inputs = random_gaussian(shape, miu=0, sigma=100.0).astype("float16").astype(dtype.lower())
    expect = np.amin(inputs, axis=axis, keepdims=keepdims)
    if axis==None and keepdims==False:
        expect = np.broadcast_to(expect, (1,))
    out_shape = get_reduce_out_shape(shape, axis=axis, keepdims=keepdims)
    output = np.full(out_shape, 3.402823e38, dtype)
    return expect, inputs, output
