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

"""minimum_run"""
import numpy as np
from akg.utils import kernel_exec as utils
from akg.ops.math import minimum
from akg.utils import validation_check as vc_util
from tests.common.tensorio import compare_tensor
from tests.common.gen_random import random_gaussian
from akg.utils.format_transform import to_tvm_nd_array
from akg.utils.result_analysis import target_profiling

def minimum_run(shape1, shape2, dtype, attrs_op=None, attrs=None):
    """minimum_run"""
    if attrs_op is not None:
        if attrs is not None:
            attrs.update(attrs_op)
        else:
            attrs = attrs_op
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(minimum, [shape1, shape2], [dtype, dtype],
                                  kernel_name=kernel_name, attrs=attrs, tuning=t)
        if t:
            expect, lhd, output, rhd = gen_data(dtype, shape1, shape2)
            return mod, expect, (lhd, rhd, output)
        return mod
    else:
        mod = utils.op_build_test(minimum, [shape1, shape2], [dtype, dtype], kernel_name='minimum', attrs=attrs)
        expect, lhd, output, rhd = gen_data(dtype, shape1, shape2)
        # result_tvm
        output = utils.mod_launch(mod, (lhd, rhd, output), expect=expect)
        if attrs.get("profiling", False):
            import akg
            target_name = attrs["target"].split()[0]
            args_list = to_tvm_nd_array([lhd, rhd, output], akg.tvm.context(target_name, 0))
            target_profiling(mod, *args_list, target=target_name, repeat_time=attrs["repeat_times"])
        # compare result
        compare_result = compare_tensor(output, expect, rtol=5e-03, equal_nan=True)

        return (lhd, rhd), output, expect, compare_result


def gen_data(dtype, shape1, shape2):
    """gen_data"""
    # check shapes
    vc_util.check_shape(shape1)
    vc_util.check_shape(shape2)
    # check types
    support_list = {"float16": np.float16, "float32": np.float32, "int32": np.int32, "int8": np.int8, "uint8": np.uint8}
    if not dtype.lower() in support_list:
        raise RuntimeError("minimum_cce only support %s while dtype is %s" % (",".join(support_list.keys()), dtype))

    # Result_Numpy
    lhd = random_gaussian(shape1, miu=1, sigma=1).astype(support_list[dtype])
    rhd = random_gaussian(shape2, miu=1, sigma=1).astype(support_list[dtype])
    expect = np.minimum(lhd, rhd)
    # inputs and output to hold the data
    output = np.full(shape1, np.nan, dtype)
    return expect, lhd, output, rhd
