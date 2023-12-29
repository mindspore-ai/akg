# Copyright 2019-2022 Huawei Technologies Co., Ltd
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
import akg
import numpy as np
from akg import tvm
from akg.utils import kernel_exec as utils
from akg.ops.math import cast
from tests.common.tensorio import compare_tensor
from tests.common.base import get_rtol_atol
from tests.common.test_utils import compute_blockdim
from akg.utils.result_analysis import target_profiling
from akg.utils.format_transform import to_tvm_nd_array


def cast_run(shape, srcType, dstType, attrs={}):
    op_attrs = [dstType]
    if attrs.get("dynamic"):
        attrs["enable_double_buffer"] = False
        var_shape = []
        for i in range(len(shape)):
            var_shape.append(tvm.var("I" + str(i)))
        build_shape = var_shape
    else:
        build_shape = shape

    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(cast, [build_shape], [srcType], op_attrs, kernel_name=kernel_name, attrs=attrs, tuning=t)
        if t:
            args, exp_output, input = gen_data(dstType, shape, srcType)
            return mod, exp_output, args
        else:
            return mod
    else:
        mod = utils.op_build_test(cast, [build_shape], [srcType], op_attrs, kernel_name='cast', attrs=attrs)
        args, exp_output, input = gen_data(dstType, shape, srcType)
        if attrs.get("dynamic"):
            for i in range(len(shape)):
                args.append(shape[i])
            block_dim = compute_blockdim(shape)
            args.append(block_dim)
        acu_output = utils.mod_launch(mod, args, outputs=(1,), expect=exp_output)
        # compare result
        rtol, atol = get_rtol_atol("cast", dstType)
        TestCase_Result = compare_tensor(acu_output, exp_output, rtol=rtol, atol=atol, equal_nan=True)
        
        if attrs.get("profiling", False):
            target_name = attrs["target"].split()[0]
            args_list = to_tvm_nd_array(args, akg.tvm.context(target_name, 0))
            target_profiling(mod, *args_list, target=target_name, repeat_time=attrs["repeat_times"])
        return input, acu_output, exp_output, TestCase_Result

def gen_data(dstType, shape, srcType):
    # Result_Numpy
    if srcType == 'int8':
        low_bound = -128
        high_bound = 127
    elif srcType == 'int32':
        low_bound = -1000
        high_bound = 1000
    else:
        low_bound = -1.0
        high_bound = 1.0

    input = np.random.uniform(low=low_bound, high=high_bound, size=tuple(shape)).astype(srcType)
    exp_output = input.astype(dstType, copy=True)

    # inputs and output to hold the data
    output = np.full(exp_output.shape, np.nan, dstType)
    args = [input, output]
    return args, exp_output, input
