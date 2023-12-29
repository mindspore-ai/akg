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

import numpy as np
import akg.tvm
from akg.utils import kernel_exec as utils
from tests.common.test_op.ascend import globalavgpool
from akg.topi.util import get_const_tuple
from tests.common.tensorio import compare_tensor


def globalavgpool_run(n, c, h, w, pool_type, dtype, attrs):
    dshape = (n, c, h, w)

    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(globalavgpool.globalavgpool, [dshape], [pool_type],
                                  kernel_name=kernel_name, attrs=attrs, tuning=t)
        if t:
            args, exp_output, input = gen_data(c, dshape, dtype, h, n, pool_type, w)
            return mod, exp_output, args
        else:
            return mod
    else:
        # Result_globalavgpool
        mod = globalavgpool.globalavgpool(n, c, h, w, pool_type, attrs)
        args, exp_output, input = gen_data(c, dshape, dtype, h, n, pool_type, w)
        acu_output = utils.mod_launch(mod, args, expect=exp_output)
        # compare result
        TestCase_Result = compare_tensor(acu_output, exp_output, rtol=5e-03, equal_nan=True)

        return input, acu_output, exp_output, TestCase_Result

        """
        ctx = akg.tvm.ndarray.cce(0)
        out_arg = akg.tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=dtype), ctx)
        arg = akg.tvm.nd.array(data, ctx)
     
        mod(arg, out_arg)
        ctx.sync()
        """


def gen_data(c, dshape, dtype, h, n, pool_type, w):
    # Result_Numpy
    input = np.random.poisson(1, size=dshape).astype(dtype)
    exp_output = np.mean(input, axis=(2, 3), keepdims=True)
    print("-------------exp_output-------------: ", exp_output)
    # inputs and output to hold the data
    A = akg.tvm.placeholder((n, c, h, w), name='A', dtype="float16")
    B = akg.topi.nn.global_pool(A, pool_type=pool_type)
    output = np.zeros(get_const_tuple(B.shape), dtype=dtype)
    args = [input, output]
    return args, exp_output, input