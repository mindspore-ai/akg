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

"""dense_run"""

import numpy as np
from akg.utils import kernel_exec as utils
from test_op import dense
from tensorio import compare_tensor
from gen_random import random_gaussian

def dense_run(batch, in_dim, out_dim, dtype, bias, attrs):
    """run function for dsl function dense."""
    op_attrs = [bias]

    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        d1 = random_gaussian((batch, in_dim), miu=1, sigma=0.1).astype(dtype)
        w1 = random_gaussian((out_dim, in_dim), miu=1, sigma=0.1).astype(dtype)
        w2 = w1.transpose().copy()

        if bias:
            b1 = random_gaussian((out_dim), miu=1, sigma=0.1).astype(dtype)
            mod = utils.op_build_test(dense.dense, [d1.shape, w1.shape, b1.shape], [dtype, dtype, dtype], op_attrs,
                                      kernel_name=kernel_name, attrs=attrs, tuning=t)
            if t:
                exp_output = np.dot(d1, w2)
                # inputs and output to hold the data
                output = np.full(exp_output.shape, np.nan, dtype)
                for o in range(out_dim):
                    exp_output[:, o] += b1[o]
                args = [d1, w1, b1, output]
                return mod, exp_output, args

            return mod
        else:
            mod = utils.op_build_test(dense.dense, [d1.shape, w1.shape], [dtype, dtype], op_attrs,
                                      kernel_name=kernel_name, attrs=attrs, tuning=t)
            if t:
                exp_output = np.dot(d1, w2)
                # inputs and output to hold the data
                output = np.full(exp_output.shape, np.nan, dtype)
                args = [d1, w1, output]
                return mod, exp_output, args
            else:
                return mod

    d1 = random_gaussian((batch, in_dim), miu=1, sigma=0.1).astype(dtype)
    w1 = random_gaussian((out_dim, in_dim), miu=1, sigma=0.1).astype(dtype)
    w2 = w1.transpose().copy()

    exp_output = np.dot(d1, w2)
    # inputs and output to hold the data
    output = np.full(exp_output.shape, np.nan, dtype)

    if bias:
        b1 = random_gaussian((out_dim), miu=1, sigma=0.1).astype(dtype)
        for o in range(out_dim):
            exp_output[:, o] += b1[o]
        inputs = [d1, w1, b1]
        args = [d1, w1, b1, output]
        mod = utils.op_build_test(dense.dense, [d1.shape, w1.shape, b1.shape], [dtype, dtype, dtype], op_attrs,
                                  kernel_name='dense', attrs=attrs)
    else:
        inputs = [d1, w1]
        args = [d1, w1, output]
        mod = utils.op_build_test(dense.dense, [d1.shape, w1.shape], [dtype, dtype], op_attrs, kernel_name='dense',
                                  attrs=attrs)
    acu_output = utils.mod_launch(mod, args, expect=exp_output)

    # compare result
    compare_result = compare_tensor(acu_output, exp_output, rtol=5e-03, equal_nan=True)

    return inputs, acu_output, exp_output, compare_result
