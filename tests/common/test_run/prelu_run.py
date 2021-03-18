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

from akg.utils import kernel_exec as utils
import numpy as np
from akg.topi.util import get_const_tuple
from tests.common.test_op import prelu
from tests.common.tensorio import compare_tensor
from tests.common.gen_random import random_gaussian

def prelu_run(shape, w_shape, dtype, rtol, attrs):
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(prelu.prelu, [shape, w_shape], [dtype, dtype], kernel_name=kernel_name, attrs=attrs, tuning=t)
        if t:
            expect, input_data, w_data = gen_data(dtype, shape, w_shape)
            return mod, expect, (input_data, w_data, expect)
        else:
            return mod
    else:
        mod = utils.op_build_test(prelu.prelu, [shape, w_shape], [dtype, dtype], kernel_name='prelu', attrs=attrs)
        expect, input_data, w_data = gen_data(dtype, shape, w_shape)
        output = utils.mod_launch(mod, (input_data, w_data, expect), expect=expect)

        # #ctx.sync()
        # reshape_output = output_b.reshape(output_b.size)
        # reshape_output_np = output_np.reshape(output_np.size)
        # errorcount = 0
        # for i in range(reshape_output.size):
        #     limitError = abs(reshape_output[i] * rtol)
        #     if abs(reshape_output[i] - reshape_output_np[i]) > limitError:
        #         errorcount += 1
        return (input_data, w_data), output, expect, compare_tensor(output, expect, rtol=rtol)


def gen_data(dtype, shape, w_shape):
    # input_data = random_gaussian(shape, miu=1, sigma=50.0).astype(dtype.lower())
    input_data = np.random.uniform(low=-1.0, high=1.0, size=get_const_tuple(shape)).astype(dtype)
    w_data = random_gaussian(w_shape, miu=1, sigma=2.0).astype(dtype.lower())
    # expect = input_data * (input_data > 0) + input_data * (input_data < 0) * w_data[0]
    if w_shape[0] == 1:
        # pass
        expect = input_data * (input_data > 0) + input_data * (input_data < 0) * w_data[0]
    else:
        w_reshape = w_data.reshape(1, w_shape[0], 1, 1)
        w_broadcast = np.broadcast_to(w_reshape, shape)
        expect = input_data * (input_data > 0) + input_data * (input_data < 0) * w_broadcast
        # pass
        # for j in range(shape[1]):
        #     for i in range(shape[0]):
        #         for k in range(shape[2]):
        #             for l in range(shape[3]):
        #                 expect[i, j, k, l] = input_data[i, j, k, l] * (input_data[i, j, k, l] > 0) + input_data[i, j, k, l] * (input_data[i, j, k, l] < 0) * w_data[j]
    return expect, input_data, w_data
