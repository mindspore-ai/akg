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
from tests.common.tensorio import compare_tensor
import akg
import akg.tvm
from akg.utils import kernel_exec as utils
from tests.common.test_op.ascend import dynamic_stitch
from tests.common.gen_random import random_gaussian


def dynamic_stitch_run(indices_shape, data_shape, indices_dtype, dtype, attrs):
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(dynamic_stitch.dynamic_stitch, [indices_shape, data_shape], [indices_dtype, dtype],
                                  kernel_name=kernel_name, attrs=attrs, tuning=t)
        if t:
            data_input, expect, indicies_input, output = gen_data(data_shape, dtype, indices_dtype, indices_shape)
            return mod, expect, (indicies_input, data_input, output)
        else:
            return mod
    else:
        mod = utils.op_build_test(dynamic_stitch.dynamic_stitch, [indices_shape, data_shape], [indices_dtype, dtype],
                                  kernel_name='dynamic_stitch', attrs=attrs)
        data_input, expect, indicies_input, output = gen_data(data_shape, dtype, indices_dtype, indices_shape)
        output = utils.mod_launch(mod, (indicies_input, data_input, output), expect=expect)
        return (indicies_input, data_input), output, expect, compare_tensor(output, expect, rtol=5e-03, equal_nan=True)


def gen_data(data_shape, dtype, indices_dtype, indices_shape):
    if not (indices_dtype.lower() in "int32"):
        raise RuntimeError("indices_dtype only support int32 while dtype is %s" % indices_dtype)
    support_list = {"float16": np.float16, "float32": np.float32, "int32": np.int32}
    if not (dtype.lower() in support_list):
        raise RuntimeError(
            "dynamic_stitch_cce only support %s while dtype is %s" % (",".join(support_list.keys()), dtype))
    length = 1
    for l in indices_shape:
        length *= l
    indicies_input = np.arange(length).reshape(indices_shape)
    indicies_input = np.random.permutation(indicies_input).astype(np.int32)
    if support_list[dtype] == np.int32:
        data_input = np.random.randint(length, size=data_shape)
    else:
        data_input = random_gaussian(data_shape, miu=1, sigma=0.1).astype(support_list[dtype])
    output_shape = [length] + data_shape[len(indices_shape):]
    output = np.full(output_shape, 0, support_list[dtype])
    expect = np.full(output_shape, 0.0, support_list[dtype])
    for index, i in np.ndenumerate(indicies_input):
        expect[i] = data_input[index]
    return data_input, expect, indicies_input, output
