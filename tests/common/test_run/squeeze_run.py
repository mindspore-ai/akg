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

import numpy as np
from akg.utils import kernel_exec as utils
from test_op import squeeze

from tensorio import compare_tensor
from akg import tvm

def squeeze_run(shape, axis, dtype, kernel_name="squeeze", attrs=None):
    op_attrs = [axis]
    if attrs is not None and attrs.get("dynamic"):
        build_shape = []
        for i in range(len(shape)):
            build_shape.append(tvm.var("I" + str(i)))
    else:
        build_shape = shape
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(squeeze.squeeze, [build_shape], [dtype], op_attrs, kernel_name=kernel_name, attrs=attrs, tuning=t)
        if t:
            expect, input, output = gen_data(axis, dtype, shape)
            return mod, expect, (input, output)
        else:
            return mod
    else:
        expect, input, output = gen_data(axis, dtype, shape)
        mod = utils.op_build_test(squeeze.squeeze, [build_shape], [dtype], op_attrs, kernel_name=kernel_name, attrs=attrs)
        args = [input, output]
        if attrs is not None and attrs.get("dynamic"):
            for i in range(len(shape)):
                args.append(shape[i])
        output = utils.mod_launch(mod, args, outputs=(1,), expect=expect)
        return input, output, expect, compare_tensor(output, expect, rtol=5e-03, equal_nan=True)


def gen_data(axis, dtype, shape):
    input = np.random.randint(100, size=shape).astype(dtype)
    expect = np.squeeze(input, axis=axis)
    output = np.full(expect.shape, np.nan, dtype)
    return expect, input, output
