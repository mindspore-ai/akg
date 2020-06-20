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
from test_op import fill

from tensorio import compare_tensor


def fill_run(shape, value, dtype, attrs):
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(fill.fill, [], [], op_attrs=[shape, value, dtype], kernel_name=kernel_name, attrs=attrs, tuning=t)
        if t:
            expect, input, output = gen_data(dtype, shape, value)
            return mod, expect, (output,)
        else:
            return mod
    else:
        mod = utils.op_build_test(fill.fill, [], [], op_attrs=[shape, value, dtype], kernel_name='fill', attrs=attrs)
        expect, input, output = gen_data(dtype, shape, value)
        output = utils.mod_launch(mod, (output,), expect=expect)
        return input, output, expect, compare_tensor(output, expect, rtol=5e-03, equal_nan=True)


def gen_data(dtype, shape, value):
    input = None
    expect = np.full(shape, value, dtype)
    output = np.full(shape, np.nan, dtype)
    return expect, input, output
