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
from tests.common.test_op import erf
from tests.common.tensorio import compare_tensor
from tests.common.gen_random import random_gaussian

def erf_run(shape, dtype, kernel_name, attrs, cce_path="./"):
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(erf.erf, [shape], [dtype], kernel_name=kernel_name, attrs=attrs, tuning=t)
        if t:
            expect, input, output = gen_data(dtype, shape)
            return mod, expect, (input, output)
        else:
            return mod
    else:
        mod = utils.op_build_test(erf.erf, [shape], [dtype], kernel_name="erf", attrs=attrs)
        expect, input, output = gen_data(dtype, shape)
        output = utils.mod_launch(mod, [input, output], expect=expect)

        return input, output, expect, compare_tensor(output, expect, rtol=1e-02, equal_nan=True)


def gen_data(dtype, shape):
    support_list = {"float16": np.float16, "float32": np.float32}
    input = random_gaussian(shape, miu=1, sigma=0.1).astype(support_list[dtype])
    nperf = np.vectorize(np.math.erf)
    expect = nperf(input)
    output = np.full(shape, 0, dtype)
    return expect, input, output
