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
import secrets
from tests.common.tensorio import compare_tensor
from akg.utils import kernel_exec as utils
from akg.ops.math import round_
from tests.common.gen_random import random_gaussian
from akg.utils.result_analysis import target_profiling
from akg.utils.format_transform import to_tvm_nd_array

secretsGenerator = secrets.SystemRandom()
def round_run(shape, dtype, attrs):
    in_shape = [shape]
    in_dtype = [dtype]

    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(round_, in_shape, in_dtype, kernel_name=kernel_name, attrs=attrs, tuning=t)
        if t:
            expect, input, output = gen_data(shape, dtype, attrs["target"])
            return mod, expect, (input, output)
        else:
            return mod
    else:
        mod = utils.op_build_test(round_, in_shape, in_dtype, kernel_name='round', attrs=attrs)
        expect, input, output = gen_data(shape, dtype, attrs["target"])
        output = utils.mod_launch(mod, (input, output), expect=expect)
        if attrs.get("profiling", False):
            import akg
            target_name = attrs["target"].split()[0]
            args_list = to_tvm_nd_array([input, output], akg.tvm.context(target_name, 0))
            target_profiling(mod, *args_list, target=target_name, repeat_time=attrs["repeat_times"])
        return input, output, expect, compare_tensor(output, expect, rtol=5e-03, atol=1.e-8, equal_nan=True)


def gen_data(shape, dtype, target):
    if target == utils.CCE:
        data = random_gaussian(shape, miu=1, sigma=10).astype(dtype)
        a = secretsGenerator.randint(0, 9)
        if a % 2 == 0:
            data = data.astype('int32') + 0.5
            data = data.astype(dtype)
        input_f16 = data.astype(np.float16)
        expect = np.round(input_f16).astype("int32")
        output = np.full(shape, np.nan, "int32")
    else:
        support_list = {"float16": np.float16, "float32": np.float32}
        data = random_gaussian(shape, miu=1, sigma=0.1).astype(support_list[dtype])
        expect = np.round(data)
        output = np.full(expect.shape, np.nan, dtype)
    return expect, data, output