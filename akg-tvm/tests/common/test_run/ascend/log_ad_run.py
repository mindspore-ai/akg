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
from akg.utils import kernel_exec as utils
from akg.ops.math.ascend import log_ad
from tests.common.tensorio import compare_tensor
from tests.common.base import get_rtol_atol


def log_ad_run(shape, dtype, kernel_name, attrs):
    input_shape = [shape, shape]
    input_dtype = [dtype, dtype]

    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(log_ad, input_shape, input_dtype, kernel_name=kernel_name, attrs=attrs, tuning=t)
        if t:
            expect, head_np, input_, output = gen_data(dtype, shape)
            return mod, expect, (head_np, input_, output)
        else:
            return mod
    else:
        mod = utils.op_build_test(log_ad, input_shape, input_dtype, kernel_name=kernel_name, attrs=attrs)
        expect, head_np, input_, output = gen_data(dtype, shape)
        output = utils.mod_launch(mod, (head_np, input_, output), expect=expect)
        rtol, atol = get_rtol_atol("log_ad", dtype)
        return input_, output, expect, compare_tensor(output, expect, rtol=rtol, atol=atol, equal_nan=True)


def gen_data(dtype, shape):
    input_ = np.random.uniform(low=1e-4, high=10.0, size=shape).astype(dtype)
    head_np = np.random.uniform(low=0, high=1.0, size=shape).astype(dtype)
    expect = 1 / input_ * head_np
    output = np.full(shape, np.nan, dtype)
    return expect, head_np, input_, output

