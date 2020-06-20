# Copyright 2020 Huawei Technologies Co., Ltd
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

"""ones_like_run"""

import numpy as np
from akg.topi.util import get_const_tuple
from akg.utils import kernel_exec as utils
from test_op import ones_like
from tensorio import compare_tensor
from base import get_rtol_atol


def ones_like_run(shape, dtype, attrs):
    mod = utils.op_build_test(ones_like.ones_like, [shape], [dtype], kernel_name='ones_like', attrs=attrs)
    input, expect, output = gen_data(dtype, shape)
    output = utils.mod_launch(mod, (input, output), expect=expect)
    rtol, atol = get_rtol_atol("ones_like", dtype)
    # compare result
    compare_res = compare_tensor(output, expect, rtol=rtol, atol=atol)
    return input, output, expect, compare_res


def gen_data(dtype, shape):
    # generate data
    input = np.random.uniform(low=-1.0, high=1.0, size=get_const_tuple(shape)).astype(dtype)
    expect = np.ones_like(input)
    output = np.full(shape, np.nan, dtype)
    return input, expect, output

