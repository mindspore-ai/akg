# Copyright 2020-2021 Huawei Technologies Co., Ltd
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

"""run function for erfc"""

import numpy as np

from tests.common.tensorio import compare_tensor
from akg.utils import kernel_exec as utils
from tests.common.gen_random import random_gaussian
from tests.common.base import get_rtol_atol
from tests.common.test_op.ascend import erfc

def erfc_run(shape, dtype, attrs):
    """run function"""
    mod = utils.op_build_test(erfc.erfc, [shape], [dtype],
                              kernel_name="erfc", attrs=attrs)
    expect, inputs, out_buf = gen_data(shape, dtype)
    output = utils.mod_launch(mod, (inputs, out_buf), expect=expect)
    rtol, atol = get_rtol_atol("erfc", dtype)
    cmp_res = compare_tensor(output, expect, rtol=rtol, atol=atol)
    return inputs, output, expect, cmp_res

def gen_data(shape, dtype):
    """generate valid data"""
    input_ = random_gaussian(shape, miu=1, sigma=0.5).astype(dtype)
    expect = (np.vectorize(np.math.erfc))(input_)
    out_buf = np.full(shape, np.nan, dtype)
    return expect, input_, out_buf
