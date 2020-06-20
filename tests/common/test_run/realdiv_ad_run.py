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
from tensorio import compare_tensor
import numpy as np
from test_op.realdiv_ad import realdiv_ad
from gen_random import random_gaussian

def realdiv_ad_run(ashape, bshape, dtype, kernel_name, attrs):
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        a = random_gaussian(ashape, miu=1, sigma=0.1).astype(dtype)
        b = random_gaussian(bshape, miu=1, sigma=0.1).astype(dtype)
        out = np.divide(a, b)
        mod = utils.op_build_test(realdiv_ad, [out.shape, ashape, bshape], [dtype, dtype, dtype], kernel_name=kernel_name, attrs=attrs, tuning=t)
        if t:
            expect, head_np, output = gen_data(b, dtype, out)
            return mod, expect, (head_np, a, b, output)
        else:
            return mod
    else:
        a = random_gaussian(ashape, miu=1, sigma=0.1).astype(dtype)
        b = random_gaussian(bshape, miu=1, sigma=0.1).astype(dtype)
        out = np.divide(a, b)

        expect, head_np, output = gen_data(b, dtype, out)
        mod = utils.op_build_test(realdiv_ad, [out.shape, ashape, bshape], [dtype, dtype, dtype], kernel_name=kernel_name, attrs=attrs)
        output = utils.mod_launch(mod, (head_np, a, b, output), expect=expect)
        return (head_np, a, b), output, expect, compare_tensor(output, expect, atol=0.1)


def gen_data(b, dtype, out):
    head_np = np.random.uniform(low=-1.0, high=1.0, size=out.shape).astype(out.dtype)
    expect = np.divide(head_np, b)
    output = np.full(expect.shape, np.nan, dtype)
    return expect, head_np, output
