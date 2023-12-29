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
from akg.utils import kernel_exec as utils
from tests.common.test_op.ascend import blas_axby_ad
from tests.common.gen_random import random_gaussian
from tests.common.base import get_rtol_atol

def blas_axby_ad_run(shape, dtype, kernel_name, attrs):
    alpha = 2.0
    beta = 3.0

    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(blas_axby_ad.blas_axby_ad, [shape], [dtype], [alpha, beta], kernel_name=kernel_name,
                                  attrs=attrs, tuning=t)
        if t:
            args, expect_dx, expect_dy, head = gen_data(alpha, beta, dtype, shape)
            return mod, (expect_dx, expect_dy), {"args": args, 'outputs': (1, 2), 'tuning': False}
        else:
            return mod
    else:
        mod = utils.op_build_test(blas_axby_ad.blas_axby_ad, [shape], [dtype], [alpha, beta], kernel_name=kernel_name,
                                  attrs=attrs)
        args, expect_dx, expect_dy, head = gen_data(alpha, beta, dtype, shape)
        output_dx, output_dy = utils.mod_launch(mod, args, outputs=(1, 2), expect=(expect_dx, expect_dy))

        rtol, atol = get_rtol_atol("blas_axby_ad", dtype)
        result = compare_tensor(expect_dx, output_dx, rtol=rtol, atol=atol, equal_nan=True) and \
            compare_tensor(expect_dy, output_dy, rtol=rtol, atol=atol, equal_nan=True)

        return (head), (expect_dx, expect_dy), (output_dx, output_dy), result


def gen_data(alpha, beta, dtype, shape):
    support_list = {"float16": np.float16, "float32": np.float32}
    head = random_gaussian(shape, miu=1, sigma=0.1).astype(support_list[dtype])
    expect_dx = np.multiply(head, alpha)
    expect_dy = np.multiply(head, beta)
    output_dx = np.full(expect_dx.shape, np.nan, dtype)
    output_dy = np.full(expect_dy.shape, np.nan, dtype)
    args = [head, output_dx, output_dy]
    return args, expect_dx, expect_dy, head
