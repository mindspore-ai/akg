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
from tests.common.test_op import matmul_ad
from tests.common.tensorio import compare_tensor
from tests.common.gen_random import random_gaussian

def matmul_ad_run(data_shape, weight_shape, dtype, attrs, cce_path="./"):

    check_list = ["float16"]
    if not (dtype.lower() in check_list):
        raise RuntimeError("matmul test only support %s while dtype is %s" % (",".join(check_list), dtype))

    mod = matmul_ad.matmul_ad(data_shape, weight_shape, dtype, attrs=attrs)
    input_data = random_gaussian(data_shape, miu=0.1, sigma=0.1)
    input_data = input_data.astype(np.float16)
    input_weight = random_gaussian(weight_shape, miu=0.1, sigma=0.1)
    input_weight = input_weight.astype(np.float16)
    expect = np.matmul(np.matmul(input_data, input_weight), np.transpose(input_weight))

    output = np.full(data_shape, 1.0, dtype)
    output = utils.mod_launch(mod, (np.matmul(input_data, input_weight), input_weight, output), expect=expect)
    return (np.matmul(input_data, input_weight), input_weight), output, expect, compare_tensor(output, expect, atol=5e-01, rtol=5e-03, equal_nan=True)
