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
from tensorio import compare_tensor
from akg.utils import kernel_exec as utils
from test_op import reduce_prod
from gen_random import random_gaussian

def reduce_prod_run(shape, dtype, axis=None, keepdims=False, attrs=None):
    ops_attrs = [axis, keepdims]

    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(reduce_prod.reduce_prod, [shape], [dtype], ops_attrs, kernel_name=kernel_name, attrs=attrs, tuning=t)
        if t:
            input, output, ref_res = gen_data(axis, dtype, keepdims, shape)
            return mod, ref_res, (input, output)
        else:
            return mod
    else:
        mod = utils.op_build_test(reduce_prod.reduce_prod, [shape], [dtype], ops_attrs, kernel_name="reduce_prod", attrs=attrs)
        input, output, ref_res = gen_data(axis, dtype, keepdims, shape)
        output = utils.mod_launch(mod, (input, output), expect=ref_res)
        test_case_result = compare_tensor(output, ref_res, rtol=5e-03, equal_nan=True)
        return input, output, ref_res, test_case_result


def gen_data(axis, dtype, keepdims, shape):
    support_list = {"float16": np.float16, "float32": np.float32, "int32": np.int32, "int8": np.int8, "uint8": np.uint8}
    input = random_gaussian(shape, miu=0.5, sigma=0.01).astype(support_list[dtype])
    if axis is None:
        axis_new = tuple([i for i in range(len(shape))])
    else:
        axis_new = axis
    ref_res = np.prod(input, axis=axis_new, keepdims=keepdims)
    output = np.full(ref_res.shape, 0, dtype)
    return input, output, ref_res
