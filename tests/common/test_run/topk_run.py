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
from tests.common.test_op import topk
from tests.common.tensorio import compare_tensor


def less_than(a, b):
    return a[0] < b[0] or (a[0] == b[0] and a[1] > b[1])


def cal_topk(A, k):
    if k > A.shape[-1]:
        raise RuntimeError("k should not be greater than shape[-1]")
    out_shape = A.shape[:-1] + (k,)

    last_dim = A.shape[-1]
    loop_cnt = 1
    for x in A.shape[:-1]:
        loop_cnt *= x

    in_len = loop_cnt * last_dim
    out_len = loop_cnt * k

    out_value = np.zeros((out_len,), dtype=A.dtype)
    out_index = np.zeros((out_len,), dtype="int32")

    A = A.flatten()

    arr_idx = np.zeros((last_dim,), dtype="int32")
    arr_val = np.zeros((last_dim,), dtype=A.dtype)
    for i in range(loop_cnt):
        base_in = i * last_dim
        base_out = i * k
        for j in range(last_dim):
            arr_idx[j] = j
            arr_val[j] = A[base_in + j]
        # get topk values by selection sort
        for x in range(k):
            p = x
            for y in range(x + 1, last_dim):
                if less_than((arr_val[p], arr_idx[p]), (arr_val[y], arr_idx[y])):
                    p = y
            out_value[base_out + x] = arr_val[p]
            out_index[base_out + x] = arr_idx[p]
            if x != p:
                arr_val[p] = arr_val[x]
                arr_idx[p] = arr_idx[x]

    return out_value.reshape(out_shape), out_index.reshape(out_shape)


def topk_run(shape, k, dtype, attrs):
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(topk.topk, [shape], [dtype], [k],
                                  kernel_name=kernel_name, attrs=attrs, tuning=t)
        if t:
            expect_index, expect_value, input, output_value = gen_data(dtype, k, shape)
            return mod, (expect_index, expect_value), (input, output_value)
        else:
            return mod
    else:
        mod = topk.topk(shape, k, dtype, "topk", attrs)
        expect_index, expect_value, input, output_value = gen_data(dtype, k, shape)
        output_index = np.full(expect_index.shape, np.nan, dtype)

        #output_value, output_index = utils.mod_launch(mod, (input, output_value, output_index))
        output_value = utils.mod_launch(mod, (input, output_value), expect=(expect_value, expect_index))
        result = compare_tensor(output_value, expect_value, rtol=5e-03, equal_nan=True) and \
            compare_tensor(output_index, expect_index, rtol=5e-03, equal_nan=True)
        return input, (output_value, output_index), (expect_value, expect_index), result


def gen_data(dtype, k, shape):
    input = np.random.randint(100, size=shape).astype(dtype)
    out_shape = shape[:-1] + (k,)
    expect_value, expect_index = cal_topk(input, k)
    output_value = np.full(expect_value.shape, np.nan, dtype)
    return expect_index, expect_value, input, output_value
