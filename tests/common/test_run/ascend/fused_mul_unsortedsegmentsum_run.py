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
from akg.ops.array import UnsortedSegmentSum
from akg.ops.math import mul
from tests.common.tensorio import compare_tensor
from tests.common.gen_random import random_gaussian

def gen_segment_ids(ishape, irange):
    segment_ids = np.random.randint(0, irange, size=(ishape)).astype("int32")
    # segment_ids[ilength-1] = irange - 1
    return segment_ids


def cal_outputs(input_data, data_type, segment_ids, output_shape):
    input_shape = input_data.shape

    # assert(input_shape[0] == len(segment_ids))
    # assert(num_segments > max(segment_ids))
    # assert(num_segments == output_shape[0])

    output_data = np.zeros(output_shape, data_type)

    if len(segment_ids.shape) == 1:
        for i in range(input_shape[0]):
            output_data[segment_ids[i]] += input_data[i]
    elif len(segment_ids.shape) == 2:
        for i in range(input_shape[0]):
            for j in range(input_shape[1]):
                output_data[segment_ids[i, j]] += input_data[i, j]
    elif len(segment_ids.shape) == 3:
        for i in range(input_shape[0]):
            for j in range(input_shape[1]):
                for k in range(input_shape[2]):
                    output_data[segment_ids[i, j, k]] += input_data[i, j, k]

    output_data = output_data.astype(data_type)
    return output_data


def mul_unsortedsegmentsum(input1, input2, ids_tensor, num_segments, target="cce"):
    import akg.tvm
    temp = mul(input1, input2, target='cce')
    output = UnsortedSegmentSum(temp, ids_tensor, num_segments, target=target)[0]
    output = akg.tvm.compute(output.shape, lambda *i: output(*i), "fused_mul_unsorted")
    return output


def fused_mul_unsortedsegmentsum_execute(shape1, shape2, ids_shape, num_segments, dtype, attrs):
    attrs["pragma_disable_whole_component"] = False
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = fused_mul_unsortedsegmentsum_compile(shape1, shape2, ids_shape, num_segments, dtype, attrs,
                                                   kernel_name=kernel_name, tuning=t)
        if t:
            expect, input1, input2, output, segment_ids = gen_data(dtype, ids_shape, num_segments, shape1, shape2)
            return mod, expect, (input1, input2, segment_ids, output)
        else:
            return mod
    else:
        expect, input1, input2, output, segment_ids = gen_data(dtype, ids_shape, num_segments, shape1, shape2)
        mod = fused_mul_unsortedsegmentsum_compile(shape1, shape2, ids_shape, num_segments, dtype, attrs)
        output = utils.mod_launch(mod, (input1, input2, segment_ids, output), expect=expect)

        return (input1, input2, segment_ids, num_segments), output, expect, compare_tensor(output, expect, rtol=5e-03,
                                                                                           equal_nan=True)


def gen_data(dtype, ids_shape, num_segments, shape1, shape2):
    support_list = {"float16": np.float16, "float32": np.float32}
    if not (dtype.lower() in support_list):
        raise RuntimeError("Auto-tensor only support %s while dtype is %s" % (",".join(support_list.keys()), dtype))
    segment_ids = gen_segment_ids(ids_shape, num_segments)
    # Generate data for testing the op
    input1 = random_gaussian(shape1, miu=1, sigma=0.1).astype(support_list[dtype])
    input2 = random_gaussian(shape2, miu=1, sigma=0.1).astype(support_list[dtype])
    temp = np.multiply(input1, input2)
    output_shape = (num_segments,) + tuple(shape1[len(ids_shape):])
    expect = cal_outputs(temp, support_list[dtype], segment_ids, output_shape)
    output = np.full(output_shape, np.nan, dtype)
    return expect, input1, input2, output, segment_ids


def fused_mul_unsortedsegmentsum_compile(shape1, shape2, ids_shape, num_segments, dtype, attrs,
                                         kernel_name='unsortedsegmentsum_run', tuning=False):
    segment_ids = gen_segment_ids(ids_shape, num_segments)
    return utils.op_build_test(mul_unsortedsegmentsum, [shape1, shape2, segment_ids.shape], [dtype, dtype, 'int32'],
                               op_attrs=[num_segments], kernel_name=kernel_name, attrs=attrs, tuning=tuning)
