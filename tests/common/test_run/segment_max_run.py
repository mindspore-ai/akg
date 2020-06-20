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

"""
unsortedsegmentsum run define
"""

import numpy as np
from akg.utils import kernel_exec as utils
from test_op.segment_max import segment_max
from gen_random import random_gaussian
from tensorio import compare_tensor


def gen_segment_ids(ishape, num_segments):

    segment_ids = np.random.randint(0, num_segments, size=(ishape)).astype("int32")
    segment_ids = np.sort(segment_ids)
    return segment_ids


def cal_outputs(input_data, data_type, segment_ids, output_shape):
    input_shape = input_data.shape
    output_data = np.ones(output_shape, data_type) * (-65504)

    if len(input_data.shape) == 1:
        for i in range(input_shape[0]):
            output_data[segment_ids[i]] = max(output_data[segment_ids[i]], input_data[i])
    elif len(input_data.shape) == 2:
        for i in range(input_shape[0]):
            for j in range(input_shape[1]):
                output_data[segment_ids[i], j] = max(output_data[segment_ids[i], j], input_data[i, j])
    elif len(input_data.shape) == 3:
        for i in range(input_shape[0]):
            for j in range(input_shape[1]):
                for k in range(input_shape[2]):
                    output_data[segment_ids[i], j, k] = max(output_data[segment_ids[i], j, k], input_data[i, j, k])

    output_data = output_data.astype(data_type)
    return output_data


def segment_max_run(shape, ids_shape, num_segments, dtype, attrs=None):
    segment_ids = gen_segment_ids(ids_shape, num_segments)

    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(segment_max, [shape], [dtype], op_attrs=[segment_ids, num_segments], kernel_name=kernel_name, attrs=attrs, tuning=t)
        if t:
            expect, input, output = gen_data(dtype, ids_shape, num_segments, segment_ids, shape)
            return mod, expect, (input, output)
        else:
            return mod
    else:
        mod = utils.op_build_test(segment_max, [shape], [dtype], op_attrs=[segment_ids, num_segments], kernel_name='segment_max', attrs=attrs)
        expect, input, output = gen_data(dtype, ids_shape, num_segments, segment_ids, shape)
        output = utils.mod_launch(mod, (input, output), expect=expect)

        return (input, segment_ids), output, expect, compare_tensor(output, expect, rtol=5e-03, equal_nan=True)


def gen_data(dtype, ids_shape, num_segments, segment_ids, shape):
    support_list = {"float16": np.float16, "float32": np.float32}
    if not (dtype.lower() in support_list):
        raise RuntimeError("Auto-tensor only support %s while dtype is %s" % (",".join(support_list.keys()), dtype))

    # Generate data for testing the op
    input = random_gaussian(shape, miu=1, sigma=10).astype(support_list[dtype])
    output_shape = (num_segments,) + tuple(shape[len(ids_shape):])
    expect = cal_outputs(input, np.float16, segment_ids, output_shape)
    output = np.full(output_shape, 0, dtype)
    return expect, input, output
