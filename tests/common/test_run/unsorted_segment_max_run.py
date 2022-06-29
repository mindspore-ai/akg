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
from akg.ops.array import unsorted_segment_max
from tests.common.gen_random import random_gaussian
from tests.common.tensorio import compare_tensor


def gen_segment_ids(ishape, irange):
    segment_ids = np.random.randint(0, irange, size=(ishape)).astype("int32")
    return segment_ids


def cal_outputs(input_data, data_type, segment_ids, output_shape):
    input_shape = input_data.shape
    output_data = np.zeros(output_shape, data_type)

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


def unsorted_segment_max_run(shape, ids_shape, num_segments, dtype, attrs):
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        segment_ids = gen_segment_ids(ids_shape, num_segments)
        mod = utils.op_build_test(unsorted_segment_max, [shape], [dtype], op_attrs=[segment_ids, num_segments],
                                  kernel_name=kernel_name, attrs=attrs, tuning=t)
        if t:
            expect, data_input, output = gen_data(dtype, ids_shape, num_segments, segment_ids, shape)
            return mod, expect, (data_input, output)
        else:
            return mod
    else:
        segment_ids = gen_segment_ids(ids_shape, num_segments)
        expect, data_input, output = gen_data(dtype, ids_shape, num_segments, segment_ids, shape)
        mod = utils.op_build_test(unsorted_segment_max, [shape], [dtype], op_attrs=[segment_ids, num_segments],
                                  kernel_name='unsorted_segment_max', attrs=attrs)
        import datetime
        begin = datetime.datetime.now()
        output = utils.mod_launch(mod, (data_input, output), expect=expect)
        end = datetime.datetime.now()
        print((begin - end).seconds)
        return (data_input, segment_ids), output, expect, compare_tensor(output, expect, rtol=5e-03, equal_nan=True)


def gen_data(dtype, ids_shape, num_segments, segment_ids, shape):
    support_list = {"float16": np.float16, "float32": np.float32}
    if not (dtype.lower() in support_list):
        raise RuntimeError("Auto-tensor only support %s while dtype is %s" % (",".join(support_list.keys()), dtype))

    # Generate data for testing the op
    data_input = random_gaussian(shape, miu=20, sigma=10).astype(support_list[dtype])
    data_input = np.abs(data_input)
    output_shape = (num_segments,) + tuple(shape[len(ids_shape):])
    expect = cal_outputs(data_input, np.float16, segment_ids, output_shape)
    output = np.full(output_shape, np.nan, dtype)
    return expect, data_input, output
