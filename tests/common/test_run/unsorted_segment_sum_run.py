# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
# limitations under the License

import numpy as np
import akg
from tests.common.base import get_rtol_atol
from tests.common.tensorio import compare_tensor
from akg.utils import kernel_exec as utils
from akg.utils import CCE
from akg.ops.array import UnsortedSegmentSum
from akg.utils.result_analysis import target_profiling
from akg.utils.format_transform import to_tvm_nd_array
from akg.utils.gen_random import random_gaussian, gen_indices_unsorted_segment_sum

def unsorted_segment_sum_run(shape, ids_shape, num_segments, dtype, attrs):
    if not attrs:
        attrs = {"target": CCE}
    if attrs["target"] != CCE:
        return unsorted_segment_sum_run_others(shape, dtype, ids_shape, "int32", num_segments, attrs)

    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = unsortedsegmentsum_compile(shape, ids_shape, num_segments, dtype, attrs, kernel_name=kernel_name, tuning=t)
        if t:
            expect, input, output, segment_ids = gen_data_ascend(dtype, ids_shape, num_segments, shape)
            return mod, expect, (input, segment_ids, output)
        else:
            return mod
    else:
        attrs["multicore_loop_switch_hoist"] = False
        mod = unsortedsegmentsum_compile(shape, ids_shape, num_segments, dtype, attrs)
        expect, input, output, segment_ids = gen_data_ascend(dtype, ids_shape, num_segments, shape)
        output = utils.mod_launch(mod, (input, segment_ids, output), expect=expect)

        rtol, atol = get_rtol_atol("unsortedsegmentsum", dtype)
        return (input, segment_ids, num_segments), output, expect, compare_tensor(output, expect, rtol=rtol, atol=atol,
                                                                                  equal_nan=True)

def unsorted_segment_sum_run_others(data_shape, data_type, indices_shape, indices_type, num, attrs=None):
    mod = unsortedsegmentsum_compile(data_shape, indices_shape, num, data_type, attrs,
                                    kernel_name='unsortedsegmentsum_run', tuning=False)
    # gen data
    input1, input2, expect = gen_data(data_shape, data_type, indices_shape, indices_type, num)
    output_shape = expect.shape

    if len(expect.shape) == 0:
        output_shape = (1, )
    #output = np.full(output_shape, np.nan, expect.dtype)
    output = np.zeros(output_shape, expect.dtype)
    output = utils.mod_launch(mod, (input1, input2, output), expect = expect)

    atol, rtol = get_rtol_atol("unsorted_segment_sum", data_type)
    res = compare_tensor(output, expect, rtol=rtol, atol=atol)
    print("Test {}".format("Pass" if res else "Failed"))
    target_name = attrs["target"].split()[0]
    if not res:
        mod_source = mod
        if target_name != "llvm":
            mod_source = mod.imported_modules[0]
        print("Error {}:========================".format(target_name))
        print(mod_source.get_source())
        raise AssertionError("Test fail")

    if attrs["profiling"]:
        input1, input2, output = to_tvm_nd_array(
            [input1, input2, output], akg.tvm.context(target_name, 0))
        target_profiling(mod, input1, input2, output, target=target_name, repeat_time=attrs["repeat_times"])
    return (input1, input2), output, expect, res

def gen_segment_ids(ishape, irange):
    segment_ids = np.random.randint(0, irange, size=(ishape)).astype("int32")
    # segment_ids[ilength-1] = irange - 1
    return segment_ids


def cal_outputs(input_data, data_type, segment_ids, num_segments, output_shape):
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

def gen_data_ascend(dtype, ids_shape, num_segments, shape):
    support_list = {"float16": np.float16, "float32": np.float32}
    if not (dtype.lower() in support_list):
        raise RuntimeError("Auto-tensor only support %s while dtype is %s" % (",".join(support_list.keys()), dtype))
    segment_ids = gen_segment_ids(ids_shape, num_segments)
    # Generate data for testing the op
    input = random_gaussian(shape, miu=1, sigma=0.1).astype(support_list[dtype])
    output_shape = (num_segments,) + tuple(shape[len(ids_shape):])
    expect = cal_outputs(input, support_list[dtype], segment_ids, num_segments, output_shape)
    output = np.full(output_shape, np.nan, dtype)
    return expect, input, output, segment_ids

def gen_data(shape1, dtype1, shape2, dtype2, num):
    input1 = random_gaussian(shape1).astype(dtype1)
    input2 = gen_indices_unsorted_segment_sum(shape1, shape2, dtype2, num)
    expect = np.zeros((num,) + shape1[len(shape2):]).astype(dtype1)
    np.add.at(expect, input2, input1)
    return input1, input2, expect

def unsortedsegmentsum_compile(shape, ids_shape, num_segments, dtype, attrs, kernel_name='unsortedsegmentsum_run',
                               tuning=False):
    segment_shape = gen_segment_ids(ids_shape, num_segments).shape if attrs["target"] == utils.CCE else ids_shape
    return utils.op_build_test(UnsortedSegmentSum, [shape, segment_shape], [dtype, 'int32'],
                               op_attrs=[num_segments], kernel_name=kernel_name, attrs=attrs, tuning=tuning)
