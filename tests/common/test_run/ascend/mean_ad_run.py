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

from akg.utils import kernel_exec as utils
from tests.common.tensorio import compare_tensor
import numpy as np
from akg.ops.nn.ascend import MeanAd
from tests.common.base import get_rtol_atol


def mean_ad_run(shape, dtype, axis, keepdims, attrs):
    support_list = {"float16": np.float16, "float32": np.float32}
    if isinstance(axis, int):
        axis = (axis,)
    axis = list(axis)
    for i in range(len(axis)):
        if axis[i] < 0:
            axis[i] += len(shape)

    mean_num = 1
    dy_shape = []
    dy_shape_br = []
    for i, dim in enumerate(shape):
        if i in axis:
            if keepdims:
                dy_shape.append(1)

            mean_num *= dim
            dy_shape_br.append(1)
        else:
            dy_shape.append(dim)
            dy_shape_br.append(dim)
    if len(dy_shape) == 0:
        dy_shape.append(1)
    dy_shape = tuple(dy_shape)

    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(MeanAd, [dy_shape], [dtype], kernel_name=kernel_name,
                                  op_attrs=[shape, axis, keepdims], attrs=attrs, tuning=t)
        if t:
            expect, head_np, output = gen_data(dtype, dy_shape, dy_shape_br, mean_num, shape, support_list)
            return mod, expect, (head_np, output)
        else:
            return mod
    else:
        mod = utils.op_build_test(MeanAd, [dy_shape], [dtype], kernel_name='mean_ad', op_attrs=[shape, axis, keepdims],
                                  attrs=attrs)
        # if(axis == None):
        #     input_np = np.random.uniform(low=-1.0, high=1.0, size=shape).astype(support_list[dtype])
        #     head_np = np.random.uniform(low=-1.0, high=1.0, size=(1)).astype(support_list[dtype])
        #     expect = np.ones(shape)
        #     expect = expect * 1.0 / (shape[0] * shape[1] * shape[2] * shape[3])
        # else:
        expect, head_np, output = gen_data(dtype, dy_shape, dy_shape_br, mean_num, shape, support_list)
        output = utils.mod_launch(mod, (head_np, output), expect=expect)
        rtol, atol = get_rtol_atol("mean_ad", dtype)
        TestCase_Result = compare_tensor(output, expect, rtol=rtol, atol=atol, equal_nan=True)

        return (head_np,), output, expect, TestCase_Result


def gen_data(dtype, dy_shape, dy_shape_br, mean_num, shape, support_list):
    input_np = np.random.uniform(low=-1.0, high=1.0, size=shape).astype(support_list[dtype])
    head_np = np.random.uniform(low=-1.0, high=1.0, size=dy_shape).astype(support_list[dtype])
    expect = np.ones(shape)
    expect = expect * 1.0 / mean_num
    expect = expect * np.broadcast_to(np.reshape(head_np, dy_shape_br), shape)
    output = np.full(expect.shape, np.nan, dtype)
    return expect, head_np, output
