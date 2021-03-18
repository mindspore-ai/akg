# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
from akg.utils import kernel_exec as utils
from tests.common.test_op.batch_matmul import batch_matmul
from tests.common.gen_random import random_gaussian


def gen_data(shape1, shape2, dtype, layout1="NHDT", layout2="NHDT", layout_out="NHDT", shape_bias=None, add_bias=False):
    support_list = {"float16": np.float16, "float32": np.float32}
    lhs = random_gaussian(shape1, miu=1, sigma=0.1).astype(support_list[dtype])
    rhs = random_gaussian(shape2, miu=1, sigma=0.1).astype(support_list[dtype])
    bias = []
    if add_bias == True:
        bias = random_gaussian(shape_bias, miu=1, sigma=0.1).astype(support_list[dtype])
    data1 = lhs
    data2 = rhs

    if len(shape1) == 3:
        layout1 = layout1[1:]
        layout2 = layout2[1:]
    if len(shape1) == 2:
        layout1 = layout1[2:]
        layout2 = layout2[2:]

    if layout1 != "NHDT":
        layout1_int = layout1.replace('N', '0').replace('H', '1').replace('D', '2').replace('T', '3')
        layout1_list = list(layout1_int)
        layout1_axis = np.argsort(layout1_list)
        data1 = np.transpose(data1, axes=layout1_axis)
    if layout2 != "NHTD":
        layout2_int = layout2.replace('N', '0').replace('H', '1').replace('T', '2').replace('D', '3')
        layout2_list = list(layout2_int)
        layout2_axis = np.argsort(layout2_list)
        data2 = np.transpose(data2, axes=layout2_axis)

    expect = np.matmul(data1, data2)
    if add_bias == True:
        expect = np.add(expect, bias)

    if layout_out != "NHDT":
        if len(shape1) == 3:
            layout_out = layout_out[1:]
        if len(shape1) == 2:
            layout_out = layout_out[2:]
        layout_out_int = layout_out.replace('N', '0').replace('H', '1').replace('D', '2').replace('T', '3')
        layout_out_list = list(layout_out_int)
        layout_out_axis = np.argsort(layout_out_list)
        expect = np.transpose(expect, axes=layout_out_axis)

    output = np.full(expect.shape, np.nan, dtype)
    print("expect shape is ", np.shape(expect))

    return lhs, rhs, bias, output, expect


def test_ms_bmm(shape1, shape2, dtype, layout1="NHDT", layout2="NHDT", layout_out="NHDT", shape_bias=None,
                add_bias=False, poly_sch=False):
    shape_tuple = (shape1, shape2)
    dtype_tuple = (dtype, dtype)
    op_attrs = [None, layout1, layout2, layout_out]
    if add_bias == True:
        shape_tuple = (shape1, shape2, shape_bias)
        dtype_tuple = (dtype, dtype, dtype)
        op_attrs = [layout1, layout2, layout_out]

    if poly_sch:
        manual_tiling = False
        attrs = {"target": "cuda"}
        if manual_tiling:
            attrs["bind_block"] = "32 16"  # block x = 32, block y = 16
            attrs["bind_thread"] = "32 8"  # thread x = 32, thread y = 8
            attrs["dim"] = "0 0 64 1 0 1 32 1 0 2 32 1"  # tile m = 64, tile n = 32, tile k = 32

        mod = utils.op_build_test(batch_matmul, shape_tuple, dtype_tuple, op_attrs=op_attrs, attrs=attrs,
                                  kernel_name="batch_matmul")

    lhs, rhs, bias, output, expect = gen_data(shape1, shape2, dtype, layout1, layout2, layout_out, shape_bias, add_bias)
    args = (lhs, rhs, output)
    if add_bias == True:
        args = (lhs, rhs, bias, output)
    output = utils.mod_launch(mod, args, expect=expect)
    res = np.allclose(output, expect, rtol=5e-03, atol=1.e-8)
    print("Test {}".format("Pass" if res else "Fail"))
    if not res:
        print("Error cuda:========================")
        print(mod.imported_modules[0].get_source())
        raise AssertionError("Test fail")

    return True
