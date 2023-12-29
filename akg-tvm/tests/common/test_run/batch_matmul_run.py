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
import akg
from akg.ops.math.gpu import BatchMatMul
from tests.common.gen_random import random_gaussian
from akg.utils import kernel_exec as utils
from akg.utils.result_analysis import target_profiling
from akg.utils.format_transform import to_tvm_nd_array


def gen_data(shape1, shape2, dtype, out_dtype="float32", layout1="NHDT", layout2="NHDT", layout_out="NHDT", shape_bias=None, add_bias=False):
    support_list = {"float16": np.float16, "float32": np.float32}
    lhs = random_gaussian(shape1, miu=1, sigma=0.1).astype(support_list[dtype])
    rhs = random_gaussian(shape2, miu=1, sigma=0.1).astype(support_list[dtype])
    bias = random_gaussian(shape_bias, miu=1, sigma=0.1).astype(
        support_list[out_dtype])

    data1 = lhs
    data2 = rhs

    if len(shape1) == 3:
        layout1 = layout1[1:]
        layout2 = layout2[1:]
    if len(shape1) == 2:
        layout1 = layout1[2:]
        layout2 = layout2[2:]

    if layout1 != "NHDT":
        layout1_int = layout1.replace('N', '0').replace(
            'H', '1').replace('D', '2').replace('T', '3')
        layout1_list = list(layout1_int)
        layout1_axis = np.argsort(layout1_list)
        data1 = np.transpose(data1, axes=layout1_axis)
    if layout2 != "NHTD":
        layout2_int = layout2.replace('N', '0').replace(
            'H', '1').replace('T', '2').replace('D', '3')
        layout2_list = list(layout2_int)
        layout2_axis = np.argsort(layout2_list)
        data2 = np.transpose(data2, axes=layout2_axis)

    if dtype != out_dtype:
        expect = np.matmul(data1.astype(out_dtype), data2.astype(out_dtype))
    else:
        expect = np.matmul(data1, data2)

    if add_bias == True:
        expect = np.add(expect, bias)

    if layout_out != "NHDT":
        if len(shape1) == 3:
            layout_out = layout_out[1:]
        if len(shape1) == 2:
            layout_out = layout_out[2:]
        layout_out_int = layout_out.replace('N', '0').replace(
            'H', '1').replace('D', '2').replace('T', '3')
        layout_out_list = list(layout_out_int)
        layout_out_axis = np.argsort(layout_out_list)
        expect = np.transpose(expect, axes=layout_out_axis)

    output = np.full(expect.shape, np.nan, out_dtype)
    print("expect shape is ", np.shape(expect))

    return lhs, rhs, bias, output, expect

def batch_matmul_run(shape1, shape2, dtype, out_dtype="float32", layout1="NHDT", layout2="NHDT", layout_out="NHDT",
                shape_bias=None, add_bias=False, tensor_core=True, poly_sch=True, attrs=None):
    op_attrs = [out_dtype, layout1, layout2, layout_out, tensor_core, add_bias]

    default_attrs = attrs
    if not attrs:
        default_attrs = {"target": "cuda"}

    if default_attrs["target"] == "cuda" and tensor_core:
        default_attrs.update({"pragma_enable_matmul": True, "enable_auto_inline": False})
    elif default_attrs["target"] == "llvm":
        if "pragma_enable_matmul" not in default_attrs.keys():
            default_attrs["pragma_enable_matmul"] = True
        if "feature" not in default_attrs.keys():
            default_attrs["feature"] = "avx"

    mod = utils.op_build_test(BatchMatMul, (shape1, shape2, shape_bias), (dtype, dtype, out_dtype),
                            op_attrs=op_attrs, attrs=default_attrs, polyhedral=poly_sch, kernel_name="batch_matmul")

    lhs, rhs, bias, output, expect = gen_data(
        shape1, shape2, dtype, out_dtype, layout1, layout2, layout_out, shape_bias, add_bias)
    args = (lhs, rhs, bias, output)
    output = utils.mod_launch(mod, args, expect=expect)
    res = np.allclose(output, expect, rtol=5e-03, atol=1.e-8)
    print("Test {}".format("Pass" if res else "Fail"))
    target_name = default_attrs["target"].split()[0]
    if not res:
        mod_source = mod
        if target_name != "llvm":
            mod_source = mod.imported_modules[0]
        print("Error {}:========================".format(target_name))
        print(mod_source.get_source())
        raise AssertionError("Test fail")

    if attrs["profiling"]:
        args = to_tvm_nd_array(args, akg.tvm.context(target_name, 0))
        target_profiling(mod, *args, target = target_name, repeat_time = attrs["repeat_times"])
    return (lhs, rhs, bias), output, expect, res
