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

from __future__ import absolute_import
import numpy as np
from akg.utils import kernel_exec as utils
from tests.common.gen_random import random_gaussian
from akg.utils.result_analysis import gpu_profiling
from akg.utils.format_transform import to_tvm_nd_array
from tests.operators.gpu.test_fused_pattern_grad import relu_grad_np
from tests.common.test_op.resnet.fused_relu_grad_bn_double_reduce_grad import fused_relu_grad_bn_double_reduce_grad

def compute_expect(inshp_data, outshp_data):
    out_shape = outshp_data.shape
    scale = out_shape[0] * out_shape[1] * out_shape[2]
    mul = np.multiply(inshp_data, inshp_data)
    mean1 = np.divide(mul, scale)

    add = np.add(outshp_data, outshp_data)
    addgrad = relu_grad_np(add, outshp_data).astype(inshp_data.dtype)
    mul1 = np.multiply(addgrad, scale)
    sub = np.subtract(mul1, inshp_data)

    outdata_cast = outshp_data.astype(inshp_data.dtype)
    mean2 = np.divide(inshp_data, scale)
    sub1 = np.subtract(outdata_cast, mean2)
    mul2 = np.multiply(sub1, inshp_data)
    div = np.divide(mul2, inshp_data)
    sub2 = np.subtract(sub, div)
    mul3 = np.multiply(mean1, sub2).astype(outshp_data.dtype)

    mul4 = np.multiply(inshp_data, inshp_data)
    mean3 = np.divide(mul4, scale)
    mean4 = np.divide(inshp_data, scale)
    sub3 = np.subtract(outshp_data.astype(inshp_data.dtype), mean4)
    mul5 = np.multiply(inshp_data, sub3)

    div1 = np.divide(mul5, inshp_data)
    sub4 = np.subtract(sub, div1)
    mul6 = np.multiply(mean3, sub4).astype(outshp_data.dtype)
    return [mul3, mul6]


def gen_data(shape, out_shape, dtype, out_dtype):
    support_list = {"float16": np.float16, "float32": np.float32}
    inshp_data = random_gaussian(shape, miu=1, sigma=0.1).astype(support_list[dtype])
    outshp_data = random_gaussian(out_shape, miu=1, sigma=0.1).astype(support_list[out_dtype])
    output = np.full(out_shape, np.nan, out_dtype)
    expect = compute_expect(inshp_data, outshp_data)
    return inshp_data, outshp_data, output, expect

def test_fused_relu_grad_bn_double_reduce_grad(shape, out_shape, dtype="float32", layout="NHWC", out_dtype="float16", poly_sch=False):
    
    shape_list = [shape] * 5 + [out_shape] + [shape] * 3 + [out_shape] + [shape] * 3 + [out_shape] * 3
    dtype_list = [dtype] * 5 +[out_dtype] +[dtype] * 3 + [out_dtype] + [dtype] * 3 +[out_dtype] * 3
    op_attrs = [layout, out_dtype]
    if poly_sch:
        mod = utils.op_build_test(
            fused_relu_grad_bn_double_reduce_grad,
            shape_list,
            dtype_list,
            op_attrs=op_attrs,
            kernel_name="fused_relu_grad_bn_double_reduce_grad",
            attrs={
                "target": "cuda"})

    inshp_data, outshp_data, output, expect = gen_data(shape, out_shape, dtype, out_dtype)
    inputs = [inshp_data] * 5 + [outshp_data] + [inshp_data] * 3 + [outshp_data] + [inshp_data] * 3 + [outshp_data] * 3
    outputs = [output, output]
    arg_list = inputs + outputs
    outputs = utils.mod_launch(mod, arg_list, outputs=tuple(range(-len(outputs), 0)), expect=expect)

    res = np.allclose(outputs, expect, rtol=5e-03, atol=1.e-8)
    print("Test {}".format("Pass" if res else "Fail"))
    if not res:
        print("Error cuda:========================")
        print(mod.imported_modules[0].get_source())
        raise AssertionError("Test fail")

    inputs = to_tvm_nd_array(inputs)
    expect = to_tvm_nd_array(expect)
    gpu_profiling(mod, *inputs, *expect, 400)
