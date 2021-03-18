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
from tests.operators.gpu.test_fused_pattern_grad import bn_beta_grad_np, bn_gamma_grad_np
from tests.common.test_op.resnet.fused_bn_update_grad import fused_bn_update_grad


def bn_update_grad(head, data_sum, in_bn, layout):
    out_dtype = data_sum.dtype
    head_cast = head.astype(out_dtype)
    inbn_cast = in_bn.astype(out_dtype)
    bn_beta_ad = bn_beta_grad_np(head_cast, layout)
    bn_gamma_ad = bn_gamma_grad_np(head_cast, inbn_cast, data_sum, layout)
    return [bn_beta_ad, bn_gamma_ad]


def gen_data(shape, out_shape, dtype, out_dtype, layout):
    support_list = {"float16": np.float16, "float32": np.float32}
    head = random_gaussian(shape, miu=1, sigma=0.1).astype(support_list[dtype])
    data_sum = random_gaussian(out_shape, miu=1, sigma=0.1).astype(support_list[out_dtype])
    in_bn = random_gaussian(shape, miu=1, sigma=0.1).astype(support_list[dtype])
    output = np.full(out_shape, 0.0, out_dtype)
    expect = bn_update_grad(head, data_sum, in_bn, layout)
    return head, data_sum, in_bn, output, expect

def test_fused_bn_update_grad(shape, out_shape, dtype="float16", out_dtype="float32", layout="NHWC", poly_sch=False):
    shape_list = [shape, out_shape, shape]
    dtype_list = [dtype, out_dtype, dtype]
    op_attrs = [layout]
    if poly_sch:
        mod = utils.op_build_test(fused_bn_update_grad, shape_list, dtype_list, op_attrs=op_attrs, kernel_name="fused_bn_update_grad", 
                          attrs={"target": "cuda", "enable_akg_reduce_lib": True, "enable_atomic_add": True})

    head, data_sum, in_bn, output, expect = gen_data(shape, out_shape, dtype, out_dtype, layout)
    outputs = [output, output]
    inputs = [head, data_sum, in_bn]
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
