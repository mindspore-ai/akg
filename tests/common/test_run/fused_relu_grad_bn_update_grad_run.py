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
from akg.utils import kernel_exec as utils
from tests.common.gen_random import random_gaussian
from akg.utils.result_analysis import target_profiling
from akg.utils.format_transform import to_tvm_nd_array
from tests.common.test_op.resnet.fused_pattern_grad_np import relu_grad_np, bn_beta_grad_np, bn_gamma_grad_np
from tests.common.test_op.resnet.fused_relu_grad_bn_update_grad import fused_relu_grad_bn_update_grad

def compute_expect(data_sum, in_bn, head_active, in_active, layout):
    out_dtype = data_sum.dtype
    relugrad = relu_grad_np(head_active, in_active).astype(out_dtype)
    inbn_cast = in_bn.astype(out_dtype)
    bn_beta_ad = bn_beta_grad_np(relugrad, layout)
    bn_gamma_ad = bn_gamma_grad_np(relugrad, inbn_cast, data_sum, layout)
    return [bn_gamma_ad, bn_beta_ad]


def gen_data(shape, out_shape, dtype, out_dtype, layout):
    support_list = {"float16": np.float16, "float32": np.float32}
    head = random_gaussian(shape, miu=1, sigma=0.1).astype(support_list[dtype])
    data_sum = random_gaussian(out_shape, miu=1, sigma=0.1).astype(support_list[out_dtype])
    in_bn = random_gaussian(shape, miu=1, sigma=0.1).astype(support_list[dtype])
    in_active = random_gaussian(shape, miu=1, sigma=0.1).astype(support_list[dtype])
    output = np.full(out_shape, 0.0, out_dtype)
    expect = compute_expect(data_sum, in_bn, head, in_active, layout)
    return head, data_sum, in_bn, in_active, output, expect

def fused_relu_grad_bn_update_grad_run(shape, out_shape, dtype="float16", layout="NHWC", out_dtype="float32", poly_sch=True, attrs=None):
    if not attrs:
        attrs = {"target": "cuda"}
    shape_list = [out_shape, shape, shape, shape]
    dtype_list = [out_dtype, dtype, dtype, dtype]
    op_attrs = [layout]
    mod = utils.op_build_test(
            fused_relu_grad_bn_update_grad,
            shape_list,
            dtype_list,
            op_attrs=op_attrs,
            kernel_name="fused_relu_grad_bn_update_grad",
            polyhedral=poly_sch,
            attrs=attrs)

    head, data_sum, in_bn, in_active, output, expect = gen_data(shape, out_shape, dtype, out_dtype, layout)
    outputs = [output, output]
    inputs = [data_sum, in_bn, head, in_active]
    arg_list = inputs + outputs
    outputs = utils.mod_launch(mod, arg_list, outputs=tuple(range(-len(outputs), 0)), expect=expect)
    res = np.allclose(outputs, expect, rtol=5e-03, atol=1.e-8)
    print("Test {}".format("Pass" if res else "Fail"))
    target_name = attrs["target"].split()[0]
    if not res:
        mod_source = mod
        if target_name != "llvm":
            mod_source = mod.imported_modules[0]
        print("Error {}:========================".format(target_name))
        print(mod_source.get_source())
        raise AssertionError("Test fail")

    if attrs["profiling"]:
        arg_list = to_tvm_nd_array(arg_list, akg.tvm.context(target_name, 0))
        target_profiling(mod, *arg_list, target=target_name, repeat_time=attrs["repeat_times"])
    return inputs, outputs, expect, res