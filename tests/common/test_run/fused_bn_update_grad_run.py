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
from akg import topi
from akg.utils import kernel_exec as utils
from tests.common.gen_random import random_gaussian
from akg.utils.result_analysis import target_profiling
from akg.utils.format_transform import to_tvm_nd_array
from tests.common.test_op.resnet.fused_bn_update_grad import fused_bn_update_grad
from tests.common.test_op.resnet.fused_pattern_grad_np import bn_beta_grad_np, bn_gamma_grad_np

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

def fused_bn_update_grad_run(shape, out_shape, dtype="float16", out_dtype="float32", layout="NHWC", poly_sch=True, attrs=None):
    if not attrs:
        attrs = {"target": "cuda"}
    attrs.update({"enable_akg_reduce_lib": True, "enable_atomic_add": True})
    shape_list = [shape, out_shape, shape]
    dtype_list = [dtype, out_dtype, dtype]
    op_attrs = [layout]
    mod = utils.op_build_test(fused_bn_update_grad, shape_list, dtype_list, op_attrs=op_attrs, kernel_name="fused_bn_update_grad",
                              polyhedral=poly_sch, attrs=attrs)

    head, data_sum, in_bn, output, expect = gen_data(shape, out_shape, dtype, out_dtype, layout)
    outputs = [output, output]
    inputs = [head, data_sum, in_bn]
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