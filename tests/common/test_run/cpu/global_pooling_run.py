# Copyright 2022 Huawei Technologies Co., Ltd
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
from akg.ops.nn.cpu import global_pooling
import akg
import tvm
import topi
import numpy as np
from akg.topi.util import get_const_tuple
from akg.utils import kernel_exec as utils
from akg.utils.result_analysis import target_profiling
from akg.utils.format_transform import to_tvm_nd_array
from tests.common.gen_random import random_gaussian

support_list = {"float32": np.float32}
support_layout_format = {"NCHW", "NCHWc"}


def gen_data(shape_data, pool_type, dtype, data_layout):
    if data_layout == "NHWC":
        pool_idxs = (1, 2)
    else:
        # NCHW or NCHWc
        pool_idxs = (2, 3)

    input0 = tvm.placeholder(shape_data, name='input0')
    output0 = topi.nn.global_pool(input0, pool_type=pool_type)
    a_np = random_gaussian(shape_data, miu=1, sigma=0.1).astype(
        support_list[dtype])
    if pool_type == 'avg':
        b_np = np.mean(a_np, axis=pool_idxs, keepdims=True)
    elif pool_type == 'max':
        b_np = np.max(a_np, axis=pool_idxs, keepdims=True)

    output_np = np.zeros(shape=b_np.shape).astype(dtype)

    return a_np, output_np, b_np


def global_pooling_run(shape_data, pool_type, dtype,
                       data_layout="NCHWc", poly_sch=True, attrs=None):

    default_attrs = {"enable_auto_fuse": True, "polytops_parameter_shifting": True, "polytops_enable_skewing": False}
    attrs = {} if attrs == None else attrs
    attrs.update(default_attrs)
    attrs["target"] = attrs.get("target", "llvm")
    op_attrs = [pool_type, data_layout]

    mod = utils.op_build_test(global_pooling, (shape_data,), (dtype,),
                              op_attrs=op_attrs, attrs=attrs,
                              kernel_name="global_pooling_" + pool_type + "_auto", polyhedral=poly_sch)

    data, output, expect = gen_data(
        shape_data, pool_type, dtype, data_layout)
    args = (data, output)
    output = utils.mod_launch(mod, args, expect=expect)
    rtol = 1e-3 if dtype == "float16" else 1e-4
    atol = 1e-3 if dtype == "float16" else 1e-4
    res = np.allclose(output, expect, rtol=rtol, atol=atol)
    print("Test {}".format("Pass" if res else "Fail"))
    target_name = attrs["target"].split()[0]
    if not res:
        mod_source = mod
        if target_name != "llvm":
            mod_source = mod.imported_modules[0]
        print("Error {}:========================".format(target_name))
        print(mod_source.get_source())
        raise AssertionError("Test fail")

    attrs["profiling"] = True
    if attrs.get("profiling", False):
        data, output = to_tvm_nd_array(
            [data, output], akg.tvm.context(target_name, 0))
        target_profiling(mod, data, output,
                         target=target_name, repeat_time=attrs.get("repeat_times", 1000))
    return (data, ), output, expect, res
