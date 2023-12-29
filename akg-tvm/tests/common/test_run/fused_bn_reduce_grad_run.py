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
from tests.common.gen_random import random_gaussian
from akg.utils import kernel_exec as utils
from akg.utils.result_analysis import target_profiling
from akg.utils.format_transform import to_tvm_nd_array
from tests.common.test_op.resnet.fused_bn_reduce_grad import fused_bn_reduce_grad

def compute_fused_bn_reduce_grad(data, inter_dtype, layout, out_dtype):

    data0 = data[0]
    data1 = data[1]
    data2 = data[2]
    data3 = data[3]
    data4 = data[4]
    data5 = data[5]
    data6 = data[6]
    data7 = data[7]

    if layout == "NCHW":
        data3 = np.transpose(data3, axes=(0, 2, 3, 1))
        data7 = np.transpose(data7, axes=(0, 2, 3, 1))

    n, h, w, c = data3.shape

    data3 = data3.astype(inter_dtype)
    data7 = data7.astype(inter_dtype)

    out1 = data4 * data5 / (n * h * w)
    out2 = data3 * (n * h * w) - data2
    out3 = (data7 - data6 / (n * h * w)) * data1 / data0
    output = out1 * (out2 - out3)
    output = output.astype(out_dtype)

    if layout == "NCHW":
        output = np.transpose(output, axes=(0, 3, 1, 2))

    return output

def gen_data(in_shape, in_dtype, inter_dtype, layout, out_dtype):

    if layout == "NHWC":
        num_channel = in_shape[3]
    else:
        num_channel = in_shape[1]

    data = [np.nan] * 8
    data[0] = random_gaussian([num_channel], miu=1, sigma=0.1).astype(inter_dtype)
    data[1] = random_gaussian([num_channel], miu=1, sigma=0.1).astype(inter_dtype)
    data[2] = random_gaussian([num_channel], miu=1, sigma=0.1).astype(inter_dtype)
    data[3] = random_gaussian(in_shape, miu=1, sigma=0.1).astype(in_dtype)
    data[4] = random_gaussian([num_channel], miu=1, sigma=0.1).astype(inter_dtype)
    data[5] = random_gaussian([num_channel], miu=1, sigma=0.1).astype(inter_dtype)
    data[6] = random_gaussian([num_channel], miu=1, sigma=0.1).astype(inter_dtype)
    data[7] = random_gaussian(in_shape, miu=1, sigma=0.1).astype(in_dtype)

    expect = compute_fused_bn_reduce_grad(data, inter_dtype, layout, out_dtype)
    output = np.full(expect.shape, np.nan, out_dtype)

    return data, output, expect

def fused_bn_reduce_grad_run(in_shape, layout='NHWC', in_dtype="float16", out_dtype='float16', poly_sch=True, attrs=None):
    if not attrs:
        attrs = {"target": "cuda"}
    if layout != "NHWC" and layout != "NCHW":
        raise NotImplementedError(
            'Layout not supported {} '.format(layout))

    inter_dtype = 'float32'
    inputs, output, expect = gen_data(in_shape, in_dtype, inter_dtype, layout, out_dtype)
    input_shape_list = [i.shape for i in inputs]
    input_dtype_list = [inter_dtype] * 3 + [in_dtype] + [inter_dtype] * 3 + [in_dtype]
    op_attrs = [layout, out_dtype]
    mod = utils.op_build_test(
            fused_bn_reduce_grad, input_shape_list, input_dtype_list,
            kernel_name="fused_bn_reduce_grad", op_attrs=op_attrs, polyhedral=poly_sch, attrs=attrs)

    outputs = [output]
    arglist = inputs + outputs
    output = utils.mod_launch(mod, arglist, expect=expect)

    res = np.allclose(output, expect, rtol=5e-03, atol=1.e-8)
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
        arglist = to_tvm_nd_array(arglist, akg.tvm.context(target_name, 0))
        target_profiling(mod, *arglist, target=target_name, repeat_time=attrs["repeat_times"])
    return inputs, outputs, expect, res