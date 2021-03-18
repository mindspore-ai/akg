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
from tests.common.test_op.resnet.fused_bn_follow_relu_avgpool import fused_bn_follow_relu_avgpool


def compute_expect(data, inter_dtype, layout, out_dtype):
    xi_conv2d1 = data[0]
    beta = data[1]
    gamma = data[2]
    bn_update = data[3]
    bn_reduce = data[4]
    xi_conv2d = data[5]

    if layout == "NCHW":
        xi_conv2d1 = np.transpose(xi_conv2d1, axes=(0, 2, 3, 1))
        xi_conv2d = np.transpose(xi_conv2d, axes=(0, 2, 3, 1))

    n, h, w, c = xi_conv2d1.shape
    add0 = (xi_conv2d.astype(inter_dtype) - bn_reduce / (n * h * w)) * bn_update * gamma + beta
    output = xi_conv2d1 + add0.astype(xi_conv2d1.dtype)
    output = np.maximum(output, 0)
    output = np.reshape(output.astype(inter_dtype), (n, h * w, c))
    output = np.mean(output, axis=1)
    output = output.astype(out_dtype)

    return output


def gen_data(in_shape, in_dtype, inter_dtype, layout, out_dtype):
    if layout == "NHWC":
        num_channel = in_shape[3]
    else:
        num_channel = in_shape[1]

    data = [np.nan] * 6
    data[0] = random_gaussian(in_shape, miu=1, sigma=0.1).astype(in_dtype)
    data[1] = random_gaussian([num_channel], miu=1, sigma=0.1).astype(inter_dtype)
    data[2] = random_gaussian([num_channel], miu=1, sigma=0.1).astype(inter_dtype)
    data[3] = random_gaussian([num_channel], miu=1, sigma=0.1).astype(inter_dtype)
    data[4] = random_gaussian([num_channel], miu=1, sigma=0.1).astype(inter_dtype)
    data[5] = random_gaussian(in_shape, miu=1, sigma=0.1).astype(in_dtype)

    expect = compute_expect(data, inter_dtype, layout, out_dtype)
    output = np.full(expect.shape, np.nan, out_dtype)

    return data, output, expect


def test_fused_bn_follow_relu_avgpool(in_shape, in_dtype='float16', layout='NHWC', out_dtype='float16', poly_sch=False):
    if layout != "NHWC" and layout != "NCHW":
        raise NotImplementedError(
            'Layout not supported {} '.format(layout))

    inter_dtype = 'float32'
    inputs, output, expect = gen_data(in_shape, in_dtype, inter_dtype, layout, out_dtype)
    input_shape_list = [i.shape for i in inputs]
    input_dtype_list = [in_dtype] + [inter_dtype] * 4 + [in_dtype]
    op_attrs = [layout, out_dtype]
    if poly_sch:
        mod = utils.op_build_test(fused_bn_follow_relu_avgpool, input_shape_list, input_dtype_list,
                                  kernel_name="fused_bn_follow_relu_avgpool", op_attrs=op_attrs,
                                  attrs={"target": "cuda",
                                         "enable_akg_reduce_lib": True, "enable_atomic_add": True})

    outputs = [output]
    arglist = inputs + outputs
    output = utils.mod_launch(mod, arglist, expect=expect)

    res = np.allclose(output, expect, rtol=5e-03, atol=1.e-8)
    print("Test {}".format("Pass" if res else "Fail"))
    if not res:
        print("Error cuda:========================")
        print(mod.imported_modules[0].get_source())
        raise AssertionError("Test fail")

    return True
