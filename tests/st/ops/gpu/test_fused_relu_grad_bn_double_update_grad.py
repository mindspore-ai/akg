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
from tests.common.gen_random import random_gaussian
from tests.common.test_op.resnet.fused_relu_grad_bn_double_update_grad import fused_relu_grad_bn_double_update_grad


def gen_data(data_shape, dtype):
    data = random_gaussian(data_shape, miu=1, sigma=0.1).astype(dtype)
    return data


def compute_py(data_1, data_2, data_3, data_4, data_5, data_6, data_7, layout='NHWC'):
    data_tmp1 = np.array([0.0]).astype('float16')
    data_tmp2 = np.greater(data_7, data_tmp1)
    data_tmp3 = np.add(data_5, data_6)
    data_tmp4 = np.where(data_tmp2, data_tmp3, data_tmp1)
    data_tmp5 = data_tmp4.astype('float32')
    data_tmp7 = np.sum(data_tmp5, axis=(0, 1, 2))

    n, h, w, c = np.shape(data_7)
    data_tmp8 = data_2.astype('float32')
    data_tmp9 = np.array([1.0 / (n * h * w)]).astype('float32')
    data_tmp10 = np.multiply(data_1, data_tmp9)
    data_tmp11 = np.broadcast_to(data_tmp10, data_tmp8.shape)
    data_tmp12 = np.subtract(data_tmp8, data_tmp11)
    data_tmp13 = np.multiply(data_tmp5, data_tmp12)
    data_tmp15 = np.sum(data_tmp13, axis=(0, 1, 2))

    data_tmp16 = data_4.astype('float32')
    data_tmp17 = np.multiply(data_3, data_tmp9)
    data_tmp18 = np.broadcast_to(data_tmp17, data_tmp16.shape)
    data_tmp19 = np.subtract(data_tmp16, data_tmp18)
    data_tmp20 = np.multiply(data_tmp5, data_tmp19)
    data_tmp22 = np.sum(data_tmp20, axis=(0, 1, 2))

    out_shape = [c]

    return data_tmp7, data_tmp15, data_tmp22, out_shape


def test_fused_relu_grad_bn_double_update_grad(shape_f16, shape_f32, layout='NHWC', poly_sch=False):
    data_1 = gen_data(shape_f32, 'float32')
    data_2 = gen_data(shape_f16, 'float16')
    data_3 = gen_data(shape_f32, 'float32')
    data_4 = gen_data(shape_f16, 'float16')
    data_5 = gen_data(shape_f16, 'float16')
    data_6 = gen_data(shape_f16, 'float16')
    data_7 = gen_data(shape_f16, 'float16')
    shape_list = [shape_f32, shape_f16, shape_f32, shape_f16, shape_f16, shape_f16, shape_f16]
    dtype_list = ['float32', 'float16', 'float32', 'float16', 'float16', 'float16', 'float16']
    data_list = [data_1, data_2, data_3, data_4, data_5, data_6, data_7]
    data_tmp7, data_tmp15, data_tmp22, out_shape = compute_py(data_1, data_2, data_3, data_4, data_5, data_6, data_7,
                                                              layout)
    expect = [data_tmp7, data_tmp15, data_tmp22]
    output = np.full(out_shape, 0.0, 'float32')
    output = [output, output, output]

    if poly_sch:
        mod = utils.op_build(fused_relu_grad_bn_double_update_grad, shape_list, dtype_list, op_attrs=[layout],
                             kernel_name="fused_relu_grad_bn_double_update_grad", attrs={"target": "cuda"})

    output = utils.mod_launch(mod, (data_1, data_2, data_3, data_4, data_5, data_6, data_7, *output),
                              outputs=tuple(range(-len(output), 0)), expect=expect)

    res = True
    res &= np.allclose(output[0], expect[0], rtol=5e-03, atol=1e-8)
    res &= np.allclose(output[1], expect[1], rtol=5e-03, atol=1e-8)
    res &= np.allclose(output[2], expect[2], rtol=5e-03, atol=1e-8)
    print("Test {}".format("Pass" if res else "Fail"))
    if not res:
        print("Error cuda:========================")
        print(mod.imported_modules[0].get_source())
        raise AssertionError("Test fail")

    return True
