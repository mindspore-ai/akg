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
from tests.common.test_op.resnet.fused_relu_grad_bn_reduce_grad import fused_relu_grad_bn_reduce_grad


def gen_data(shape, dtype):
    return random_gaussian(shape, miu=1, sigma=0.1).astype(dtype)

def compute_py(data_1, data_2, data_3, data_4, data_5, data_6, data_7, data_8, data_9, layout):
    data_tmp1 = np.multiply(data_4, data_5)
    n, h, w, c = np.shape(data_9)
    data_tmp2 = np.full(np.shape(data_tmp1), 1.0/(n*h*w), 'float32')
    data_tmp3 = np.multiply(data_tmp1, data_tmp2)

    data_tmp5 = np.full(np.shape(data_9), 0.0, 'float16')
    data_tmp6 = np.greater(data_9, data_tmp5)
    data_tmp7 = np.where(data_tmp6, data_8, data_tmp5)

    data_tmp8 = data_tmp7.astype('float32')
    data_tmp9 = np.full(np.shape(data_9), n*h*w, 'float32')
    data_tmp10 = np.multiply(data_tmp8, data_tmp9)

    data_tmp12 = np.subtract(data_tmp10, data_3)

    data_tmp14 = data_7.astype('float32')
    data_tmp15 = np.multiply(data_6, data_tmp2)
    data_tmp17 = np.subtract(data_tmp14, data_tmp15)
    data_tmp18 = np.multiply(data_2, data_tmp17)
    data_tmp20 = np.divide(data_tmp18, data_1)
    data_tmp21 = np.subtract(data_tmp12, data_tmp20)
    data_tmp22 = np.multiply(data_tmp3, data_tmp21)

    expect = data_tmp22.astype('float16')
    output = np.full(np.shape(expect), np.nan, 'float16')

    return expect, output

def fused_relu_grad_bn_reduce_grad_run(shape_1, shape_2, layout='NHWC', poly_sch=True, attrs=None):
    if not attrs:
        attrs = {"target": "cuda"}
    data_1 = gen_data(shape_1, 'float32')
    data_2 = gen_data(shape_1, 'float32')
    data_3 = gen_data(shape_1, 'float32')
    data_4 = gen_data(shape_1, 'float32')
    data_5 = gen_data(shape_1, 'float32')
    data_6 = gen_data(shape_1, 'float32')
    data_7 = gen_data(shape_2, 'float16')
    data_8 = gen_data(shape_2, 'float16')
    data_9 = gen_data(shape_2, 'float16')

    expect, output = compute_py(data_1, data_2, data_3, data_4, data_5, data_6, data_7, data_8, data_9, layout)
    input_list = [shape_1, shape_1, shape_1, shape_1, shape_1, shape_1, shape_2, shape_2, shape_2]
    dtype_list = ['float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float16', 'float16', 'float16']
    op_attrs = [layout]
    mod = utils.op_build_test(fused_relu_grad_bn_reduce_grad, input_list, dtype_list, kernel_name="fused_relu_grad_bn_reduce_grad",
                            op_attrs=op_attrs, polyhedral=poly_sch, attrs=attrs)

    args = [data_1, data_2, data_3, data_4, data_5, data_6, data_7, data_8, data_9, output]
    output = utils.mod_launch(mod, args, expect=expect)
    res = np.allclose(output, expect, rtol=5e-03, atol=1e-08)
    print("Test {}".format("Pass" if res else "Failed"))
    target_name = attrs["target"].split()[0]
    if not res:
        mod_source = mod
        if target_name != "llvm":
            mod_source = mod.imported_modules[0]
        print("Error {}:========================".format(target_name))
        print(mod_source.get_source())
        raise AssertionError("Test fail")

    if attrs["profiling"]:
        inputs = to_tvm_nd_array([data_1, data_2, data_3, data_4, data_5, data_6, data_7, data_8, data_9, output],
                                akg.tvm.context(target_name, 0))
        target_profiling(mod, *inputs, target=target_name, repeat_time=attrs["repeat_times"])
    return (data_1, data_2, data_3, data_4, data_5, data_6, data_7, data_8, data_9), output, expect, res