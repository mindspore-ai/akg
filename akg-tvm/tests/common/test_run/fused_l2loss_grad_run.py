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
from tests.common.test_op.resnet.fused_l2loss_grad import fused_l2loss_grad

def gen_data(data_shape, dtype):
    data = random_gaussian(data_shape, miu=1, sigma=0.1).astype(dtype)
    return data

def compute_py(data_f16, data_f32, layout, fill_data):
    if layout == "NCHW":
        data_f16 = np.transpose(data_f16, axes=(0, 2, 3, 1))
    elif layout != "NHWC":
        raise NotImplementedError('Layout not supported {} '.format(layout))

    data_f16 = data_f16.astype('float32')
    data_constant = np.array([float(fill_data)])
    expect = np.multiply(data_constant, data_f32)
    expect = np.add(expect, data_f16)
    output = np.full(np.shape(expect), np.nan, 'float32')
    return expect, output

def fused_l2loss_grad_run(shape, layout, fill_data=4e-05, poly_sch=True, attrs=None):
    if not attrs:
        attrs = {"target": "cuda"}

    data_1 = gen_data(shape, 'float16')
    data_2 = gen_data(shape, 'float32')

    expect, output = compute_py(data_1, data_2, layout, fill_data)
    input_list = [shape, shape]
    dtype_list = ['float16', 'float32']
    op_attrs = [layout, fill_data]
    mod = utils.op_build_test(fused_l2loss_grad, input_list, dtype_list, kernel_name="fused_l2loss_grad",
                              op_attrs=op_attrs, polyhedral=poly_sch, attrs=attrs)

    args = [data_1, data_2, output]
    output = utils.mod_launch(mod, args, expect = expect)
    res = np.allclose(output, expect, rtol=5e-03, atol=1e-8)
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
        args = to_tvm_nd_array(args, akg.tvm.context(target_name, 0))
        target_profiling(mod, *args, target=target_name, repeat_time=attrs["repeat_times"])
    return (data_1, data_2), output, expect, res