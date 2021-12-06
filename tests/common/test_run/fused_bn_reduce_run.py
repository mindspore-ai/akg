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
from tests.common.test_op.resnet.fused_bn_reduce import fused_bn_reduce

def compute_fused_bn_reduce(data, layout, out_dtype):
    if layout == "NCHW":
        data = np.transpose(data, axes=(0, 2, 3, 1))

    n, h, w, c = data.shape

    inter_dtype = 'float32'
    if data.dtype != inter_dtype:
        data = data.astype(inter_dtype)
    data = np.reshape(data, (n * h * w, c))

    output1 = np.sum(data, axis=0)
    output1 = output1.astype(out_dtype)

    squared = np.multiply(data, data)
    output2 = np.sum(squared, axis=0)
    output2 = output2.astype(out_dtype)

    return [output1, output2]

def gen_data(in_shape, in_dtype, layout, out_dtype):
    support_list = {"float16": np.float16, "float32": np.float32}
    data = random_gaussian(in_shape, miu=1, sigma=0.1).astype(support_list[in_dtype])
    expect = compute_fused_bn_reduce(data, layout, out_dtype)
    output = np.full(expect[0].shape, 0.0, out_dtype)
    output = [output, output]
    return data, output, expect

def fused_bn_reduce_run(in_shape, layout='NHWC', in_dtype='float16', out_dtype='float32', poly_sch=True, attrs=None):
    if not attrs:
        attrs = {"target": "cuda"}
    attrs.update({"enable_akg_reduce_lib": True, "enable_atomic_add": True})
    if layout != "NHWC" and layout != "NCHW":
        raise NotImplementedError(
            'Layout not supported {} '.format(layout))

    op_attrs = [layout, out_dtype]
    mod = utils.op_build_test(fused_bn_reduce, [in_shape], [in_dtype], kernel_name="fused_bn_reduce",
        op_attrs=op_attrs, polyhedral=poly_sch, attrs=attrs)

    data, outputs, expect = gen_data(in_shape, in_dtype, layout, out_dtype)
    inputs = [data]
    arglist = inputs + outputs
    output = utils.mod_launch(mod, arglist, outputs=tuple(range(-len(outputs), 0)), expect=expect)

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