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
from akg.ops.nn.cpu import depthwise_conv2d_nchwc
import akg
import numpy as np
from akg.topi.nn.util import get_pad_tuple
from tests.common.gen_random import random_gaussian
from akg.utils import kernel_exec as utils
from akg.utils.result_analysis import target_profiling
from akg.utils.format_transform import to_tvm_nd_array
from topi.testing import depthwise_conv2d_python_nchw, dilate_python
from .cpu_test_utils import unpack_nchwc_to_nchw_python, unpack_kcrsxy_to_kcrs_python, pack_nchw_to_nchwc_python

support_list = {"float32": np.float32}
support_layout_format = {"NCHW", "NCHWc"}


def gen_data(shape_data, shape_weight, stride, padding, dilation, dtype, data_layout, output_layout):
    data = random_gaussian(shape_data, miu=1, sigma=0.1).astype(
        support_list[dtype])
    weight = random_gaussian(
        shape_weight, miu=1, sigma=0.1).astype(support_list[dtype])
    expect = compute_np_depthwise_conv2d(
        data, weight, stride, padding, dilation, dtype, data_layout, output_layout)
    output = np.full(expect.shape, np.nan, dtype)
    return data, weight, output, expect


def compute_np_depthwise_conv2d(data, weight, stride, padding, dilation, dtype, data_layout="NCHWc", output_layout="NCHWc"):

    if data_layout == "NCHWc":
        data_nchw = unpack_nchwc_to_nchw_python(data, dtype)
        weight_nchw = unpack_kcrsxy_to_kcrs_python(weight, dtype)
    elif data_layout == "NCHW":
        data_nchw = data
        weight_nchw = weight
    else:
        raise ValueError("Only layout NCHWc/NCHW supported currently")

    hd, wd = dilation
    weight_nchw_with_dilation = dilate_python(weight_nchw, (1, 1, hd, wd))
    res_nchw = depthwise_conv2d_python_nchw(
        data_nchw, weight_nchw_with_dilation, stride, padding)

    if output_layout == "NCHWc":
        res_nchwc = pack_nchw_to_nchwc_python(
            res_nchw, c_inner=weight.shape[-1], dtype=dtype)
        return res_nchwc

    return res_nchw


def depthwise_conv2d_run(shape_data, shape_weight, stride=(1, 1), padding="SAME", dilation=(1, 1), dtype="float32",
                         data_layout="NCHWc", output_layout="NCHWc", poly_sch=True, attrs=None):
    if data_layout not in support_layout_format or output_layout not in support_layout_format:
        raise ValueError("Only layout NCHWc/NCHW supported")

    default_attrs = {"enable_auto_fuse": False, "pragma_enable_conv2d_direct": True}
    attrs = {} if attrs == None else attrs
    attrs.update(default_attrs)
    attrs["target"] = attrs.get("target", "llvm")
    c_inners = [-1, -1] # use default
    kh = shape_weight[2]
    padding_num = get_pad_tuple(padding, (kh, kh))
    op_attrs = [stride, padding_num, dilation, dtype, output_layout, c_inners]

    mod = utils.op_build_test(depthwise_conv2d_nchwc, (shape_data, shape_weight), (dtype, dtype),
                              op_attrs=op_attrs, attrs=attrs,
                              kernel_name="depthwise_conv2d_nchw_auto", polyhedral=poly_sch)

    data, weight, output, expect = gen_data(
        shape_data, shape_weight, stride, padding, dilation, dtype, data_layout, output_layout)
    args = (data, weight, output)
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

    if attrs.get("profiling", False):
        data, weight, output = to_tvm_nd_array(
            [data, weight, output], akg.tvm.context(target_name, 0))
        target_profiling(mod, data, weight, output,
                         target=target_name, repeat_time=attrs.get("repeat_times", 1000))
    return (data, weight), output, expect, res
