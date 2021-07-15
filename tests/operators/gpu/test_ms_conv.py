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
from akg.ops.math_gpu.conv import conv
from akg.ops.math_gpu.tensorcore_conv import tensorcore_conv
from tests.common.gen_random import random_gaussian
from akg.utils import kernel_exec as utils
from akg.utils.result_analysis import gpu_profiling
from akg.utils.format_transform import to_tvm_nd_array

support_list = {"float16": np.float16, "float32": np.float32}

def has_pad(padding):
    p_l, p_r, p_t, p_b = padding
    return not(p_l == 0 and p_r == 0 and p_t == 0 and p_b == 0)

def gen_data(shape_data, shape_weight, layout, stride, padding, dilation, dtype, out_dtype):
    data = random_gaussian(shape_data, miu=1, sigma=0.1).astype(
        support_list[dtype])
    weight = random_gaussian(
        shape_weight, miu=1, sigma=0.1).astype(support_list[dtype])
    expect = compute_np_conv2d(
        data, weight, layout, stride, padding, dilation, dtype, out_dtype)
    output = np.full(expect.shape, np.nan, out_dtype)
    return data, weight, output, expect

def compute_np_conv2d(data, weight, layout, stride, padding, dilation, dtype, out_dtype):
    if layout == "NHWC":
        data = np.transpose(data, (0, 3, 1, 2))
        weight = np.transpose(weight, (0, 3, 1, 2))
    n, c, h, w = data.shape
    out_c, c, kh, kw = weight.shape
    s_h, s_w = stride
    d_h, d_w = dilation
    p_t, p_b, p_l, p_r = padding
    """
    initialize data with padding
    """
    shape_data_pad = (n, c, h + p_t + p_b, w + p_l + p_r)
    data_pad = np.zeros(shape_data_pad).astype(support_list[dtype])
    if has_pad(padding):
        data_pad[:, :, p_t:p_t+h, p_l:p_l+w] = data
    else:
        data_pad = data
    """
    compute expect
    """
    whd = (kh - 1) * d_h + 1
    wwd = (kw - 1) * d_w + 1
    out_h = (h + p_t + p_b - whd) // s_h + 1
    out_w = (w + p_l + p_r - wwd) // s_w + 1
    out_shape = (n, out_c, out_h, out_w)
    expect = np.zeros(out_shape).astype(support_list[out_dtype])
    for f in range(out_c):
        for i in range(out_h):
            for j in range(out_w):
                expect[:, f, i, j] = np.sum(
                    data_pad[:, :, i*s_h: i*s_h +whd: d_h, j*s_w: j*s_w+wwd: d_w]
                    * weight[f, :, :, :],
                    axis=(1, 2, 3))
    if layout == "NHWC":
        expect = np.transpose(expect, (0, 2, 3, 1))
    print("expect shape is ", np.shape(expect))
    return expect

def test_ms_conv(shape_data, shape_weight, stride=(1,1), padding=(0,0,0,0), dilation=(1,1), dtype="float16",
        out_dtype="float16", layout="NHWC", tensor_core=True, poly_sch=True, attrs=None):
    if layout != "NHWC" and layout != "NCHW":
        raise ValueError("Layout NHWC and NCHW supported")
    use_tensor_core = False
    if tensor_core and layout == "NHWC" and dtype == "float16":
        use_tensor_core = True
    op_attrs = [stride, padding, dilation]
    default_attrs = {"target": "cuda", "enable_auto_fuse": False}
    if attrs:
        default_attrs.update(attrs)
    if use_tensor_core:
        op_attrs += [out_dtype]
        default_attrs.update({"pragma_enable_matmul": True, "pragma_enable_conv_tensor_core": True})
        if poly_sch:
            mod = utils.op_build_test(
                tensorcore_conv, (shape_data, shape_weight), (dtype, dtype),
                op_attrs=op_attrs, attrs=default_attrs, kernel_name="tensorcore_conv_auto")
    elif poly_sch:
        mod = utils.op_build_test(
            conv, (shape_data, shape_weight), (dtype, dtype),
            op_attrs=op_attrs, attrs=default_attrs, kernel_name="conv_auto")

    data, weight, output, expect = gen_data(
        shape_data, shape_weight, layout, stride, padding, dilation, dtype, out_dtype)
    args = (data, weight, output)
    output = utils.mod_launch(mod, args, expect=expect)
    rtol = 1e-3 if dtype == "float16" else 1e-4
    atol = 1e-3 if dtype == "float16" else 1e-4
    res = np.allclose(output, expect, rtol=rtol, atol=atol)
    print("Test {}".format("Pass"))
    if not res:
        print("Error cuda:===================================")
        print(mod.imported_modules[0].get_source())
        raise AssertionError("Test fail")

    data, weight, output, expect = to_tvm_nd_array(
        [data, weight, output, expect])
    gpu_profiling(mod, data, weight, output, expect, repeat_time=2)
