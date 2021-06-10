# Copyright 2020 Huawei Technologies Co., Ltd
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
from akg.ops.math_gpu.conv_fusion import conv_fusion
from tests.common.gen_random import random_gaussian
from akg.utils import kernel_exec as utils
from akg.utils.result_analysis import gpu_profiling
from akg.utils.format_transform import to_tvm_nd_array

def has_pad(padding):
    p_l, p_r, p_t, p_b = padding
    return not(p_l == 0 and p_r == 0 and p_t == 0 and p_b == 0)

def gen_data(shape_data, shape_filter, stride, padding, dilation, dtype, out_dtype):
    support_list = {"float16": np.float16, "float32": np.float32}
    data = random_gaussian(shape_data, miu=1, sigma=0.1).astype(support_list[dtype])
    filter_ = random_gaussian(shape_filter, miu=1, sigma=0.1).astype(support_list[dtype])

    n, c, h, w = shape_data
    c_out, c, kh, kw = shape_filter
    s_h, s_w = stride
    d_h, d_w = dilation
    p_l, p_r, p_t, p_b = padding

    out_h = (h + p_t + p_b - kh) // s_h + 1
    out_w = (w + p_l + p_r - kw) // s_w + 1
    out_shape = (n, c_out, out_h, out_w)
    shape_data_pad = (n, c, h + p_t + p_b, w + p_l + p_r)

    """
    initialization data with padding
    """
    data_pad = np.zeros(shape_data_pad).astype(support_list[dtype])
    if has_pad(padding):
        data_pad[:,:,p_t:p_t+h,p_l:p_l+w] = data
    else:
        data_pad = data

    whd = (kh - 1) * d_h + 1
    wwd = (kw - 1) * d_w + 1
    expect = np.zeros(out_shape).astype(support_list[out_dtype])
    for f in range(c_out):
        for i in range(out_h):
            for j in range(out_w):
                expect[:,f,i,j] = np.sum(data_pad[:,:,i*s_h:i*s_h+whd:d_h,j*s_w:j*s_w+wwd:d_w]*filter_[f,:,:,:], axis=(1,2,3))

    output = np.full(expect.shape, np.nan, out_dtype)
    print("expect shape is ", np.shape(expect))

    return data, filter_, output, expect

def fusion_gen_data(shape_data, shape_filter1, shape_filter2, stride1, stride2, padding1, padding2, dilation1, dilation2, dtype, out_dtype):
    data, filter_, output1, expect_data = gen_data(shape_data, shape_filter1, stride1, padding1, dilation1, dtype, out_dtype)
    support_list = {"float16": np.float16, "float32": np.float32}
    filter2 = random_gaussian(shape_filter2, miu=1, sigma=0.1).astype(support_list[dtype])

    n, c, h, w = expect_data.shape
    c_out, c, kh, kw = shape_filter2
    s_h, s_w = stride2
    d_h, d_w = dilation2
    p_l, p_r, p_t, p_b = padding2

    out_h = (h + p_t + p_b - kh) // s_h + 1
    out_w = (w + p_l + p_r - kw) // s_w + 1
    out_shape = (n, c_out, out_h, out_w)
    shape_data_pad = (n, c, h + p_t + p_b, w + p_l + p_r)

    """
    initialization data with padding
    """
    data_pad = np.zeros(shape_data_pad).astype(support_list[dtype])
    if has_pad(padding2):
        data_pad[:,:,p_t:p_t+h,p_l:p_l+w] = expect_data
    else:
        data_pad = expect_data

    whd = (kh - 1) * d_h + 1
    wwd = (kw - 1) * d_w + 1
    expect = np.zeros(out_shape).astype(support_list[out_dtype])
    for f in range(c_out):
        for i in range(out_h):
            for j in range(out_w):
                expect[:,f,i,j] = np.sum(data_pad[:,:,i*s_h:i*s_h+whd:d_h,j*s_w:j*s_w+wwd:d_w]*filter_[f,:,:,:], axis=(1,2,3))

    output = np.full(expect.shape, np.nan, out_dtype)
    print("expect shape is ", np.shape(expect))

    return data, filter_, filter2, output, expect

def test_ms_conv_fusion(shape_data, shape_filter1, shape_filter2, stride1, stride2, padding1, padding2, dilation1, dilation2, dtype, out_dtype="float32", poly_sch=True):
    op_attrs = [stride1, stride2, padding1, padding2, dilation1, dilation2]
    attrs = {"target":"cuda", "enable_auto_fuse":False, "shared_memory_tensors":"out input_1 input_2 input_3", "pragma_disable_loop_fusion": True,
             "dim": "3 0 1 1 3 1 1 1 3 2 4 4 3 3 52 52 3 4 64 64"}

    if poly_sch:
        mod = utils.op_build_test(conv_fusion, (shape_data, shape_filter1, shape_filter2), (dtype, dtype, dtype), op_attrs=op_attrs, attrs=attrs, kernel_name="conv_fusion_auto")

    data, weight1, weight2, output, expect = fusion_gen_data(shape_data, shape_filter1, shape_filter2, stride1, stride2, padding1, padding2, dilation1, dilation2, dtype, out_dtype)
    args = (data, weight1, weight2, output)
    output = utils.mod_launch(mod, args, expect=expect)
    res = np.allclose(output, expect, rtol=5e-3, atol=1.e-8)
    print("Test {}".format("Pass"))
    if not res:
        print("Error cuda:===================================")
        print(mod.imported_modules[0].get_source())
        raise AssertionError("Test fail")

    data, weight1, weight2, output, expect = to_tvm_nd_array([data, weight1, weight2, output, expect])
    gpu_profiling(mod, data, weight1, weight2, output, expect, repeat_time=2)
