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
# limitations under the License.

"""run function for quantized_max_pool"""

import numpy as np

from tests.common.tensorio import compare_tensor
from akg.utils import kernel_exec as utils
from tests.common.test_run.ascend.maxpool_run import benchmark as maxpool_benchmark
from tests.common.gen_random import random_gaussian
from tests.common.base import get_rtol_atol
from tests.common.test_op.ascend.quantized_max_pool import quantized_max_pool
from tests.common.test_run.ascend.quantized_avg_pool_run import compare_int

def benchmark(data, aided_inputs, ksize, strides, padding, data_format,
              quant_algo, *_):
    """caculate output for quantized maxpool"""
    hw_indices = [1, 2] if data_format in ("NHWC",) else [2, 3]
    kernel = [ksize[i] for i in hw_indices]
    stride = [strides[i] for i in hw_indices]

    shape = data.shape
    in_h, in_w = [shape[i] for i in hw_indices]

    if kernel[0] >= in_h and kernel[1] >= in_w:
        kernel[0] = in_h
        kernel[1] = in_w
        padding = "VALID"
        stride = [1, 1]

    out = maxpool_benchmark(data, kernel, stride, padding)
    if quant_algo is not None:
        scale_req, offset_req = aided_inputs
        out = out * scale_req[0]
        if quant_algo[0] == 1:
            out = np.add(out, offset_req[0])

    out_type = "float16" if quant_algo is None else (
        "int8" if quant_algo[0] == 0 else "uint8")
    if out_type in ("int8", "uint8"):
        out = np.maximum(np.iinfo(out_type).min, out)
        out = np.minimum(np.iinfo(out_type).max, out)
    return out.astype(out_type)

def quantized_max_pool_run(shape, dtype1, shape_list, dtype2, ksize, strides,
                           padding, data_format, quant_algo,
                           scale_mode, scale_sqrt, attrs):
    """run function"""
    if not isinstance(shape_list, (list, tuple, type(None))):
        raise RuntimeError("shape_list should be a list, tuple or None!")
    op_attrs = [ksize, strides, padding, data_format,
                quant_algo, scale_mode, scale_sqrt]
    if shape_list is None:
        mod = utils.op_build_test(quantized_max_pool, [shape], [dtype1],
                                  op_attrs=[None] + op_attrs,
                                  kernel_name='quantized_maxpool', attrs=attrs)
    else:
        mod = utils.op_build_test(quantized_max_pool,
                                  [shape, shape_list], [dtype1, dtype2],
                                  op_attrs=op_attrs,
                                  kernel_name='quantized_maxpool', attrs=attrs)
    expect, inputs, out_buf = gen_data(shape, dtype1, shape_list, dtype2, ksize,
                                       strides, padding, data_format, quant_algo,
                                       scale_mode, scale_sqrt)
    output = utils.mod_launch(mod, (*inputs, *out_buf), expect=expect)
    rtol, atol = get_rtol_atol("quantized_maxpool", dtype1)
    if expect.dtype in ("int8", "uint8"):
        cmp_res = compare_int(output, expect)
    else:
        cmp_res = compare_tensor(output, expect, rtol=rtol, atol=atol)
    return inputs, output, expect, cmp_res

def gen_data(shape, dtype1, shape_list, dtype2, ksize, strides, padding,
             data_format, quant_algo, scale_mode, scale_sqrt):
    """generate data"""
    data = random_gaussian(shape).astype(dtype1)
    inputs = [data]
    aided_inputs = []
    if shape_list is not None:
        # Here use the data's max and min to calculate the requantize scale and
        # offset, while the requantize parameter is different in practice.
        in_max = data.max()
        in_min = data.min()
        out_type = "int8" if quant_algo[0] == 0 else "uint8"
        diff = in_max - in_min
        scale = (np.iinfo(out_type).max - np.iinfo(out_type).min) / diff
        offset = - in_min / diff
        quantize_params = [scale, offset]
        for value, shp in zip(quantize_params, shape_list):
            aided_inputs.append(
                np.broadcast_to(np.array([value]).astype(dtype2), shp))
        inputs = inputs + aided_inputs
    expect = benchmark(data, aided_inputs, ksize, strides, padding, data_format,
                       quant_algo, scale_mode, scale_sqrt)
    out_buf = np.full(expect.shape, 0, expect.dtype)
    return expect, inputs, (out_buf,)
