# Copyright 2019 Huawei Technologies Co., Ltd
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

"""operator dsl fuction: matmul"""

import logging
import numpy as np
import akg
import akg.tvm
from akg.utils import kernel_exec as utils
from tests.common.test_run.ascend.matmul_run import extract_dim, get_converted_shapes
from tests.common.test_run.ascend.matmul_run import matmul_data
from tests.common.base import get_rtol_atol
from tests.common.tensorio import compare_tensor

logging.basicConfig(level=logging.DEBUG)

BLOCK_REDUCE = 16


def output_shape_compute(left_shape, right_shape, left_format, right_format, out_format, trans_a, trans_b):
    if len(left_shape) < 4 or len(right_shape) < 4:
        raise RuntimeError("matmul left matrix and right matrix should be fractal shape")
    format_list = ["zZ", "nZ", "zN"]
    if not(left_format in format_list):
        raise RuntimeError("matmul left_format only support %s" % (",".join(format_list)))
    if not(right_format in format_list):
        raise RuntimeError("matmul right_format only support %s" % (",".join(format_list)))
    if not(out_format in format_list):
        raise RuntimeError("matmul out_format only support %s" % (",".join(format_list)))

    batch = left_shape[:-4]
    # left matrix default is zZ
    mo, ko, mi, ki = left_shape[-4:]
    if trans_a:
        ko, mo, ki, mi = left_shape[-4:]
    if left_format == "nZ":
        mo, ko, ki, mi = left_shape[-4:]
        if trans_a:
            ko, mo, mi, ki = left_shape[-4:]
    elif left_format == "zN":
        ko, mo, mi, ki = left_shape[-4:]
        if trans_a:
            mo, ko, ki, mi = left_shape[-4:]

    # right matrix default is zZ
    ko, no, ki, ni = right_shape[-4:]
    if trans_b:
        no, ko, ni, ki = right_shape[-4:]
    if right_format == "nZ":
        ko, no, ni, ki = right_shape[-4:]
        if trans_b:
            no, ko, ki, ni = right_shape[-4:]
    elif right_format == "zN":
        no, ko, ki, ni = right_shape[-4:]
        if trans_b:
            ko, no, ni, ki = right_shape[-4:]

    output_shape = tuple(batch) + (mo, no, mi, ni)
    if out_format == "nZ":
        output_shape = tuple(batch) + (mo, no, ni, mi)
    elif out_format == "zN":
        output_shape = tuple(batch) + (no, mo, mi, ni)
    k = ko * ki
    return output_shape, k


def matmul4D_compute(x, y, bias_value, out_dtype, left_format, right_format, out_format, transpose_x=False, transpose_y=False, attrs=None):
    # for gemv use transpose of AB --> gevm trans(trans(B) * trans(A))

    data_dtype = x.dtype.lower()
    check_list = ["int8", "uint8", "float16", "float32", "int32"]
    if not (data_dtype in check_list):
        raise RuntimeError("matmul_cce ony supports %s while dtype is %s" % (",".join(check_list), x.dtype))

    if bias_value is None:
        bias_name = ''
        bias = 0
    else:
        bias_name = bias_value.name
        bias = 0 if bias_value is None else 1

    output_shape_zN, k = output_shape_compute(x.shape, y.shape, left_format, right_format, "zN", transpose_x, transpose_y)
    output_shape_zZ, k = output_shape_compute(x.shape, y.shape, left_format, right_format, "zZ", transpose_x, transpose_y)
    shape_A = x.shape
    shape_B = y.shape
    key = ()

    key += (tuple(shape_A), tuple(shape_B), bias, left_format, right_format, out_format, transpose_x, transpose_y, x.dtype, out_dtype)
    hash_key = str(key)
    # bypass 2 left matrix ddr -> l0
    # bypass 1 right matrix ddr -> l0
    bypass_list = [0, 1, 2]
    bypass = 0
    if attrs is not None and 'bypass' in attrs:
        bypass = attrs['bypass']

    if not (bypass in bypass_list):
        raise RuntimeError("matmul_cce ony supports %s while bypass is %d" % (",".join(str(bypass_list)), bypass))

    def matmul_compute(output_shape, adj_x, adj_y, left_format, right_format, output_format, x, y, k, *indices):
        N = len(output_shape)
        # reduce axis
        ko = akg.tvm.reduce_axis((0, k // BLOCK_REDUCE), name='ko')
        ki = akg.tvm.reduce_axis((0, BLOCK_REDUCE), name='ki')
        if output_format == "zN":
            if left_format == "zZ":
                x_indices = indices[:(N - 4)] + indices[(N - 3):(N - 2)] + (ko,) + indices[(N - 2):(N - 1)] + (ki,)
                if adj_x:
                    x_indices = indices[:(N - 4)] + (ko,) + indices[(N - 3):(N - 2)] + (ki,) + indices[(N - 2):(N - 1)]
            elif left_format == "zN":
                x_indices = indices[:(N - 4)] + (ko,) + indices[(N - 3):(N - 1)] + (ki,)

            if right_format == "nZ":
                y_indices = indices[:(N - 4)] + (ko, ) + indices[(N - 4):(N - 3)] + indices[(N - 1):] + (ki,)
                if adj_y:
                    y_indices = indices[:(N - 4)] + indices[(N - 4):(N - 3)] + (ko, ki) + indices[(N - 1):]
            elif right_format == "zZ":
                y_indices = indices[:(N - 4)] + (ko, ) + indices[(N - 4):(N - 3)] + (ki,) + indices[(N - 1):]
                if adj_y:
                    y_indices = indices[:(N - 4)] + indices[(N - 4):(N - 3)] + (ko,) + indices[(N - 1):] + (ki,)
        return akg.lang.ascend.mmad((x(*x_indices) * y(*y_indices)).astype(out_dtype), axis=[ko, ki])

    if left_format == "zZ":
        data_trans = "N"
        data_trans_block = "N"
        if transpose_x:
            data_trans = "Y"
    elif left_format == "zN":
        if transpose_x:
            raise RuntimeError("left matrix format zN, only support transpose_x = False")
        data_trans_block = "Y"
        data_trans = "Y"

    if right_format == "nZ":
        weight_trans = "N"
        weight_trans_block = "N"
        weight_trans_block_in = "N"
        if transpose_y:
            weight_trans = "Y"
    elif right_format == "zZ":
        if not transpose_y:
            weight_trans_block_in = "Y"
            weight_trans_block = "N"
            weight_trans = "Y"
        elif transpose_y:
            weight_trans = "Y"
            weight_trans_block = "Y"
            weight_trans_block_in = "N"

    result_matmul = akg.tvm.compute(output_shape_zN, lambda *indices: matmul_compute(output_shape_zN, transpose_x, transpose_y, left_format, right_format, "zN",
                                                                                     x, y, k, *indices), name="resMatmul",
                                    attrs={
                                        "pragma_gemm_data": x.name,
                                        "pragma_data_transpose": data_trans,
                                        "pragma_data_transpose_block": data_trans_block,
                                        "pragma_gemm_weight": y.name,
                                        "pragma_weight_transpose": weight_trans,
                                        "pragma_weight_transpose_block": weight_trans_block,
                                        "pragma_weight_transpose_block_inner": weight_trans_block_in,
                                        "pragma_conv_bypass_l1": bypass,
                                        "bias": bias_name,
                                    })

    def matmul_reshape(shape, result_matmul, *indices):
        N = len(shape)
        new_indices = indices[:(N - 4)] + indices[(N - 3):(N - 2)] + indices[(N - 4):(N - 3)] + indices[(N - 2):]
        return result_matmul(*new_indices)

    if out_format == "zZ":
        result = akg.tvm.compute(output_shape_zZ, lambda *indices: matmul_reshape(output_shape_zZ, result_matmul, *indices), name="result")
    else:
        result = result_matmul

    def bias_compute(output_shape, result, bias, output_format, *indices):
        N = len(output_shape)
        # reduce axis
        if output_format == "zN":
            bias_indices = indices[:(N - 4)] + indices[(N - 4):(N - 3)] + (0, 0) + indices[(N - 1):]
        elif output_format == "zZ":
            bias_indices = indices[:(N - 4)] + (0,) + indices[(N - 3):(N - 2)] + (0,) + indices[(N - 1):]
        return result(*indices) + bias(*bias_indices)
    if bias == 1:
        if out_format == "zN":
            out = akg.tvm.compute(output_shape_zN, lambda *indices: bias_compute(output_shape_zN, result, bias_value, out_format, *indices),
                                  name="output")
        elif out_format == "zZ":
            out = akg.tvm.compute(output_shape_zZ, lambda *indices: bias_compute(output_shape_zZ, result, bias_value, out_format, *indices),
                                  name="output")
    else:
        out = result

    return out


def matmul(x, y, b, out_dtype, left_format="zZ", right_format="nZ", out_format="zN", transpose_x=False, transpose_y=False, has_bias=False, attrs=None):
    """
    Computes matrix multiplication x * y + b.

    Args:
        x: akg.tvm.Tensor of type int8, uint8, float16, float32, int32. Left matrix.
        y: akg.tvm.Tensor of same type as x. Right matrix.
        b: akg.tvm.Tensor of same type as x. Bias tensor.
        out_dtype: str. Data type of output tensor.
        left_format: str. Data format of left matrix. Supported data format list ["zZ", "nZ", "zN"].
        right_format: str. Data format of right matrix. Supported data format list ["zZ", "nZ", "zN"].
        out_format: str. Data format of output tensor. Supported data format list ["zZ", "nZ", "zN"].
        transpose_x: Boolean. Specifies whether x is transposed or not.
        transpose_y: Boolean. Specifies whether y is transposed or not.
        has_bias: Boolean. Specifies whether bias tensor exists or not.
        attrs: Dict. Used in matmul computation.

    Note:
        before call matmul, 2d to Fractal is needed.

    Returns:
        akg.tvm.Tensor with type out_dtype.
    """
    # vc_util.ops_dtype_check([x.dtype, y.dtype], vc_util.DtypeForDavinci.ALL_FLOAT)
    # shape_x = [shape_element.value for shape_element in x.shape]
    # vc_util.check_shape(shape_x)
    # shape_y = [shape_element.value for shape_element in y.shape]
    # vc_util.check_shape(shape_y)
    """
    m = akg.tvm.var("I2")
    n = akg.tvm.var("I1")
    k = akg.tvm.var("KO")
    x = akg.tvm.placeholder((1, m, k, 16, 16), name='A', dtype=x.dtype)
    y = akg.tvm.placeholder((1, k, n, 16, 16), name='B', dtype=y.dtype)
    # b = akg.tvm.placeholder((1, m, 16, n, 16), name='BIAS')
    """

    out = matmul4D_compute(x, y, b, out_dtype, left_format, right_format, out_format, transpose_x, transpose_y, attrs)
    attr_map = {"pragma_rmselfdep": False, "enable_post_poly_loop_partition": False, "enable_multicore": False, "enable_isolate_loop": False, "enable_double_buffer": False}
    return out, attr_map


def dynamic_matmul_compile(shape_x, shape_y, bias, left_format, right_format, output_format, adj_x, adj_y, dtype, out_dtype, kernel_name, attrs):
    batch_tuple, m, k, n = extract_dim(shape_x, shape_y, adj_x, adj_y)
    m = akg.tvm.var("I2")
    n = akg.tvm.var("I1")
    k = akg.tvm.var("KO")
    x = akg.tvm.placeholder((1, m, k, 16, 16), name='A', dtype=dtype)
    y = akg.tvm.placeholder((1, k, n, 16, 16), name='B', dtype=dtype)
    """
    m = (m + 15) // 16 * 16
    n = (n + 15) // 16 * 16
    k = (k + 15) // 16 * 16
    """
    shape_xx, shape_yy, bias_shape, out_shape, k = get_converted_shapes(m, n, k, batch_tuple, adj_x, adj_y, bias,
                                                                        left_format, right_format, output_format)
    input_shapes = [shape_xx, shape_yy, bias_shape]
    input_types = [dtype, dtype, dtype]

    has_bias = False
    if bias == 1:
        has_bias = True
    op_attrs = [out_dtype, left_format, right_format, output_format, adj_x, adj_y, has_bias, attrs]
    if has_bias == False:
        input_shapes = [x, y]
        input_types = [dtype, dtype]
        op_attrs = [None, out_dtype, left_format, right_format, output_format, adj_x, adj_y, has_bias, attrs]
    return utils.op_build_test(matmul, input_shapes, input_types, op_attrs, kernel_name, attrs)


def matmul_execute(shape_x, shape_y, bias, left_format, right_format, out_format, adj_x, adj_y, dtype, out_dtype, kernel_name, attrs):
    '''
    There are four types of fractal format in Davinci core: zZ, zN, nZ, nN
    general matmul format
    left_trans: False right_trans False: zZ * nZ = zN
    left_trans: True  right_trans False: nN * nZ = zN
    left_trans: False right_trans True : zZ * zN = zN
    left_trans: True  right_trans True : nN * zN = zN

    Now we need to support: zN * nZ = zN
    use left_format to specify, left matrix data format
    use right_format to specify, right matrix data format
    '''
    batch_tuple, m, k, n = extract_dim(shape_x, shape_y, adj_x, adj_y)
    m = (m + 15) // 16 * 16
    n = (n + 15) // 16 * 16
    k = (k + 15) // 16 * 16
    shape_xx, shape_yy, bias_shape, out_shape, k = get_converted_shapes(m, n, k, batch_tuple, adj_x, adj_y, bias, left_format, right_format, out_format)
    mod = dynamic_matmul_compile(shape_x, shape_y, bias, left_format, right_format, out_format, adj_x, adj_y, dtype, out_dtype, kernel_name, attrs)
    # Generate data
    m_x, m_y, bench_mark, bias_data = matmul_data(batch_tuple, m, k, n, dtype, out_dtype, bias, adj_x, adj_y, left_format, right_format, out_format)

    # mod launch
    output = np.full(out_shape, np.nan, out_dtype)
    if bias == 0:
        output = utils.mod_launch(mod, (m_x, m_y, output, 1, 1, 1, 1, 1, 1, 1, 1, 1), outputs=(2,), expect=bench_mark)
    elif bias == 1:
        output = utils.mod_launch(mod, (m_x, m_y, bias_data, output), expect=bench_mark)

    # compare result
    rtol, atol = get_rtol_atol("matmul", dtype)
    compare_result = compare_tensor(output, bench_mark, rtol=rtol, atol=atol, equal_nan=True)
    # compare_result = utils.result_compare(output, bench_mark, r_tol=5e-3)
    return (m_x, m_y), output, bench_mark, compare_result


def get_shape(bs, m, n, k, trans_a, trans_b):
    if trans_a:
        shape_A = (*bs, k, m)
    else:
        shape_A = (*bs, m, k)
    if trans_b:
        shape_B = (*bs, n, k)
    else:
        shape_B = (*bs, k, n)
    return shape_A, shape_B


if __name__ == "__main__":
    # auto tiling
    print(matmul_execute((256, 1024), (1024, 64), 0, "zZ", "nZ", "zN", False, False, "float16", "float32", "dynamic_matmul",{}))

    # manual setdim
    # matmul_execute((256, 1024), (1024, 64), 0, "zZ", "nZ", "zN", False, False, "float16", "float32",
    #                "dynamic_matmul", {"dim": "0 0 1 1 0 0 1 1 0 0 16 16 0 0 16 16 0 0 1 1"})
