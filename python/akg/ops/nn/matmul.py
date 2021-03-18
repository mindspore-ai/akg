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

"""operator dsl function: matmul"""

import logging
import akg.tvm
import akg
from akg.utils import kernel_exec as utils
from akg.utils import custom_tiling as ct_util
from akg.utils import validation_check as vc_util
from akg.ops.math import cast

logging.basicConfig(level=logging.DEBUG)

matmul_set_dim_map = {
     str(((1, 16, 49, 16, 16), (1, 49, 49, 16, 16), 0, 'zZ', 'zZ', 'zZ', False, False, 'float16')) : ([(1,1),(2,2),(16,16),(16,16),(49,49)], {"bypass" : 0}),
     str(((1, 16, 49, 16, 16), (1, 49, 16, 16, 16), 0, 'zZ', 'zZ', 'zZ', False, False, 'float16')) : ([(2,2),(2,2),(16,16),(16,16),(49,49)], {"bypass" : 0}),
     str(((1, 2, 64, 16, 16), (1, 2, 64, 16, 16), 0, 'zZ', 'zZ', 'zZ', True, False, 'float16')) : ([(2,2),(64,64),(16,16),(16,16),(2,2)], {"bypass" : 0}),
     str(((1, 2, 128, 16, 16), (1, 2, 128, 16, 16), 0, 'zZ', 'zZ', 'zZ', True, False, 'float16')) : ([(2,2),(64,64),(16,16),(16,16),(2,2)], {"bypass" : 0}),

     # bert best tile
     # (16, 1024), (16, 1024)
     str(((64, 1, 16, 16), (64, 1, 16, 16), 0, "zN", "zN", "zN", False, True, "float16")) : ([(16,16),(16,16),(32,32)], {"bypass" : 2}),
     # (8192, 4096), (8192, 1024)
     str(((256, 512, 16, 16), (64, 512, 16, 16), 0, "zN", "zN", "zN", True, False, "float16")) : ([(8,8),(16,16),(16,16),(16,16),(8,1)], {"bypass" : 0}),
     # (8192, 1024), (1024, 4096)
     str(((64, 512, 16, 16), (256, 64, 16, 16), 0, "zN", "zN", "zN", False, False, "float16")) : ([(16,16),(8,4),(16,16),(16,16),(64,8)], {"bypass" : 0}),
     # (16, 16), (16, 1024)
     str(((1, 1, 16, 16), (64, 1, 16, 16), 0, "zN", "zN", "zN", True, False, "float16")) : ([(8,8),(16,16),(16,16),(16,16)], {"bypass" : 0}),
     # (1216, 1024), (1024, 1024)
     str(((64, 76, 16, 16), (64, 64, 16, 16), 0, "zN", "zN", "zN", False, False, "float16")) : ([(4,4),(19,19),(16,16),(16,16),(4,1)], {"bypass" : 0}),
     # (8192, 4096), (4096, 1024)
     str(((256, 512, 16 ,16), (64, 256, 16, 16), 0, "zN", "zN", "zN", False, False, "float16")) : ([(8,8),(32,32),(16,16),(16,16),(2,1)], {"bypass" : 0}),
     # (8192, 1024), (4096, 1024)
     str(((64, 512, 16, 16), (64, 256, 16, 16), 0, "zN", "zN", "zN", False, True, "float16")) : ([(8,8),(32,32),(16,16),(16,16),(2,1)], {"bypass" : 0}),
     # (8192, 1024), (8192, 4096)
     str(((64, 512, 16, 16), (256, 512, 16, 16), 0, "zN", "zN", "zN", True, False, "float16")) : ([[8, 8], [32, 32], [16, 16], [16, 16], [16, 2]], {"bypass": 0}),
     # (1216, 1024), (1024, 1024)
     str(((64, 76, 16, 16), (64, 64, 16, 16), 0, "zN", "zN", "zN", False, True, "float16")) : ([(8,8),(19,19),(16,16),(16,16),(16,1)], {"bypass" : 2}),
     # (8192, 1024), (1024, 1024)
     str(((64, 512, 16, 16), (64, 64, 16, 16), 0, "zN", "zN", "zN", False, False, "float16")) : ([(16,4),(16,8),(16,16),(16,16),(64,16)], {"bypass" : 0}),
     # (1216, 30522), (30522, 1024)
     str(((1908, 76, 16, 16), (64, 1908, 16, 16), 0, "zN", "zN", "zN", False, False, "float16")) : ([(8,8),(19,19),(16,16),(16,16),(6,1)], {"bypass" : 0}),
     # (1216, 30522), (1216, 1024)
     str(((1908, 76, 16, 16), (64, 76, 16, 16), 0, "zN", "zN", "zN", True, False, "float16")) : ([(4,4),(18,18),(16,16),(16,16),(2,2)], {"bypass" : 0}),
     # (1216, 1024), (30522, 1024)
     str(((64, 76, 16, 16), (64, 1908, 16, 16), 0, "zN", "zN", "zN", False, True, "float16")) : ([(9,9),(19,19),(16,16),(16,16),(64,1)], {"bypass" : 0}),
     # (8192, 1024), (8192, 1024)
     str(((64, 512, 16, 16), (64, 512, 16, 16), 0, "zN", "zN", "zN", True, False, "float16")) : ([(4,4),(16,16),(16,16),(16,16),(16,4)], {"bypass" : 0}),
     # (1216, 1024), (1216, 1024)
     str(((64, 76, 16, 16), (64, 76, 16, 16), 0, "zN", "zN", "zN", True, False, "float16")) : ([(16,16),(8,8),(16,16),(16,16),(4,2)], {"bypass" : 0}),
     # (16, 1024), (16, 1024)
     str(((64, 1, 16, 16), (64, 1, 16, 16), 0, "zN", "zN", "zN", True, False, "float16")) : ([(8,8),(2,2),(16,16),(16,16),(16,16)], {"bypass" : 0}),
     # (16, 1024), (1024, 1024)
     str(((64, 1, 16, 16), (64, 64, 16, 16), 0, "zN", "zN", "zN", False, True, "float16")) : ([(8,8),(16,16),(16,16),(32,8)], {"bypass" : 2}),
     # (16, 16), (16, 1024)
     str(((1, 1, 16, 16), (64, 1, 16, 16), 0, "zN", "zN", "zN", False, False, "float16")) : ([(8,8),(16,16),(16,16),(16,16)], {"bypass" : 0}),
     # (8192, 1024), (1024, 1024)
     str(((64, 512, 16, 16), (64, 64, 16, 16), 0, "zN", "zN", "zN", False, True, "float16")) : ([(16,8),(8,8),(16,16),(16,16),(64,8)], {"bypass" : 1}),
     # (8192, 4096), (1024, 4096)
     str(((256, 512, 16, 16), (256, 64, 16, 16), 0, "zN", "zN", "zN", False, True, "float16")) : ([(8,8),(32,32),(16,16),(16,16),(1,1)], {"bypass" : 0}),
     # (16, 16, 512, 512), (16, 16, 512, 64)
     str(((16, 16, 32, 32, 16, 16), (16, 16, 4, 32, 16, 16), 0, "zN", "zN", "zN", False, False, "float16")) : ([(1,1),(1,1),(4,4),(16,16),(16,16),(16,16),(16,4)], {"bypass" : 0}),
     str(((16, 16, 32, 32, 16, 16), (16, 16, 4, 32, 16, 16), 0, "zN", "zN", "zN", True, False, "float16")) : ([(1,1),(1,1),(4,4),(16,16),(16,16),(16,16),(16,4)], {"bypass" : 0}),
     # (16, 16, 512, 64), (16, 16, 512, 64)
     str(((16, 16, 4, 32, 16, 16), (16, 16, 4, 32, 16, 16), 0, "zN", "zN", "zN", False, True, "float16")) : ([(1,1),(1,1),(32,4),(32,32),(16,16),(16,16),(4,4)], {"bypass" : 0}),
     # (24, 16, 512, 512), (24, 16, 512, 64)
     str(((24, 16, 32, 32, 16, 16), (24, 16, 4, 32, 16, 16), 0, 'zN', 'zN', 'zN', False, False, 'float16')) : ([(1,1),(1,1),(4,4),(8,1),(16,16),(16,16),(32,32)], {"bypass" : 0}),
     str(((24, 16, 32, 32, 16, 16), (24, 16, 4, 32, 16, 16), 0, 'zN', 'zN', 'zN', True, False, 'float16')) : ([(1,1),(1,1),(4,4),(32,32),(16,16),(16,16),(8,1)], {"bypass" : 0}),
     # (24, 16, 512, 64), (24, 16, 512, 64)
     str(((24, 16, 4, 32, 16, 16), (24, 16, 4, 32, 16, 16), 0, 'zN', 'zN', 'zN', False, True, 'float16')) : ([(1,1),(1,1),(32,16),(32,16),(16,16),(16,16),(4,2)], {"bypass" : 0}),
}


def matmul_set_dim(A, B, b, out_dtype, left_format, right_format, output_format, adj_x, adj_y):
    shape_A = A.shape[1:5] if len(A.shape) == 5 else A.shape
    shape_B = B.shape[1:5] if len(B.shape) == 5 else B.shape
    bias = 0 if b is None else 1
    key = ()

    key += (tuple(shape_A), tuple(shape_B), bias, left_format, right_format, output_format, adj_x, adj_y, A.dtype)
    hash_key = str(key)
    if hash_key in matmul_set_dim_map:
        configs = matmul_set_dim_map[hash_key]
        if isinstance(configs, tuple):
            tiles = configs[0]
        else:
            tiles = configs
        set_dims = ct_util.set_dims(tiles)
        return set_dims, hash_key

    return "", hash_key


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

    key += (tuple(shape_A), tuple(shape_B), bias, left_format, right_format, out_format, transpose_x, transpose_y, x.dtype)
    hash_key = str(key)
    # bypass 2 left matrix ddr -> l0
    # bypass 1 right matrix ddr -> l0
    bypass_list = [0, 1, 2]
    bypass = 0
    if attrs is not None and 'bypass' in attrs:
        bypass = attrs['bypass']
    elif hash_key in matmul_set_dim_map:
        configs = matmul_set_dim_map[hash_key]
        if isinstance(configs, tuple):
            if len(configs) > 1 and "bypass" in configs[1]:
                bypass = configs[1]["bypass"]

    if not (bypass in bypass_list):
        raise RuntimeError("matmul_cce ony supports %s while bypass is %d" % (",".join(str(bypass_list)), bypass))

    def matmul_compute(output_shape, adj_x, adj_y, left_format, right_format, output_format, x, y, k, *indices):
        N = len(output_shape)
        # reduce axis
        ko = akg.tvm.reduce_axis((0, k // utils.BLOCK_REDUCE), name='ko')
        ki = akg.tvm.reduce_axis((0, utils.BLOCK_REDUCE), name='ki')
        if output_format == "zN":
            if left_format == "zZ":
                x_indices = indices[:(N - 4)] + indices[(N - 3):(N - 2)] + (ko,) + indices[(N - 2):(N - 1)] + (ki,)
                if adj_x:
                    x_indices = indices[:(N - 4)] + (ko,) + indices[(N - 3):(N - 2)] + (ki,) + indices[(N - 2):(N - 1)]
            elif left_format == "zN":
                x_indices = indices[:(N - 4)] + (ko,) + indices[(N - 3):(N - 2)] + indices[(N-2):(N-1)] + (ki,)
                if adj_x:
                    x_indices = indices[:(N - 4)] + indices[(N - 3):(N - 2)] + (ko,) + (ki,) + indices[(N-2):(N-1)]

            if right_format == "nZ":
                y_indices = indices[:(N - 4)] + (ko, ) + indices[(N - 4):(N - 3)] + indices[(N - 1):] + (ki,)
                if adj_y:
                    y_indices = indices[:(N - 4)] + indices[(N - 4):(N - 3)] + (ko, ki) + indices[(N - 1):]
            elif right_format == "zZ":
                y_indices = indices[:(N - 4)] + (ko, ) + indices[(N - 4):(N - 3)] + (ki,) + indices[(N - 1):]
                if adj_y:
                    y_indices = indices[:(N - 4)] + indices[(N - 4):(N - 3)] + (ko,) + indices[(N - 1):] + (ki,)
            elif right_format == "zN":
                y_indices = indices[:(N - 4)] + indices[(N - 4):(N - 3)] + (ko,) + (ki,) + indices[(N - 1):]
                if adj_y:
                    y_indices = indices[:(N - 4)] + (ko,) + indices[(N - 4):(N - 3)] + indices[(N - 1):] + (ki,)

        return akg.lang.cce.mmad((x(*x_indices) * y(*y_indices)).astype("float32"), axis=[ko, ki])


    if left_format == "zZ":
        data_trans = "N"
        data_trans_block = "N"
        data_trans_block_in = "N"
        if transpose_x:
            data_trans = "Y"
    elif left_format == "zN":
        data_trans = "Y"
        data_trans_block = "Y"
        data_trans_block_in = "N"
        if transpose_x:
            data_trans = "Y"
            data_trans_block = "N"
            data_trans_block_in = "Y"

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
    elif right_format == "zN":
        weight_trans = "Y"
        weight_trans_block = "N"
        weight_trans_block_in = "N"
        if transpose_y:
            weight_trans = "N"
            weight_trans_block = "N"
            weight_trans_block_in = "N"

    result_matmul = akg.tvm.compute(output_shape_zN, lambda *indices: matmul_compute(output_shape_zN, transpose_x, transpose_y, left_format, right_format, "zN",
                                                                                 x, y, k, *indices), name="resMatmul",
                                attrs={
        "pragma_gemm_data": x.name,
        "pragma_data_transpose": data_trans,
        "pragma_data_transpose_block": data_trans_block,
        "pragma_data_transpose_block_inner": data_trans_block_in,
        "pragma_gemm_weight": y.name,
        "pragma_weight_transpose": weight_trans,
        "pragma_weight_transpose_block": weight_trans_block,
        "pragma_weight_transpose_block_inner": weight_trans_block_in,
        "pragma_conv_bypass_l1": bypass,
        "bias": bias_name,
    })

    if bias_value == None and out_dtype == "float16":
        result_matmul = cast.cast(result_matmul, out_dtype)

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
            bias_indices = indices[N - 4] * utils.BLOCK_OUT + indices[N - 1]
        elif output_format == "zZ":
            bias_indices = indices[N - 3] * utils.BLOCK_OUT + indices[N - 1]
        return result(*indices) + bias(bias_indices)
    if bias == 1:
        if bias_value.dtype == "float16":
            bias_value = cast.cast(bias_value, "float32")
        if out_format == "zN":
            out = akg.tvm.compute(output_shape_zN, lambda *indices: bias_compute(output_shape_zN, result, bias_value, out_format, *indices),
                              name="output")
        elif out_format == "zZ":
            out = akg.tvm.compute(output_shape_zZ, lambda *indices: bias_compute(output_shape_zZ, result, bias_value, out_format, *indices),
                              name="output")
        if out_dtype == "float16" and bias_value.dtype == "float32":
            out = cast.cast(out, out_dtype)
    else:
        out = result

    return out

def matmul(x, y, b, out_dtype, left_format="zZ", right_format="nZ", out_format="zN", transpose_x=False, transpose_y=False, attrs=None):
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
        attrs: Dict. Used in matmul computation.

    Note:
        before call matmul, 2d to Fractal is needed.

    Returns:
        akg.tvm.Tensor with type out_dtype.
    """
    vc_util.ops_dtype_check([x.dtype, y.dtype], vc_util.DtypeForDavinci.ALL_FLOAT)
    shape_x = [shape_element.value for shape_element in x.shape]
    vc_util.check_shape(shape_x)
    shape_y = [shape_element.value for shape_element in y.shape]
    vc_util.check_shape(shape_y)
    if left_format not in ["zZ", "zN"]:
        raise ValueError("unsupport left_format now: %s" % left_format)
    if right_format not in ["nZ", "zZ", "zN"]:
        raise ValueError("unsupport right_format now: %s" % right_format)
    if out_format not in ["zN", "zZ"]:
        raise ValueError("unsupport out_format now: %s" % out_format)

    out = matmul4D_compute(x, y, b, out_dtype, left_format, right_format, out_format, transpose_x, transpose_y, attrs)
    attr_map = {"pragma_rmselfdep": False}

    dims_info, _ = matmul_set_dim(x, y, b, out_dtype, left_format, right_format, out_format, transpose_x, transpose_y)
    attr_map["dim"] = dims_info

    return out, attr_map
