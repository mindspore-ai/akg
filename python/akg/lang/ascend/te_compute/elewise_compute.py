#!/usr/bin/env python3
# coding: utf-8
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

"""elewise compute"""
from decorator import decorator
import akg.tvm
from akg.utils.validation_check import judge_var
from akg.utils.format_transform import get_shape_from_tensor
from akg.utils import validation_check as vc_util
from .cast_compute import cast
from .util import save_op_output_dtype, get_intr_types, is_cast_support
need_save_type = True


def set_is_need_save_dtype():
    global need_save_type
    need_save_type = False


name_index = [0]


def _auto_cast_of_elewise_one(func, arg, supported_types):
    temp_tensor = arg
    dtype = temp_tensor.dtype
    if dtype not in supported_types:
        if "float32" in supported_types and is_cast_support(dtype, "float32"):
            temp_tensor = cast(temp_tensor, "float32")
        else:
            temp_tensor = cast(temp_tensor, "float16")
    return func(temp_tensor)


def _auto_cast_of_elewise_two(func, args, supported_types):
    if isinstance(args[1], akg.tvm.tensor.Tensor):
        lhs = args[0]
        rhs = args[1]
        # get tensor from tuple(tensor, attrs)
        if isinstance(lhs, tuple):
            lhs = list(lhs)[0]
        dtype_l = lhs.dtype
        dtype_r = rhs.dtype

        lhs_t = lhs
        rhs_t = rhs
        if dtype_l not in supported_types or dtype_r not in supported_types or dtype_l != dtype_r:
            if "float32" in supported_types and is_cast_support(dtype_l, "float32")\
                    and is_cast_support(dtype_r, "float32"):
                lhs_t = cast(lhs, "float32")
                rhs_t = cast(rhs, "float32")
            else:
                lhs_t = cast(lhs, "float16")
                rhs_t = cast(rhs, "float16")

        return func(lhs_t, rhs_t)

    temp_tensor = args[0]
    if isinstance(temp_tensor, tuple):
        temp_tensor = list(temp_tensor)[0]
    scalar = args[1]
    dtype = temp_tensor.dtype
    if dtype not in supported_types:
        if "float32" in supported_types and is_cast_support(dtype, "float32"):
            temp_tensor = cast(temp_tensor, "float32")
            dtype = "float32"
        else:
            temp_tensor = cast(temp_tensor, "float16")
            dtype = "float16"

    tmp_arg = scalar
    scalar_type = judge_var(scalar)
    if scalar_type == "tvm_const" and scalar.dtype != dtype:
        tmp_arg = akg.tvm.const(scalar.value, dtype=dtype)

    if scalar_type == "python_const":
        tmp_arg = akg.tvm.const(scalar, dtype=dtype)
    return func(temp_tensor, tmp_arg)


def _auto_cast_of_elewise_three(func, args, supported_types):
    if isinstance(args[2], akg.tvm.tensor.Tensor):
        x = args[0]
        y = args[1]
        z = args[2]

        dtype_x = x.dtype
        dtype_y = y.dtype
        dtype_z = z.dtype

        x_t = x
        y_t = y
        z_t = z

        if dtype_x != dtype_y or dtype_x != dtype_z or dtype_z != dtype_y:
            raise RuntimeError("Input tensors must has same dtype!")

        if dtype_x not in supported_types:
            if "float32" in supported_types and is_cast_support(dtype_x, "float32"):
                x_t = cast(x, "float32")
                y_t = cast(y, "float32")
                z_t = cast(z, "float32")
            else:
                x_t = cast(x, "float16")
                y_t = cast(y, "float16")
                z_t = cast(z, "float16")

        return func(x_t, y_t, z_t)

    lhs = args[0]
    rhs = args[1]
    scalar = args[2]

    dtype_l = lhs.dtype
    dtype_r = rhs.dtype

    lhs_t = lhs
    rhs_t = rhs
    if dtype_l not in supported_types or dtype_r not in supported_types or dtype_l != dtype_r:
        if "float32" in supported_types and is_cast_support(dtype_l, "float32")\
                and is_cast_support(dtype_r, "float32"):
            lhs_t = cast(lhs, "float32")
            rhs_t = cast(rhs, "float32")
            dtype_l = "float32"
        else:
            lhs_t = cast(lhs, "float16")
            rhs_t = cast(rhs, "float16")
            dtype_l = "float16"

    tmp_arg = scalar
    scalar_type = judge_var(scalar)
    if scalar_type == "tvm_const" and scalar.dtype != dtype_l:
        tmp_arg = akg.tvm.const(scalar.value, dtype=dtype_l)

    if scalar_type == "python_const":
        tmp_arg = akg.tvm.const(scalar, dtype=dtype_l)
    return func(lhs_t, rhs_t, tmp_arg)


@decorator
def auto_cast_of_elewise(func, *args, **kwargs):
    """
    auto cast dectorator.

    Note:
        Before calling elewise api, check the input tensor is supported by the intr.
        If not supported, casting the input tensor to supported dtype.
        (On condition that the cast type is supported.
        If the cast type is not supported,raising a RuntimeError).
    """
    global need_save_type
    intr = func.__name__

    if need_save_type:
        save_op_output_dtype(func, *args)

    need_save_type = True
    supported_types = get_intr_types("Intrinsic_" + intr)

    if len(args) == 1:
        return _auto_cast_of_elewise_one(func, args[0], supported_types)
    if len(args) == 2:
        return _auto_cast_of_elewise_two(func, args, supported_types)
    if len(args) == 3:
        return _auto_cast_of_elewise_three(func, args, supported_types)
    return func(*args, **kwargs)


@auto_cast_of_elewise
def vmuls(raw_tensor, scalar):
    """
    multiply a tensor by a scalar, dtype of raw_tensor and scalar must be the same.

    Args:
        raw_tensor (tvm.tensor.Tensor): input.
        scalar (Union[float, int, tvm const]): input.

    Returns:
        tvm.tensor.Tensor, raw_tensor * scalar.
    """
    dtype = raw_tensor.dtype

    return single_elewise_op(raw_tensor, dtype, 'elewise_single_VS_mul', args=[scalar])


@auto_cast_of_elewise
def vadds(raw_tensor, scalar):
    """
    add a tensor by a scalar, dtype of raw_tensor and scalar must be the same.

    Args:
        raw_tensor (tvm.tensor.Tensor): input tensor.
        scalar (Union[float, int, tvm const]): input scalar.

    Returns:
        tvm.tensor.Tensor, raw_tensor + scalar.
    """
    dtype = raw_tensor.dtype

    return single_elewise_op(raw_tensor, dtype, 'elewise_single_VS_add', args=[scalar])


@auto_cast_of_elewise
def vlog(raw_tensor):
    """
    calculate ln(raw_tensor).

    Args:
        raw_tensor (tvm.tensor.Tensor): input tensor.

    Returns:
        tvm.tensor.Tensor, log(raw_tensor).
    """
    dtype = raw_tensor.dtype

    return single_elewise_op(raw_tensor, dtype, 'elewise_single_log')


@auto_cast_of_elewise
def vexp(raw_tensor):
    """
    calculate exp(raw_tensor).

    Args:
        raw_tensor (tvm.tensor.Tensor): input tensor.

    Returns:
        tvm.tensor.Tensor, exp(raw_tensor).
    """
    dtype = raw_tensor.dtype

    return single_elewise_op(raw_tensor, dtype, 'elewise_single_exp')


@auto_cast_of_elewise
def vabs(raw_tensor):
    """
    calculate abs(raw_tensor).

    Args:
        raw_tensor (tvm.tensor.Tensor): input tensor.

    Returns:
        tvm.tensor.Tensor, abs(raw_tensor).
    """
    dtype = raw_tensor.dtype

    return single_elewise_op(raw_tensor, dtype, 'elewise_single_abs')


@auto_cast_of_elewise
def vrec(raw_tensor):
    """
    calculate vrec(raw_tensor).

    Args:
        raw_tensor (tvm.tensor.Tensor): input tensor.

    Returns:
        tvm.tensor.Tensor, vrec(raw_tensor).
    """
    dtype = raw_tensor.dtype

    return single_elewise_op(raw_tensor, dtype, 'elewise_single_rec')


@auto_cast_of_elewise
def vrelu(raw_tensor):
    """
    calculate vrelu(raw_tensor).

    Args:
        raw_tensor (tvm.tensor.Tensor): input tensor.

    Returns:
        tvm.tensor.Tensor, vrelu(raw_tensor).
    """
    dtype = raw_tensor.dtype

    return single_elewise_op(raw_tensor, dtype, 'elewise_single_relu')


@auto_cast_of_elewise
def vnot(raw_tensor):
    """
    calculate vnot(raw_tensor).

    Args:
        raw_tensor (tvm.tensor.Tensor): input tensor.

    Returns:
        tvm.tensor.Tensor, vnot(raw_tensor).
    """
    dtype = raw_tensor.dtype

    return single_elewise_op(raw_tensor, dtype, 'elewise_single_not')


@auto_cast_of_elewise
def vsqrt(raw_tensor):
    """
    calculate vsqrt(raw_tensor).

    Args:
        raw_tensor (tvm.tensor.Tensor): input tensor.

    Returns:
        tvm.tensor.Tensor, vsqrt(raw_tensor).
    """
    dtype = raw_tensor.dtype

    return single_elewise_op(raw_tensor, dtype, 'elewise_single_sqrt')


@vc_util.check_input_type(akg.tvm.tensor.Tensor, str, str, (type(None), list))
def single_elewise_op(input_tensor, dtype, op, args=None):
    """factory method of single elewise operations."""
    in_tensor = input_tensor
    shape = get_shape_from_tensor(in_tensor)
    if op == "elewise_single_log":
        lambda_func = lambda *indice: akg.tvm.log(in_tensor(*indice))
    elif op == "elewise_single_exp":
        lambda_func = lambda *indice: akg.tvm.exp(in_tensor(*indice))
    elif op == "elewise_single_rec":
        lambda_func = lambda *indice: akg.tvm.const(1.0, dtype) / in_tensor(*indice)
    elif op == "elewise_single_VS_add":
        if not len(args) == 1:
            raise RuntimeError("The length of the args must be 1, but got %s" % len(args))
        lambda_func = lambda *indice: in_tensor(*indice) + args[0].astype(dtype)
    elif op == "elewise_single_VS_mul":
        if not len(args) == 1:
            raise RuntimeError("The length of the args must be 1, but got %s" % len(args))
        lambda_func = lambda *indice: in_tensor(*indice) * args[0].astype(dtype)
    elif op == "elewise_single_abs":
        lambda_func = lambda *indice: akg.tvm.abs(in_tensor(*indice))
    elif op == "elewise_single_relu":
        lambda_func = lambda *indice: akg.tvm.select(in_tensor(*indice) >=
                                                     0, in_tensor(*indice), akg.tvm.const(0, dtype=dtype))
    elif op == "elewise_single_not":
        lambda_func = lambda *indice: - in_tensor(*indice)
    elif op == "elewise_single_sqrt":
        lambda_func = lambda *indice: akg.tvm.sqrt(in_tensor(*indice))
    else:
        raise RuntimeError("operation %s not support yet" % op)

    name = op.split("_")[-1] + "_" + str(name_index[0])
    name_index[0] += 1

    with akg.tvm.tag_scope(op):
        tmp = akg.tvm.compute(shape, lambda_func, name=name)
    return tmp


@auto_cast_of_elewise
def vmul(lhs, rhs):
    """
    calculate elewise multiply.

    Args:
        lhs (tvm.tensor.Tensor): left hand tensor.
        rhs (tvm.tensor.Tensor): right hand tensor.

    Returns:
        tvm.tensor.Tensor, lhs * rhs.
    """
    return binary_elewise_op(lhs, rhs, op="elewise_binary_mul")


@auto_cast_of_elewise
def vadd(lhs, rhs):
    """
    calculate elewise add.

    Args:
        lhs (tvm.tensor.Tensor): left hand tensor.
        rhs (tvm.tensor.Tensor): right hand tensor.

    Returns:
        tvm.tensor.Tensor, lhs + rhs.
    """
    return binary_elewise_op(lhs, rhs, op="elewise_binary_add")


@auto_cast_of_elewise
def vsub(lhs, rhs):
    """
    calculate elewise sub.

    Args:
        lhs (tvm.tensor.Tensor): left hand tensor.
        rhs (tvm.tensor.Tensor): left hand tensor.

    Returns:
        tvm.tensor.Tensor, lhs - rhs.
    """
    return binary_elewise_op(lhs, rhs, op="elewise_binary_sub")


@auto_cast_of_elewise
def vmin(lhs, rhs):
    """
    calculate elewise compare, return the min one.

    Args:
        lhs (tvm.tensor.Tensor): left hand tensor.
        rhs (tvm.tensor.Tensor): right hand tensor.

    Return:
        tvm.tensor.Tensor, min value.
    """
    return binary_elewise_op(lhs, rhs, op="elewise_binary_min")


@auto_cast_of_elewise
def vmax(lhs, rhs):
    """
    calculate elewise compare, return the min one.

    Args:
        lhs (tvm.tensor.Tensor): left hand tensor.
        rhs (tvm.tensor.Tensor): left hand tensor.

    Returns:
        tvm.tensor.Tensor, max(lhs , rhs).
    """
    return binary_elewise_op(lhs, rhs, op="elewise_binary_max")


@auto_cast_of_elewise
def vor(lhs, rhs):
    """
    calculate elewise or op, return the or value.

    Args:
        lhs (tvm.tensor.Tensor): left hand tensor.
        rhs (tvm.tensor.Tensor): left hand tensor.

    Returns:
        tvm.tensor.Tensor, or(lhs , rhs).
    """
    return binary_elewise_op(lhs, rhs, op="elewise_binary_or")


@auto_cast_of_elewise
def vand(lhs, rhs):
    """
    calculate elewise and op, return the and value.

    Args:
        lhs (tvm.tensor.Tensor): left hand tensor.
        rhs (tvm.tensor.Tensor): left hand tensor.

    Returns:
        tvm.tensor.Tensor, max(lhs , rhs).
    """
    return binary_elewise_op(lhs, rhs, op="elewise_binary_and")


@auto_cast_of_elewise
def vaxpy(lhs, rhs, scalar):
    """
    calculate elewise scalar * lhs + rhs, return the min one.

    Args:
        lhs (tvm.tensor.Tensor): left hand tensor.
        rhs (tvm.tensor.Tensor): left hand tensor.
        scalar(tvm.tensor.Tensor): input scalar.

    Returns:
        tvm.tensor.Tensor, max(lhs , rhs).
    """
    return binary_elewise_op(lhs, rhs, op="elewise_binary_scalar_axpy", args=[scalar])


@vc_util.check_input_type(akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor)
def binary_elewise_op(lh_tensor, rh_tensor, op, args=None):
    """factory method of binary elewise operations."""
    shape_binary_elewise_op_check(lh_tensor, rh_tensor)
    if lh_tensor.dtype != rh_tensor.dtype:
        raise RuntimeError("dtype must be the same while lhType is %s, rhType is %s" %
                           (lh_tensor.dtype, rh_tensor.dtype))

    shape = lh_tensor.shape
    dtype = lh_tensor.dtype
    if op == "elewise_binary_add":
        lambda_func = lambda *indice: lh_tensor(*indice) + rh_tensor(*indice)
    elif op == "elewise_binary_sub":
        lambda_func = lambda *indice: lh_tensor(*indice) - rh_tensor(*indice)
    elif op == "elewise_binary_mul":
        lambda_func = lambda *indice: lh_tensor(*indice) * rh_tensor(*indice)
    elif op == "elewise_binary_min":
        lambda_func = lambda *indice: akg.tvm.min(lh_tensor(*indice), rh_tensor(*indice))
    elif op == "elewise_binary_max":
        lambda_func = lambda *indice: akg.tvm.max(lh_tensor(*indice), rh_tensor(*indice))
    elif op == "elewise_binary_or":
        lambda_func = lambda *indice: akg.tvm.select(akg.tvm.any(lh_tensor(*indice) > 0, rh_tensor(*indice) > 0),
                                                     lh_tensor(*indice), rh_tensor(*indice))
    elif op == "elewise_binary_and":
        lambda_func = lambda *indice: akg.tvm.select(akg.tvm.all(lh_tensor(*indice) > 0, rh_tensor(*indice) > 0),
                                                     lh_tensor(*indice), rh_tensor(*indice))
    elif op == "elewise_binary_scalar_axpy":
        lambda_func = lambda *indice: args[0].astype(dtype) * lh_tensor(*indice) + rh_tensor(*indice)
    else:
        raise RuntimeError("operation %s not support yet" % op)

    name = op.split("_")[-1] + "_" + str(name_index[0])
    name_index[0] += 1

    with akg.tvm.tag_scope(op):
        tmp = akg.tvm.compute(shape, lambda_func, name=name)

    return tmp


@vc_util.check_input_type(akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor)
def shape_binary_elewise_op_check(lhs, rhs):
    if len(lhs.shape) != len(rhs.shape):
        raise RuntimeError("The lhs shape ndim %d must be equal to the rhs %d" % (len(lhs.shape), len(rhs.shape)))

    for i in range(len(lhs.shape)):
        if not isinstance(lhs.shape, akg.tvm.expr.IntImm) or not isinstance(rhs.shape, akg.tvm.expr.IntImm):
            continue
        if lhs.shape[i].value != rhs.shape[i].value:
            raise RuntimeError("The lhs shape must be equal to the rhs")


@auto_cast_of_elewise
def vmla(x, y, z):
    """
    calculate x * y + z,  only support float16, float32.

    Args:
        x (tvm.tensor.Tensor): input.
        y (tvm.tensor.Tensor): input.
        z (tvm.tensor.Tensor): input.

    Returns:
        tvm.tensor.Tensor, X * Y + Z.
    """
    return multiple_elewise_op(x, y, z, op="elewise_multiple_mla")


@auto_cast_of_elewise
def vmadd(x, y, z):
    """
    calculate x * z + y,  only support  float16, float32.

    Args:
        x (tvm.tensor.Tensor): input.
        y (tvm.tensor.Tensor): input.
        z (tvm.tensor.Tensor): input.

    Returns:
        tvm.tensor.Tensor, X * Z + Y.
    """
    return multiple_elewise_op(x, y, z, op="elewise_multiple_madd")


@auto_cast_of_elewise
def vmaddrelu(x, y, z):
    """
    calculate relu(x * z + y),  only support  float16, float32.

    Args:
        x (tvm.tensor.Tensor): input.
        y (tvm.tensor.Tensor): input.
        z (tvm.tensor.Tensor): input.

    Returns:
        tvm.tensor.Tensor, relu(X * Z + Y).
    """
    return multiple_elewise_op(x, y, z, op="elewise_multiple_maddrelu")


@vc_util.check_input_type(akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor, str)
def multiple_elewise_op(x, y, z, op):
    """factory method of binary multiple operations."""
    shape_multi_elewise_op_check(x, y, z)
    if x.dtype != y.dtype or x.dtype != z.dtype or z.dtype != y.dtype:
        raise RuntimeError("dtype must be the same to each other")

    shape = x.shape
    dtype = x.dtype
    if op == "elewise_multiple_mla":
        lambda_func = lambda *indice: x(*indice) * y(*indice) + z(*indice)
    elif op == "elewise_multiple_madd":
        lambda_func = lambda *indice: x(*indice) * y(*indice) + z(*indice)
    elif op == "elewise_multiple_maddrelu":
        lambda_func = lambda *indice: akg.tvm.select((x(*indice) * y(*indice) + z(*indice)) >= 0,
                                                     x(*indice) * y(*indice) + z(*indice),
                                                     akg.tvm.const(0, dtype=dtype))
    else:
        raise RuntimeError("operation %s not support yet" % op)

    name = op.split("_")[-1] + "_" + str(name_index[0])
    name_index[0] += 1

    with akg.tvm.tag_scope(op):
        tmp = akg.tvm.compute(shape, lambda_func, name=name)

    return tmp


@vc_util.check_input_type(akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor)
def shape_multi_elewise_op_check(x, y, z):
    if len(x.shape) != len(y.shape) or len(x.shape) != len(z.shape) or len(z.shape) != len(y.shape):
        raise RuntimeError("The input shape ndim must be equal to the each other")

    for i in range(len(x.shape)):
        if x.shape[i].value != y.shape[i].value or x.shape[i].value != z.shape[i].value\
                or y.shape[i].value != z.shape[i].value:
            raise RuntimeError("The input shape must be equal to the each other")
