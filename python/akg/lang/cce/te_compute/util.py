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

"""util"""
from decorator import decorator
import akg.tvm
from akg.utils import kernel_exec

# Save op's output dtype, when first call the template api,we will save the dtype.
# Before auto scheduling,get the dtype and convert the res tensor to this dtype,
# and set the dtype to None.
op_output_dtype = None

dtype_map = {
    "float32": "f32",
    "float16": "f16",
    "int8": "s8",
    "uint8": "u8",
    "int32": "s32",
}


def save_op_output_dtype(func, *args):
    """
    Save op's output dtype, when first call the template api,

    Note:
        we will save the dtype.
        Before auto scheduling,get the dtype and convert the res tensor
        to this dtype, and set the dtype to None.
    """
    global op_output_dtype
    if op_output_dtype is None:
        if func.__name__ == "broadcast":
            if isinstance(args[0], int):
                output_dtype = "int32"
            elif isinstance(args[0], float):
                output_dtype = "float16"
            else:
                output_dtype = args[0].dtype
        elif func.__name__ == "concat":
            output_dtype = args[0][0].dtype
        else:
            output_dtype = args[0].dtype

        op_output_dtype = output_dtype


def get_op_output_dtype():
    """get saved op's output dtype and set saved dtype to None."""
    global op_output_dtype
    res = op_output_dtype
    op_output_dtype = None
    return res


@decorator
def dtype_check_decorator(func, *args, **kwargs):
    """check type decorator"""
    intput_check_list = {
        "broadcast": ["int8", "uint8", "float16", "float32", "int32", "bool"],
        "concat": ["int8", "uint8", "float16", "float32", "int32"],
        "unsorted_segment_sum": ["float16", "float32", "int32", "uint8", "int8"],
        "unsorted_segment_mean": ["float16", "float32", "uint8", "int8"],
        "unsorted_segment_prod": ["float16", "float32", "int32", "uint8", "int8"],
        "unsorted_segment_min": ["float16", "float32", "int32", "uint8", "int8"],
        "unsorted_segment_max": ["float16", "float32", "int32", "uint8", "int8"],
    }
    if func.__name__ == "cast":
        input_dtype = args[0].dtype
        output_dtype = args[1]
        judge_dtype = (input_dtype + "2" + output_dtype) if (input_dtype != output_dtype) else ""
    elif func.__name__ == "broadcast":
        if isinstance(args[0], int):
            judge_dtype = "int32"
        elif isinstance(args[0], float):
            judge_dtype = "float16"
        else:
            judge_dtype = args[0].dtype
    elif func.__name__ == "concat":
        judge_dtype = args[0][0].dtype
    else:
        judge_dtype = args[0].dtype

    if isinstance(intput_check_list[func.__name__], list):
        check_bool(judge_dtype in intput_check_list[func.__name__],
                   "%s input_dtype just support %s, while input dtype is %s" % (
                       func.__name__, str(intput_check_list[func.__name__]), judge_dtype))
    else:
        intri_name = "Intrinsic_" + func.__name__
        s_dtypes = intput_check_list[func.__name__](intri_name)

        check_bool(judge_dtype in s_dtypes,
                   "%s input_dtype just support %s, while input dtype is %s" % (
                       func.__name__, str(s_dtypes), judge_dtype))

    return func(*args, **kwargs)


def get_value(key):
    """
    call global func to get product value.

    Args:
        key (str): key.
    """
    mode = kernel_exec.get_runtime_mode()
    if "cloud" in mode:
        product = "1.6"
    else:
        product = "1.1"

    if "Buffer" in key:
        f = akg.tvm.get_global_func("cce.product_conf_buffer")

        value = f(product, key)
        if value == 0:
            raise RuntimeError("Get the cce product value is 0")

        return value
    if "Compiler" in key:
        f = akg.tvm.get_global_func("cce.product_conf_compiler")

        value = f(product, key)
        if value == "":
            raise RuntimeError("Get the cce product value is None")

        return value
    if "Intrinsic" in key:
        f = akg.tvm.get_global_func("cce.product_conf_intrinsic")

        value = f(product, key)
        if value == "":
            raise RuntimeError("Get the cce product value is None")

        return value
    if "Core" in key:
        f = akg.tvm.get_global_func("cce.product_conf_core")

        value = f(product, key)
        if value == 0:
            raise RuntimeError("Get the cce product value is None")

        return value
    return None


def get_intr_types(intr):
    """get intrinsic types"""
    return str_to_tuple(get_value(intr))


def str_to_tuple(string_):
    """string to tuple"""
    if string_:
        return string_.split(",")
    return []


def is_cast_support(src_type, dst_type):
    """check cast support"""
    if src_type not in dtype_map:
        raise RuntimeError("%s is unsupported dtype!" % src_type)

    if dst_type not in dtype_map:
        raise RuntimeError("%s is unsupported dtype!" % dst_type)

    if src_type == dst_type:
        return True

    cast_type = dtype_map[src_type] + "2" + dtype_map[dst_type]

    if cast_type == "s322f16":
        cast_type = "deq"

    conv_list = get_intr_types("Intrinsic_vconv")
    if cast_type in conv_list:
        return True

    return False


def judge_var(num):
    """judge var if a akg.tvm.var, akg.tvm.const or python data type"""
    var_dict = {"python_const": [int, float],
                "tvm_const": [akg.tvm.expr.IntImm, akg.tvm.expr.UIntImm, akg.tvm.expr.FloatImm],
                "tvm_var": [akg.tvm.expr.Var]}
    num_type = type(num)
    for i in var_dict:
        if num_type in var_dict[i]:
            return i
    raise RuntimeError("Input var Error")


def shape_to_list(shape):
    """translate akg.tvm.shape to list type in python"""
    tmp = []
    for i in shape:
        if isinstance(i, akg.tvm.expr.Var):
            tmp.append(i)
        else:
            tmp.append(i.value)
    return tmp


def refine_axis(axis, shape):
    """refine axis"""
    if isinstance(axis, (tuple, list)):
        local_axis = axis
    else:
        local_axis = [axis]
    res_axis = []
    shape_len = len(shape)
    for i in local_axis:
        if i < 0:
            laxis = shape_len + i
        else:
            laxis = i
        if (laxis >= shape_len) or (laxis < 0):
            raise RuntimeError("wrong axis.")
        res_axis.append(laxis)
    return sorted(res_axis)


def check_bool(bool_res, append_str):
    """check boolean"""
    if not bool_res:
        raise RuntimeError(append_str)
