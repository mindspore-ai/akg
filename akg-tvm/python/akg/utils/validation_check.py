#!/usr/bin/env python3
# coding: utf-8
# Copyright 2019-2022 Huawei Technologies Co., Ltd
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

"""validation check functions"""
from functools import wraps, reduce
from enum import Enum
import akg.tvm
import akg.topi
from akg.utils.format_transform import get_bytes, get_shape

MAX_DATA_SIZE = 2 ** 31


class DtypeForDavinci(Enum):
    """Davinci supported dtype."""
    ALL_TYPES = ["float16", "float32", "int32", "int8", "uint8"]
    ALL_FLOAT = ["float16", "float32"]
    ALL_INT = ["int8", "int32"]
    ALL_UINT = ["uint8"]
    FLOAT16 = ["float16"]
    FLOAT32 = ["float32"]
    INT8 = ["int8"]
    INT16 = ["int16"]
    INT32 = ["int32"]
    INT64 = ["int64"]
    UINT8 = ["uint8"]
    UINT16 = ["uint16"]
    UINT32 = ["uint32"]
    UINT64 = ["uint64"]
    BOOL = ["bool"]


supported_bits = {
    "8": 1, "16": 2, "32": 4, "64": 8, "bool": 1
}

CCE = "cce"
CUDA = "cuda"
LLVM = "llvm"
CPU = "cpu"
BINDS = "binds"


def check_supported_target(target):
    supported_target = [CCE, CUDA, LLVM]
    if target.split()[0] not in supported_target:
        raise RuntimeError("the target %s is not supported!" % get_backend(target))


def get_backend(target_):
    target = target_.split()[0]
    if target == CCE:
        return "Ascend"
    elif target == CUDA:
        return "GPU"
    elif target == LLVM:
        return "CPU"
    return "UNKNOWN"


def check_input_type_dict(input_dict, input_key, input_name):
    """
    check input parameter type for new type: dict.

    Note:
        rule1: key of input_dict should be in the input_key
        rule2: type of input_dict[shape] should be in (list, tuple), if have shape
        rule3: type of input_dict[dtype] should be in (str), if have dtype

    Args:
        input_dict (dict): input_dict
        input_key (list or tuple): all input key list, the key of input must in input_key
        input_name (str): input param name, only used for error print

    Returns:
        None
    """

    def _check_input_type(input_key, input_type):
        if not isinstance(input_dict[input_key], input_type):
            raise RuntimeError(
                "the input parameter %s[%s] must be %s, while type of input is %s" %
                (input_name, input_key, input_type, type(input_dict[input_key])))

    for key in input_dict.keys():
        if key not in input_key:
            raise RuntimeError(
                "the input parameter %s must have arrt <%s>" %
                (input_name, key))

        # check shape's type of input_dict, if have shape
        if key == "shape":
            _check_input_type(key, (list, tuple))

        # check dtype's type of input_dict, if have dtype
        if key == "dtype":
            _check_input_type(key, (str,))


def check_input_type_list_tuple(inputs, expect):
    """check inputs by a list or tuple of expected types."""
    if not isinstance(inputs, expect[1][0]):
        raise RuntimeError("the input parameter %s must be (list, tuple), while"
                           " type of input is %s" % (expect[0], type(inputs)))
    for inp in inputs:
        if not isinstance(inp, expect[1][1]):
            raise RuntimeError("The element in parameter %s must be %s, while "
                               "type of input is %s" % (
                                   expect[0], expect[1][1], type(inp)))


def check_input_type(*type_args):
    """check input parameter type."""

    def out_wrapper(func):
        """outer wrapper function."""
        formal_parameter = func.__code__.co_varnames
        formal_parameter_list = list(zip(formal_parameter, type_args))

        @wraps(func)
        def in_wrapper(*args, **kwargs):
            """inner wrapper function."""
            for i, arg_v in enumerate(args):
                # add for new input dict, if dict, will check shape and dtype
                if isinstance(arg_v, dict):
                    check_input_type_dict(arg_v, arg_v.keys(),
                                          formal_parameter_list[i][0])

                if isinstance(formal_parameter_list[i][1], tuple):
                    if isinstance(formal_parameter_list[i][1][0], tuple) \
                            and len(formal_parameter_list[i][1]) == 2:
                        check_input_type_list_tuple(arg_v, formal_parameter_list[i])
                        continue

                if not isinstance(arg_v, formal_parameter_list[i][1]):
                    raise RuntimeError("the %sth input parameter %s must be %s, "
                                       "while type of input is %s" % (str(i), formal_parameter_list[i][0],
                                                                      formal_parameter_list[i][1],
                                                                      type(arg_v)))
            for i in kwargs:
                for j in formal_parameter_list:
                    if i in j:
                        if not isinstance(kwargs[i], j[1]):
                            raise RuntimeError("the input parameter %s must be "
                                               "%s, while type of input is %s"
                                               "" % (i, j[1], type(kwargs[i])))
                        break
            return func(*args, **kwargs)

        return in_wrapper

    return out_wrapper


def shape_dtype_max_size_check(shape, dtype):
    """check validation of tensor's shape."""
    if shape:
        for x in shape:
            if not isinstance(x, int):
                return
        mul = get_bytes(dtype) * int(reduce(lambda x, y: int(x) * int(y), shape))
        if mul > MAX_DATA_SIZE:
            error_msg = "*".join([str(sh) for sh in shape])
            raise RuntimeError("Invalid shape, data is {} bytes ({}), which "
                               "exceed max data size {} bytes"
                               .format(mul, error_msg, MAX_DATA_SIZE))


def tensor_max_size_check(tensor):
    """check validation of tensor's shape."""
    if not isinstance(tensor, akg.tvm.tensor.Tensor):
        raise RuntimeError("tensor should be an akg.tvm.tensor.Tensor, but got "
                           "type {}".format(type(tensor)))
    shape = get_shape(tensor)
    dtype = tensor.dtype
    shape_dtype_max_size_check(shape, dtype)


def check_shape(tensor, length=None, tensor_name=""):
    """The common check rule for placeholder data."""
    shape = get_shape(tensor)
    if not shape:
        raise RuntimeError("The ndim of input tensor {} must more than 0, "
                           "actual input is {}".format(tensor_name, len(shape)))

    for shape_v in shape:
        if isinstance(shape_v, (akg.tvm.expr.Var, akg.tvm.expr.Mul, akg.tvm.expr.FloorDiv, akg.tvm.expr.IntImm)):
            continue
        if not isinstance(shape_v, int) or shape_v <= 0:
            raise RuntimeError("The type of tensor {} axis value must be "
                               "positive int and value more than 0,"
                               "actual input is ({}) {}".
                               format(tensor_name, type(shape_v), shape_v))

    if length and len(shape) != length:
        raise ValueError('The length of {} should be {}, while actual length is {}'.
                         format(tensor_name, length, len(shape)))


def ops_dtype_check(dtype, args):
    """check validation of op's dtype."""
    expected_dtype = list()

    def _get_expect_dtype(expected_dtype, arg):
        if isinstance(arg, str):
            expected_dtype.append(arg)
        elif isinstance(arg, DtypeForDavinci):
            expected_dtype += arg.value
        elif isinstance(arg, (list, tuple)):
            for t in arg:
                _get_expect_dtype(expected_dtype, t)
        else:
            raise TypeError("arg should be either a string, a DtypeForDavinci "
                            "or a list/tuple of string or DtypeForDavinci, "
                            "while current is {}".format(type(arg)))

    _get_expect_dtype(expected_dtype, args)

    if isinstance(dtype, (list, tuple)):
        checking_dtype = [d.lower() for d in dtype]
    elif isinstance(dtype, str):
        checking_dtype = [dtype.lower()]
    else:
        raise TypeError("dtype should be either a string or a tuple/list of string")
    error_msg = "Supported dtype: {}, while received dtype: {}"
    if not set(checking_dtype).issubset(set(expected_dtype)):
        raise RuntimeError(error_msg.format(expected_dtype, checking_dtype))


def reduce_axis_check(reduce_shape, reduce_axis):
    """check validation of reduce axis for certain reduce shape."""
    dim = len(reduce_shape)
    if dim == 1 and isinstance(reduce_shape[0], int) and int(reduce_shape[0]) == 1:
        raise RuntimeError("Error, reduce shape is 1. Scalar is not supported "
                           "for reduction, please input a vector.")
    if isinstance(reduce_axis, int):
        if reduce_axis not in range(-dim, dim):
            raise RuntimeError("Reduce axis should be in range [%d. %d)"
                               "" % (-dim, dim))
    elif isinstance(reduce_axis, (tuple, list)):
        if len(reduce_axis) > len(reduce_shape):
            raise RuntimeError("Reduce axis list exceed reduce shape length: "
                               "%d vs %d, error" % (len(reduce_axis), len(reduce_shape)))
        processed_axis = []
        for axis in reduce_axis:
            processed_axis.append(int(axis + dim) if axis < 0 else int(axis))
        if len(set(processed_axis)) < len(processed_axis):
            raise RuntimeError("Reduce axis list contains %d duplicated element, please check"
                               % (len(processed_axis) - len(set(processed_axis))))
        for axis in processed_axis:
            if axis >= dim:
                raise RuntimeError("Invalid reduce axis, axis should less than %d" % dim)
    elif reduce_axis is not None:
        raise RuntimeError("axis should be a list, tuple or int.")


def elemwise_shape_check(shape_a, shape_b):
    """check validation of tensor's shape for element-wise op."""
    check_shape(shape_a)
    check_shape(shape_b)
    if len(shape_a) != len(shape_b):
        raise RuntimeError("Element-wise operation needs same data length, "
                           "while current is %s vs %s" % (len(shape_a), len(shape_b)))
    for i, shp in enumerate(shape_a):
        if int(shp) != int(shape_b[i]):
            raise RuntimeError("Element-wise operation needs same data shape, "
                               "while current is %s vs %s" % (shp, shape_b[i]))


def elemwise_dtype_check(dtype_a, dtype_b, supported_type=None):
    """check validation of tensor's dtype for element-wise op."""
    if supported_type:
        ops_dtype_check(dtype_a, supported_type)
        ops_dtype_check(dtype_b, supported_type)
    if dtype_a.lower() != dtype_b.lower():
        raise RuntimeError("Element-wise operation needs same data type, while "
                           "current is %s vs %s" % (dtype_a.lower(), dtype_b.lower()))


def auto_broadcast_check(shape_a, shape_b):
    """automatic broadcast check."""
    shape_l = get_shape(shape_a)
    shape_r = get_shape(shape_b)

    if len(shape_l) <= len(shape_r):
        shape_short = shape_l
        shape_long = shape_r
    else:
        shape_short = shape_r
        shape_long = shape_l

    dim_diff = len(shape_long) - len(shape_short)
    for i in range(dim_diff):
        shape_short.insert(0, 1)
    for i, shp in enumerate(shape_short):
        if int(shp) != int(shape_long[i]) and 1 not in [int(shp), int(shape_long[i])]:
            raise RuntimeError("Invalid auto broadcast, dim %d should be 1 or equal, "
                               "while now is %d vs %d" % (i, shp, shape_long[i]))


def broadcast_check(ori_shape, dst_shape):
    """check valid broadcast from ori_shape to dst_shape."""
    shape_l = get_shape(ori_shape)
    shape_r = get_shape(dst_shape)

    if len(shape_l) <= len(shape_r):
        dim_diff = len(shape_r) - len(shape_l)
        shape_l = ([1] * dim_diff) + shape_l
    else:
        raise RuntimeError("Cannot broadcast from shape %s to %s" % (str(ori_shape), str(dst_shape)))

    for i, shp in enumerate(shape_l):
        if int(shp) != int(shape_r[i]) and int(shp) != 1:
            raise RuntimeError("Cannot broadcast from shape %s to %s" % (str(ori_shape), str(dst_shape)))


def gemm_format_check(lhs_input, rhs_input, lhs_trans=False, rhs_trans=False):
    """check gemm format (shape length and value)."""
    dim = len(lhs_input)
    if len(rhs_input) != dim:
        raise RuntimeError("Dimensions are different, lhs input is of %d dimension "
                           "while rhs input is of %d dimension, " % (dim, len(rhs_input)))

    b_pos = [0] if dim == 3 else [0, 1]
    lhs_k_pos = -2 if lhs_trans else -1
    rhs_k_pos = -1 if rhs_trans else -2

    def length_check(tensor):
        if len(tensor) < 2 or len(tensor) > 4:
            raise RuntimeError("Gemm only support 2d shape (height, weight) "
                               "or 3d shape (batch, height, weight) "
                               "or 4d shape (batch_o, batch_i, height, weight) "
                               " while shape length is %d!" % (len(tensor)))

    def value_same_check(loc, lhs, rhs):
        if lhs != rhs:
            raise RuntimeError("%s size is not compatible, lhs input: %d and rhs input: %d" % (loc, lhs, rhs))

    def value_check(loc):
        if loc == "B":
            if len(lhs_input) > 2:
                for pos in b_pos:
                    value = int(lhs_input[pos])
                    cmp_value = int(rhs_input[pos])
                    value_same_check(loc, value, cmp_value)
        if loc == "K":
            if isinstance(lhs_input[lhs_k_pos], akg.tvm.expr.Var) or isinstance(rhs_input[rhs_k_pos], akg.tvm.expr.Var):
                return
            value = int(lhs_input[lhs_k_pos])
            cmp_value = int(rhs_input[rhs_k_pos])
            value_same_check(loc, value, cmp_value)

    for data in [lhs_input, rhs_input]:
        length_check(data)
    for location in ["B", "K"]:
        value_check(location)


def convolution_format_check(x_shape, w_shape, pad, stride, dilation):
    """check convolution format."""

    def conv_shape_check(shape):
        if (not isinstance(shape, (tuple, list))) or (len(shape) != 4):
            raise RuntimeError("conv tensor shape should be 4d list or tuple")

        conv_dtype = "float16"
        size = get_bytes(conv_dtype)
        for i in shape:
            if (not isinstance(i, int)) or (i <= 0):
                raise RuntimeError("conv tensor shape should be 4d list or "
                                   "tuple of positive integer")
            size *= i

        if size > MAX_DATA_SIZE:
            raise RuntimeError("runtime can not support tensor more than 2G size")

    def conv_pad_check(pad):
        if (not isinstance(pad, (tuple, list))) or (len(pad) != 4):
            raise RuntimeError("conv pad should be 4d list or tuple")

        for i in pad:
            if (not isinstance(i, int)) or (i < 0):
                raise RuntimeError("conv pad should be 4d list or tuple of "
                                   "positive integer or zero")

    def conv_stride_check(stride):
        if (not isinstance(stride, (tuple, list))) or (len(stride) != 2):
            raise RuntimeError("conv stride should be 2d list or tuple")

        for i in stride:
            if (not isinstance(i, int)) or (i <= 0):
                raise RuntimeError("conv stride should be 2d list or tuple of positive integer")

    def conv_dilation_check(dilation):
        if (not isinstance(dilation, (tuple, list))) or (len(dilation) != 2):
            raise RuntimeError("conv dilation should be 2d list or tuple")

        for i in dilation:
            if (not isinstance(i, int)) or (i <= 0):
                raise RuntimeError("conv dilation should be 2d list or tuple of positive integer")

    conv_shape_check(x_shape)
    conv_shape_check(w_shape)
    conv_pad_check(pad)
    conv_stride_check(stride)
    conv_dilation_check(dilation)

    if x_shape[1] != w_shape[1]:
        raise RuntimeError("conv's feature_map and filter tensor should be the same channel")

    if x_shape[2] + pad[0] + pad[1] < w_shape[2]:
        raise RuntimeError("kernel_h should be <= h + pad_left + pad_right: %d"
                           "" % (x_shape[2] + pad[0] + pad[1]))

    if x_shape[3] + pad[2] + pad[3] < w_shape[3]:
        raise RuntimeError("kernel_w should be <= w + pad_top + pad_bottom: %d"
                           "" % (x_shape[3] + pad[2] + pad[3]))

    if (pad[0] >= w_shape[2]) or (pad[1] >= w_shape[2]) \
            or (pad[2] >= w_shape[3]) or (pad[3] >= w_shape[3]):
        raise RuntimeError("pad value cannot be more than the filter value")


def davinci_format_check(shape, tensor_format, dim=-1):
    """check validation of tensor's shape for certain format used in davinci chip."""
    all_format_shape = {"NCHW": 4,
                        "NHWC": 4,
                        "NC1HWC0": 5,
                        "DefaultFormat": [2, 4]}
    if dim not in [-1, 2, 4, 5]:
        raise RuntimeError("Only support 2d, 4d, 5d format check, please set "
                           "dim to the dim want to check "
                           "or use default value -1 to check both all the dim")
    if dim == -1:
        support_format_shape = all_format_shape
    else:
        support_format_shape = {}
        for k, v in all_format_shape.items():
            if isinstance(v, int) and v == dim:
                support_format_shape[k] = v
            if isinstance(v, list) and dim in v:
                support_format_shape[k] = v

    support_shape = {"NC1HWC0": (4, 16)}
    if not isinstance(tensor_format, str):
        raise RuntimeError("Invalid davinci format, should be a string, "
                           "but get %s" % (type(tensor_format)))
    if tensor_format not in support_format_shape.keys():
        raise RuntimeError("Invalid davinci format {}, davinci support {}"
                           .format(tensor_format, support_format_shape.keys()))
    if isinstance(support_format_shape[tensor_format], int):
        if len(shape) != support_format_shape[tensor_format]:
            raise RuntimeError("Invalid shape {} for davinci format {}, needs "
                               "{} dim shape, current length{}"
                               .format(shape, tensor_format,
                                       support_format_shape[tensor_format], len(shape)))
    if isinstance(support_format_shape[tensor_format], list):
        if len(shape) not in support_format_shape[tensor_format]:
            raise RuntimeError("Invalid shape {} for davinci format {}, needs {} dim shape"
                               .format(shape, tensor_format,
                                       support_format_shape[tensor_format]))
    if tensor_format in support_shape.keys():
        check_dim = support_shape[tensor_format][0]
        expect_shape = support_shape[tensor_format][1]
        if int(shape[check_dim]) != expect_shape:
            raise RuntimeError("Invalid shape {} for davinci format {}, dim {} "
                               "should be {}, while current is {}"
                               .format(shape, tensor_format, check_dim,
                                       expect_shape, shape[check_dim]))


def is_valid_reduce_axis(tensor, reduce_axis):
    """
    if the reduce axis correspond to shape[axis] is 1, we can not refine the shape,or the reduce axis will be wrong.

    Args:
        tensor (tvm.tensor.Tensor): input tensor.
        reduce_axis (Union[list, tuple, int]): axis want to reduce.

    Returns:
        True or False.
    """
    # if the reduce axis correspond to shape[axis] is 1, we can not refine the
    # shape, or the reduce axis will be wrong
    shape = get_shape(tensor)
    if hasattr(reduce_axis, 'index'):
        for id_ite in reduce_axis:
            if shape[id_ite] == 1:
                return False
    else:
        if shape[reduce_axis] == 1:
            return False

    return True


def axis_check(shape_len, axis):
    """Check the value of axis and return the sorted axis."""

    def _axis_value_type_check(value):
        if not isinstance(value, int):
            raise RuntimeError("type of axis value should be int")
        if value >= shape_len or value < -shape_len:
            raise RuntimeError(
                "input axis is out of range, axis value can be from %d to %d" %
                (-shape_len, shape_len - 1))
        if value < 0:
            value = shape_len + value
        return value

    if not hasattr(axis, 'index'):
        axis = _axis_value_type_check(axis)
        return axis
    for i, axs in enumerate(axis):
        axis[i] = _axis_value_type_check(axs)

    axis = sorted(set(axis))
    return axis


def check_value_on_integer(arg_name, arg_value, low=None, high=None):
    """Judging integer type."""
    type_match = isinstance(arg_value, int) and not isinstance(arg_value, bool)
    if not type_match:
        raise ValueError("%s should be an int , but got type %s"
                         "" % (arg_name, type(arg_value)))
    if low and arg_value < low:
        raise ValueError("%s should be greater than or equal to %f, but got %f"
                         "" % (arg_name, low, arg_value))
    if high and arg_value >= high:
        raise ValueError("%s should be less than %f, but got %f"
                         "" % (arg_name, high, arg_value))


def check_typename(arg_name, arg_type, valid_types):
    """Does it contain the _name_ attribute."""

    def get_typename(t):
        return t.__name__ if hasattr(t, '__name__') else str(t)

    if arg_type in valid_types:
        return arg_type
    type_names = [get_typename(t) for t in valid_types]
    if len(valid_types) == 1:
        raise ValueError('type of {} should be {}, but got {}'.format(
            arg_name, type_names[0], get_typename(arg_type)))
    raise ValueError('type of {} should be one of {}, but got {}'.format(
        arg_name, type_names, get_typename(arg_type)))


def check_equal(arg_name1, arg_name2, arg1, arg2):
    """Check equal."""
    if arg1 != arg2:
        raise ValueError('{} should be equal to {}'.format(arg_name1, arg_name2))


def check_greater(arg_name1, arg_name2, arg1, arg2):
    """Check greater."""
    if arg1 <= arg2:
        raise ValueError('{} should be greater than {}'.format(arg_name1, arg_name2))


def check_5d(arg_name, shape5d, shape4d):
    """Check 5D shape."""
    blocksize = 16
    if len(shape4d) != 4:
        raise ValueError('invalid 4D shape of {}'.format(arg_name))
    if len(shape5d) != 5:
        raise ValueError('invalid 5D shape of {}'.format(arg_name))
    d1, d2, d3, d4 = shape4d
    if [x.value for x in shape5d] != [d1, (d2 + blocksize - 1) // blocksize, d3, d4, blocksize]:
        raise ValueError('the 4D shape and 5D shape of {} do not match'.format(arg_name))


def check_shape_length_equal(tensor_name, tensor_shape, shape_length):
    """Shape length equal judgment."""
    if isinstance(shape_length, (tuple, list)):
        if not len(tensor_shape) in shape_length:
            raise ValueError("The shape length of {} should be one of {}, but get {}"
                             .format(tensor_name, shape_length, len(tensor_shape)))
    elif len(tensor_shape) != shape_length:
        raise ValueError("The shape length of {} should be {}, but get {}"
                         .format(tensor_name, shape_length, len(tensor_shape)))


def check_shape_length_greater(tensor_name, tensor_shape, shape_length):
    """Shape length greater judgment."""
    if len(tensor_shape) <= shape_length:
        raise ValueError("The shape length of {} should be greater than {}, but get {}"
                         .format(tensor_name, shape_length, len(tensor_shape)))


def judge_var(num):
    """judge var if a tvm.var, tvm.const or python data type."""
    var_dict = {
        "python_const": [int, float],
        "tvm_const": [
            akg.tvm.expr.IntImm, akg.tvm.expr.UIntImm, akg.tvm.expr.FloatImm],
        "tvm_var": [akg.tvm.expr.Var]}
    num_type = type(num)
    for i in var_dict:
        if num_type in var_dict[i]:
            return i
    raise RuntimeError("Input var dtype {} error".format(type(num)))


def check_pad(arg_name, pad, length=None):
    """Check pad."""
    if not pad:
        raise ValueError("{} should not be None".format(arg_name))
    if not isinstance(pad, (tuple, list)):
        raise ValueError("{} should be tuple or list".format(arg_name))
    for i in pad:
        if not isinstance(i, int):
            raise ValueError("Elements in {} should be int".format(arg_name))
        if i < 0:
            raise ValueError("Elements in {} should not be less than zero"
                             "".format(arg_name))
    if length:
        if length != len(pad):
            raise ValueError("The length of {} should be {}".format(
                arg_name, length))


def check_int_list(array, array_name):
    """check whether all the elements are integers."""
    for num in array:
        if not isinstance(num, int):
            raise RuntimeError("Type of value in %s should be int, but got type %s" % (array_name, type(num)))


def comp_conv_backprop_out_shape(fmap_shape, filter_shape, pad_, stride_, dilation_):
    """Computes output shape of  "conv backprop"."""
    convolution_format_check(fmap_shape, filter_shape, pad_, stride_, dilation_)

    block_size = 16
    in_n, in_c, in_h, in_w = fmap_shape
    cout, _, w_h, w_w = filter_shape

    in_c = (in_c + block_size - 1) // block_size * block_size
    cout = (cout + block_size - 1) // block_size * block_size

    pad_top, pad_bottom, pad_left, pad_right = pad_
    stride_h, stride_w = stride_

    dilation_h, dilation_w = dilation_
    if dilation_h != 1 or dilation_w != 1:
        raise ValueError("The value of elements in dilation_ must be 1.")

    out_n = in_n
    out_c = cout
    out_h = (in_h + pad_top + pad_bottom - w_h) // stride_h + 1
    out_w = (in_w + pad_left + pad_right - w_w) // stride_w + 1

    dy_shape = (out_n, out_c, out_h, out_w)
    dx_shape = (in_n, in_c, in_h, in_w)
    dw_shape = (cout, in_c, w_h, w_w)
    return dy_shape, dx_shape, dw_shape

def comp_output_params(is_special5d, shape, in_rank, axis):
    if is_special5d:
        axes = [3, 2, 0]
        mid_shape = [1, shape[1], 1, 1, shape[4]]
    else:
        axes = [i for i in range(in_rank - 1, -1, -1) if i != axis]
        mid_shape = [1] * in_rank
        mid_shape[axis] = shape[axis]

    out_params = {
        "is_special5d": bool(is_special5d),
        "axis": axis,
        "axes": tuple(axes),
        "mid_shape": mid_shape
    }
    return out_params

def check_inputs_in_rank(data, axis, in_rank, data_format):
    """check in_rank of  inputs availability for fused_batch_norm and get axis"""
    if in_rank <= 1:
        raise AssertionError("Do not support 1D data.")
    if data_format == "DefaultFormat":
        if not isinstance(axis, int):
            raise RuntimeError("axis should be instance of int but {}"
                               "".format(axis))
        if axis not in range(-in_rank, in_rank):
            raise AssertionError("axis must be in range [%d, %d)"
                                 "" % (-in_rank, in_rank))
        if axis < 0:
            axis = in_rank + axis
    elif data_format == "NHWC":
        if in_rank != 4:
            raise AssertionError("Data shape {} mismatch data_format \"NHWC\"."
                                 "".format(data.shape))
        axis = 3
    elif data_format == "NCHW":
        if in_rank != 4:
            raise AssertionError("Data shape {} mismatch data_format \"NCHW\"."
                                 "".format(data.shape))
        axis = 1
    else:
        axis = 1
    return axis

def is_all_1_but_axis_equal(shape1, shape2, axis):
    if not isinstance(axis, (list, tuple)):
        axis = (axis,)

    for i, _ in enumerate(shape2):
        if i not in axis:
            if isinstance(shape1[i], akg.tvm.expr.Var) or int(shape1[i]) != 1:
                return False
        else:
            if isinstance(shape1[i], akg.tvm.expr.Var):
                if shape1[i] != shape2[i]:
                    return False
            else:
                if int(shape1[i]) != int(shape2[i]):
                    return False
    return True

def check_input_shape_equal_5(data, shape, *input_agrs):
    if len(data.shape) != 5:
        raise AssertionError("data shape {} mismatch data_format "
                                "\"NC1HWC0\".".format(data.shape))
    for i, element in enumerate(input_agrs):
        if len(element.shape) != 5 \
                or not is_all_1_but_axis_equal(element.shape, shape, (1, 4)):
            raise AssertionError("the {} parameter mismatch NC1HWC0 data (while shape "
                                    "is {}, input shape is {})!!!".format(i, element.shape, data.shape))
