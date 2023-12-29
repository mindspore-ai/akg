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

"""dsl create helping function"""
import collections
import itertools
import logging
import math
import akg
from akg.utils import format_transform as ft_util
from akg.utils import validation_check as vc_util


class TensorUtils:
    """Class for creating tensor."""
    CREATE_SCH_ONLY = 'create_sch_only'

    @classmethod
    def get_tensor_attrs(cls, tensor):
        """get tensor attrs."""
        tensor_attrs = dict()
        if "attrs" in dir(tensor.op):
            tensor_attrs = dict(tensor.op.attrs.items())
        return tensor_attrs

    @classmethod
    def update_tensor_attrs(cls, tensor, attrs):
        """update tensor attrs."""
        tensor_attrs = cls.get_tensor_attrs(tensor)
        tensor_attrs.update(attrs)
        tensor = akg.tvm.compute(tensor.shape,
                                 lambda *indice: tensor[indice],
                                 name=tensor.op.name,
                                 tag=tensor.op.tag,
                                 attrs=tensor_attrs)
        return tensor

    @classmethod
    def is_create_sch_only(cls, tensor):
        tensor_attrs = cls.get_tensor_attrs(tensor)
        if cls.CREATE_SCH_ONLY in tensor_attrs.keys():
            return True
        return False

    @classmethod
    def is_output_value(cls, tensor):
        """check output value."""
        return not cls.is_create_sch_only(tensor)

    @classmethod
    def inplace_set(cls, input_tensor, output_tensor, buffer_name="data_buf"):
        """inplace set."""
        input_tensor_shape = ft_util.get_shape(input_tensor)
        output_tensor_shape = ft_util.get_shape(output_tensor)
        if not input_tensor_shape == output_tensor_shape:
            raise RuntimeError("Shape of the input_tensor and the output_tensor should be equal, "
                               "but got %s and %s" % (input_tensor_shape, output_tensor_shape))
        output_tensor = cls.update_tensor_attrs(output_tensor, {cls.CREATE_SCH_ONLY: 1})
        data_buf = akg.tvm.decl_buffer(input_tensor.shape, input_tensor.dtype, name=buffer_name)
        binds_info = {input_tensor: data_buf, output_tensor: data_buf}
        return output_tensor, binds_info

    @classmethod
    def inplace_set_tensors(cls, input_tensors, output_tensors, buffer_names=None):
        """
        inplace set for tensors

        Args:
            in_tensors (Union[list, tuple]): Origin input tensors.
            out_tensors (Union[list, tuple]): Origin output tensors.
            buffer_names (Union[list, tuple] or None): Buffer names used to bind.

        Return:
            inplace_tensors (list): Output tensors with the inplace info.
            binds_infos (dict): Dictionary that maps the input tensor and the output
                                tensor to buffer.
        """
        if not buffer_names:
            buffer_names = ["data_buf_%s" % i for i in range(len(input_tensors))]
        for arg in (input_tensors, output_tensors, buffer_names):
            if not isinstance(arg, (tuple, list)):
                raise RuntimeError("arg must be tuple or list!")
        if len(input_tensors) != len(output_tensors) or len(input_tensors) != len(buffer_names):
            raise RuntimeError("length of the input_tensors, output_tensors and buffer_names must be equal!")

        inplace_tensors = []
        binds_infos = dict()
        for input_tensor, output_tensor, buffer_name in zip(input_tensors, output_tensors, buffer_names):
            inplace_tensor, binds_info = cls.inplace_set(input_tensor, output_tensor, buffer_name)
            inplace_tensors.append(inplace_tensor)
            binds_infos.update(binds_info)
        return inplace_tensors, binds_infos


def produce_shapes(shape1, shape2):
    """two input shapes produce three output shape."""
    shape1 = list(shape1)
    shape2 = list(shape2)
    flag = 0
    if len(shape1) < len(shape2):
        shape1, shape2 = shape2, shape1
        flag = 1

    output_shape_len = len(shape1)
    dec = output_shape_len - len(shape2)
    for i in range(dec):
        shape2 = [1] + shape2

    out_shape = []
    for i in range(output_shape_len):
        if (shape1[i] != shape2[i]) and (shape1[i] != 1) and (shape2[i] != 1):
            raise RuntimeError("input shapes not match!")
        if isinstance(shape1[i], int) and isinstance(shape2[i], int) and shape1[i] > shape2[i]:
            out_shape.append(shape1[i])
        else:
            out_shape.append(shape2[i])

    if flag == 1:
        shape1, shape2 = shape2, shape1

    return shape1, shape2, out_shape


def get_reduce_out_shape(in_shape, axis=None, keepdims=False):
    """
    Computes ouput shape in reduction operators.

    Args:
        in_shape : input shape
        axis (Union[int, list, tuple]): The reduction axis. Default value is None, in this case,
            all dimensions will be reduced.
        keepdims (bool): If True, retains reduced dimensions with length 1, default value is False.

    Returns:
        output shape.
    """

    dims = len(in_shape)
    if axis is None:
        axis = list(range(dims))
    if not isinstance(axis, (int, list, tuple)):
        raise ValueError("axis must be of the following type: int, list, tuple.")
    if isinstance(axis, int):
        axis = [axis]
    axis = list(axis)
    for i, axis_val in enumerate(axis):
        if axis_val < 0:
            axis[i] = axis_val + dims
        if axis_val >= dims:
            raise ValueError("axis[{}] is {}, which exceeds max dimension {}".format(i, axis[i], dims))
    remaining_axis = []
    for i in range(dims):
        if i not in axis:
            remaining_axis.append(i)
    out_shape = []
    for i in range(dims):
        if i in remaining_axis:
            out_shape.append(in_shape[i])
        else:
            if keepdims:
                out_shape.append(1)
    if not out_shape:
        out_shape.append(1)
    return out_shape


def get_input_pad_shape(shape, dtype):
    """Function for getting input pad shape."""
    pad_unit = ft_util.get_bytes(dtype, allow_none=True)

    if pad_unit is None:
        logging.warning("%s is not support in TensorAddPad, the result is not undefined.", dtype)
        return shape

    lastdim = int(math.ceil(shape[-1] / pad_unit) * pad_unit)
    pad_shape = [*shape[:-1], '{},{}'.format(shape[-1], lastdim)] if lastdim != shape[-1] else shape

    return pad_shape


def mul_axis_sum(data, axes, keepdims, name=None, attrs=None):
    """calculate sum one by one."""
    if name is None and attrs is None:
        for axis in axes:
            data = akg.topi.sum(data, axis=axis, keepdims=keepdims)
    else:
        shape = [x.value for x in data.shape]
        for axis in axes[:-1]:
            data = akg.topi.sum(data, axis=axis, keepdims=keepdims)
        l_axis = shape[axes[-1]]
        k = akg.tvm.reduce_axis((0, l_axis), name="k")
        res_shape = [1 if i in axes else shape[i] for i in range(len(shape))]

        def sumfunc(*i):
            new_i = list(i)
            new_i[axes[-1]] = k
            return akg.tvm.sum(data(*tuple(new_i)), axis=k)

        if name is None:
            data = akg.tvm.compute(res_shape, sumfunc, attrs=attrs)
        elif attrs is None:
            data = akg.tvm.compute(res_shape, sumfunc, name=name)
        else:
            data = akg.tvm.compute(res_shape, sumfunc, name=name, attrs=attrs)
    return data


def update_by_moving_average(hat_z, z, momentum):
    r"""
    Update value with moving average.

    Note:
        :math:`\hat{z_{new}} = momentum * \hat{z} + (1-momentum) * z`
        where \f$ \hat{z} \f$ is the estimated statistic and \f$ z \f$ is the new observed value.

    Args:
        hat_z (tvm.tensor.Tensor): Tensor of type float16, float32.
        z (tvm.tensor.Tensor): Tensor of type float16, float32.
        momentum (float): must meet '0.0 < momentum < 1.0'.

    Returns:
        tvm.tensor.Tensor, updated value.
    """
    run = akg.lang.ascend.vmuls(hat_z, momentum)
    now = akg.lang.ascend.vmuls(z, (1 - momentum))
    return akg.lang.ascend.vadd(run, now)


def _cal_pad_strategy_valid(out_shape, pad_sizes, pool_shapes, kernel, stride):
    """For strategy 'VALID'."""
    for i in range(2):
        out_shape.append(math.ceil((pool_shapes[i] - (kernel[i] - 1)) / stride[i]))
        if out_shape[i] <= 0:
            raise ValueError("With pad mode 'VALID', the value of the kernel (or window) size should be less than or "
                             "equal to that of the corresponding input shape!")
    pad_sizes += [0, 0]  # for h
    pad_sizes += [0, 0]  # for w


def _cal_pad_strategy_same(out_shape, pad_sizes, pool_shapes, kernel, stride):
    """For strategy 'SAME'."""
    for i in range(2):
        out_shape.append(math.ceil(pool_shapes[i] / stride[i]))
        diff_shape = ((out_shape[i] - 1) * stride[i] + kernel[i]) - pool_shapes[i]
        diff_shape = diff_shape if diff_shape > 0 else 0
        pad_shape = [math.floor(diff_shape / 2), math.ceil(diff_shape / 2)]
        pad_sizes += pad_shape


def _cal_pad_strategy_constants(strategy, out_shape, pad_sizes, pool_shapes, kernel, stride, contrain_var):
    """For strategy 'CONSTANTS'."""
    if len(strategy) != 4:
        raise RuntimeError("When with strategy 'CONSTANTS', strategy should be list or tuple of 4 int numbers, but "
                           "get {}".format(strategy))
    vc_util.check_pad('pad', strategy, 4)
    for i in range(2):
        pad_shape = [strategy[i * 2], strategy[i * 2 + 1]]
        if contrain_var:
            out_shape.append(akg.tvm.floordiv((pool_shapes[i] +
                                               (pad_shape[0] + pad_shape[1]) - kernel[i]), (stride[i])) + 1)
        else:
            out_shape.append(math.floor((pool_shapes[i] +
                                         (pad_shape[0] + pad_shape[1]) - kernel[i]) / float(stride[i])) + 1)

        pad_sizes += pad_shape
    height, width = out_shape
    if (isinstance(height, int) and height <= 0) or (isinstance(width, int) and width <= 0):
        raise ValueError("The height and width of calculated output shape [{}, {}] are invalid. Please check the "
                         "input parameters!".format(height, width))


def cal_pad_shapes_by_strategy(shape, kernel, stride, strategy):
    """
    Calculate the pad size and output shape by padding strategy.

    Args:
        shape (Union[list, tuple]): Input shape, a list or tuple of 5 int numbers.
        kernel (Union[list, tuple]): List or tuple of two int numbers for pooling window's size.
        stride (Union[list, tuple]): List or tuple of two int numbers for window's stride.
        strategy (Union[str, list]): A string or list for padding strategy, should be 'VALID',
             'SAME' or instance of list(including four int numbers, as 'CONSTANTS' strategy).

    Returns:
        pad_sizes: Padding sizes(a list of four int numbers: [H_head_pad, H_tail_pad, W_head_pad, W_tail_pad]).
        out_shape: Output tensor's shape(a list of two int numbers: [output_H, output_W]).
    """

    pool_shapes = [shape[2], shape[3]]
    out_shape = []
    pad_sizes = []
    contrain_var = False
    for sh in [shape, kernel, stride]:
        for s in sh:
            if not isinstance(s, (int, akg.tvm.expr.IntImm)):
                contrain_var = True
    if isinstance(strategy, str) and strategy.upper() == "VALID":
        _cal_pad_strategy_valid(out_shape, pad_sizes, pool_shapes, kernel, stride)
    elif isinstance(strategy, str) and strategy.upper() == "SAME":
        _cal_pad_strategy_same(out_shape, pad_sizes, pool_shapes, kernel, stride)
    elif isinstance(strategy, (list, tuple)):
        _cal_pad_strategy_constants(strategy, out_shape, pad_sizes, pool_shapes, kernel, stride, contrain_var)
    else:
        raise RuntimeError("Padding strategies only support 'VALID', 'CONSTANTS' or 'SAME', but get {}"
                           .format(strategy))

    return pad_sizes, out_shape


def broadcast_gradient_args(x, y):
    """
    Return the reduction indices for computing gradients of x op y with broadcast.

    Args:
        x (Union[list, tuple]): the shape of data input
        y (Union[list, tuple]): the shape of data input

    Returns:
        rx (list): the reduction indices for computing gradients of x
        ry (list): the reduction indices for computing gradients of y
    """
    rx = []
    ry = []
    for i, item in enumerate(x):
        if item < y[i]:
            rx.append(i)
        elif item > y[i]:
            ry.append(i)

    return rx, ry


def get_broadcast_shape(shape1, shape2):
    shape_out = collections.deque()
    reversed_shapes = map(reversed, (shape1, shape2))
    for items in itertools.zip_longest(*reversed_shapes, fillvalue=1):
        max_size = int(items[0] if items[1] == 1 else items[1])
        if any(int(item) not in (1, max_size) for item in items):
            raise ValueError(f'operands could not be broadcast together with {shape1}, {shape2}')
        shape_out.appendleft(max_size)
    return list(shape_out)


def zero_const(dtype):
    return akg.tvm.const(0, dtype)


def one_const(dtype):
    return akg.tvm.const(1, dtype)


def neg_one_const(dtype):
    return akg.tvm.const(-1, dtype)


def half_const(dtype):
    return akg.tvm.const(0.5, dtype)


def pi_const(dtype):
    return akg.tvm.const(3.1415926535897932384626433832795, dtype)


def get_value(val, type):
    if isinstance(val, type) and type in [akg.tvm.expr.IntImm, akg.tvm.expr.FloatImm]:
        return val.value
    return val
