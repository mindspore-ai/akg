#!/usr/bin/env python3
# coding: utf-8
# Copyright 2019-2021 Huawei Technologies Co., Ltd
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

"""operator dsl function: bias_add"""

import akg
from akg.ops.array.reshape import reshape
import akg.utils as utils
from akg.utils.format_transform import get_shape


@utils.check_input_type(akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor, str, (str, type(None)))
def bias_add(data1, data2, data_format, target=utils.CCE):
    """
    Adds bias data2 to input tensor data1.

    Args:
        data1 (tvm.tensor.Tensor): Tensor of type float16, float32.
        data2 (tvm.tensor.Tensor): The bias tensor, should be of same type as data1.
                                   If shape(data2) != shape(data1), broadcast will happen.
        data_format (str): Data format of input tensors, could be NC1HWC0, NHWC or DefaultFormat.

    Returns:
        tvm.tensor.Tensor of same shape and type as data1.

    Supported Platforms:
        'Ascend'
    """
    utils.check_shape(data1.shape)
    utils.check_shape(data2.shape)
    shape1 = get_shape(data1)
    shape2 = get_shape(data2)
    utils.davinci_format_check(shape1, data_format)
    utils.ops_dtype_check([data1.dtype, data2.dtype], utils.DtypeForDavinci.ALL_FLOAT)

    if data_format == 'NC1HWC0':
        data2_new = akg.lang.ascend.broadcast(data2, shape1)
        res = akg.lang.ascend.vadd(data1, data2_new)
    else:
        if len(shape2) != 1:
            raise RuntimeError("data2 should be a 1D Tensor!")

        if data_format == "NHWC":
            if len(shape1) != 4:
                raise RuntimeError("bias_add only support 4D shape when data format is NHWC!")
            c_dim_len = shape1[3]
            if c_dim_len != shape2[0]:
                raise ValueError("The size of bias should be equal to the channel dimension, "
                                 " while the size of bias is {0} and the channel dimension is "
                                 "{1}".format(shape2[0], c_dim_len))
            data2_reshaped = reshape(data2, [1, 1, 1, shape2[0]], target)
        elif data_format == "DefaultFormat":
            if len(shape1) != 2 and len(shape1) != 4:
                raise RuntimeError("bias_add only support 2D and 4D shape when data format is DefaultFormat!")
            c_dim_len = shape1[1]
            if c_dim_len != shape2[0]:
                raise ValueError("The size of bias should be equal to the channel dimension, "
                                 " while the size of bias is {0} and the channel dimension is "
                                 "{1}".format(shape2[0], c_dim_len))
            if len(shape1) == 2:
                data2_reshaped = reshape(data2, [1, shape2[0]], target)
            else:
                # NCHW
                data2_reshaped = reshape(data2, [1, shape2[0], 1, 1], target)

        data2_new = akg.lang.ascend.broadcast(data2_reshaped, shape1)
        res = akg.lang.ascend.vadd(data1, data2_new)

        akg.register_variables("reshape_diff", [data2], data2_reshaped)

    return res
