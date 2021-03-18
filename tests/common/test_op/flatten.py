# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the Apache License Version 2.0.You may not use this
# file except in compliance with the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# Apache License for more details at
# http://www.apache.org/licenses/LICENSE-2.0

"""operator dsl function: flatten"""
import akg.topi
import akg.tvm
from akg.utils import validation_check as vc_util
from akg.utils import kernel_exec as utils
from akg.utils.format_transform import get_shape

@vc_util.check_input_type(akg.tvm.tensor.Tensor)
def flatten(x):
    """
    reshape into (batch, c*h*w).

    Args:
        x (akg.tvm.tensor.Tensor): the first dimension is batch 

    Returns:
       akg.tvm.tensor.Tensor
    """
    # check shape
    vc_util.check_shape(x)
    shape = get_shape(x)

    # check input tensor data_type
    vc_util.ops_dtype_check(x.dtype, [vc_util.DtypeForDavinci.ALL_FLOAT, 
        vc_util.DtypeForDavinci.INT8, vc_util.DtypeForDavinci.INT16,
        vc_util.DtypeForDavinci.INT32, vc_util.DtypeForDavinci.INT64,
        vc_util.DtypeForDavinci.UINT8, vc_util.DtypeForDavinci.UINT16,
        vc_util.DtypeForDavinci.UINT32, vc_util.DtypeForDavinci.UINT64])

    size = 1
    for i in range(1, len(shape)):
        size = size * shape[i]

    new_shape = [shape[0], size]
    res = akg.topi.reshape(x, new_shape)
    return res
