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
import akg.utils as utils
import akg.utils as utils
from akg.utils.format_transform import get_shape

@utils.check_input_type(akg.tvm.tensor.Tensor)
def flatten(x):
    """
    reshape into (batch, c*h*w).

    Args:
        x (akg.tvm.tensor.Tensor): the first dimension is batch 

    Returns:
       akg.tvm.tensor.Tensor
    """
    # check shape
    utils.check_shape(x)
    shape = get_shape(x)

    # check input tensor data_type
    utils.ops_dtype_check(x.dtype, [utils.DtypeForDavinci.ALL_FLOAT, 
        utils.DtypeForDavinci.INT8, utils.DtypeForDavinci.INT16,
        utils.DtypeForDavinci.INT32, utils.DtypeForDavinci.INT64,
        utils.DtypeForDavinci.UINT8, utils.DtypeForDavinci.UINT16,
        utils.DtypeForDavinci.UINT32, utils.DtypeForDavinci.UINT64])

    size = 1
    for i in range(1, len(shape)):
        size = size * shape[i]

    new_shape = [shape[0], size]
    res = akg.topi.reshape(x, new_shape)
    return res
