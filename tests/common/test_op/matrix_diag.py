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

"""operator dsl function: matrix_diag"""

import akg.tvm
from akg.tvm.hybrid import script
from akg.utils import validation_check as vc_util
from akg.utils.dsl_create import zero_const
from akg.utils.format_transform import get_shape


@vc_util.check_input_type(akg.tvm.tensor.Tensor, ((list, tuple), int))
def matrix_diag(data, out_shape):
    """
    Generate a batched tensor whose value in diagonal lines are defined in `data`.

    Args:
        data (tvm.tensor.Tensor): A tensor of type float16, float32 or int32. Rank is L.
        out_shape (Union[list, tuple]): Output shape of length L + 1.
            The value of `out_shape[0, ..., L-1]` should be equal to `data.shape[0, ..., L-1]`.

    Returns:
        tvm.tensor.Tensor, has same type as "data", shape is "out_shape".
    """
    dtype = data.dtype
    vc_util.ops_dtype_check(dtype, [vc_util.DtypeForDavinci.ALL_FLOAT,
                                    vc_util.DtypeForDavinci.INT32])

    shape = get_shape(data)
    vc_util.check_shape(data)
    vc_util.check_shape(out_shape, length=len(shape) + 1)
    if tuple(shape[:-1]) != tuple(out_shape[:-2]):
        raise RuntimeError("The value of out_shape[:-2] should be equal to data.shape[:-1]")

    res = akg.tvm.compute(out_shape,
                          lambda *i: akg.tvm.if_then_else(akg.tvm.all(i[-1] == i[-2], i[-1] < shape[-1]),
                                                          data(*i[:-1]),
                                                          zero_const(dtype)),
                          name="diag")

    return res
