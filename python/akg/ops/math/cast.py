# Copyright 2020-2022 Huawei Technologies Co., Ltd
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

"""operator dsl function: cast"""
import akg.tvm
import akg.topi
import akg.utils as utils
from akg.utils.kernel_exec import product_is_mini
from akg.utils.format_transform import get_shape


@utils.check_input_type(akg.tvm.tensor.Tensor, str)
def _cast(data, dst_type):
    if data.dtype == "int64" and dst_type == "float16":
        data = akg.topi.cast(data, 'float32')
    if data.dtype == "float16" and dst_type == "int32":
        data = topi.trunc(data)
    if data.dtype == "float32" and dst_type == "uint8":
        data = akg.topi.cast(data, 'int32')
    utils.check_shape(data.shape)
    out = akg.topi.cast(data, dst_type)

    return out


@utils.check_input_type(akg.tvm.tensor.Tensor, str)
def _cast_ascend(data, dst_type):
    ori_type = data.dtype
    shape = get_shape(data)
    # dtype check
    dst_check_list = ["int8", "float32", "float16", "uint8", "int32"]

    if dst_type not in dst_check_list:
        raise RuntimeError("cast only support cast to %s while dtype is %s" %
                           (",".join(dst_check_list), dst_type))

    if product_is_mini():
        # product mini has not conv insn between float32 and int32.
        if ori_type == "float32" and dst_type == "int32":
            tmp = akg.topi.cast(data, "float16")
            return akg.topi.cast(tmp, dst_type)
        if ori_type == "int32" and dst_type == "float32":
            tmp = akg.topi.cast(data, "float16")
            return akg.topi.cast(tmp, dst_type)

    dtype_pair = (ori_type, dst_type)
    support_dtype = (('float32', 'float16'), ('float16', 'float32'), ('float16', 'int8'), ('float16', 'uint8'),
                     ('int32', 'float16'), ('int32', 'float32'), ('float16', 'int32'), ('float32', 'int32'),
                     ('uint8', 'float16'), ('int8', 'float16'))
    tmp_trans_dtype = (('int8', 'float32'), ('float32', 'int8'), ('bool', 'float32'), ('uint8', 'float32'),
                       ('uint8', 'int32'), ('bool', 'int32'), ('float32', 'uint8'))
    if dtype_pair not in support_dtype and dtype_pair not in tmp_trans_dtype and ori_type != dst_type:
        raise RuntimeError("Don't support cast from ", ori_type, " to ", dst_type)
    need_tmp_transfer = dtype_pair in tmp_trans_dtype

    if need_tmp_transfer:
        if data.dtype == 'float32' and dst_type == 'int8' and not product_is_mini():
            tmp = akg.tvm.compute(shape, lambda *indice: akg.tvm.trunc(data(*indice)).astype('int32'))
            tmp = akg.topi.cast(tmp, 'float16')
            out = akg.tvm.compute(shape, lambda *indice: akg.tvm.trunc(tmp(*indice)).astype(dst_type))
        else:
            tmp = akg.topi.cast(data, 'float16')
            out = akg.topi.cast(tmp, dst_type)
    else:
        if data.dtype in ('float16', 'float32') and dst_type in ('int8, int32') and not product_is_mini():
            out = akg.tvm.compute(shape, lambda *indice: akg.tvm.trunc(data(*indice)).astype(dst_type))
        else:
            out = akg.topi.cast(data, dst_type)

    return out


def cast(data, dst_type, target=utils.CCE):
    """
    cast data to target type.

    Args:
        data (tvm.tensor.Tensor): Tensor to be casted.
        dst_type (str): target cast type.

    Returns:
        tvm.tensor.Tensor, type is dst_type.
    
    Supported Platforms:
        'Ascend', 'GPU', 'CPU'
    """
    utils.check_supported_target(target)
    if target == utils.CCE:
        return _cast_ascend(data, dst_type)
    else:
        return _cast(data, dst_type)