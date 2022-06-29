# Copyright 2021-2022 Huawei Technologies Co., Ltd
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

"""operator dsl function:assign"""

import akg
import akg.utils as utils
from akg.utils.dsl_create import TensorUtils


@utils.check_input_type(akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor, (str, type(None)))
def assign(ref, val, target=utils.CUDA):
    """
    Assign val to ref.

    Args:
        ref: Tensor, which is mutable.
        val: Tensor, which will be assigned to ref.

    Returns:
        fake_output: Tensor, all zeros has the same shape as ref, needed by ME.
        ref_val: Tensor, ref assigned with val.
        attrs: Dictionary, indicates that ref and ref_val share the same buf.
    
    Supported Platforms:
        'Ascend', 'GPU', 'CPU'
    """
    utils.check_supported_target(target)
    dtype = val.dtype
    utils.ops_dtype_check(dtype, [utils.DtypeForDavinci.ALL_FLOAT, utils.DtypeForDavinci.INT8,
                                    utils.DtypeForDavinci.INT16, utils.DtypeForDavinci.INT32,
                                    utils.DtypeForDavinci.INT64, utils.DtypeForDavinci.UINT8,
                                    utils.DtypeForDavinci.UINT16, utils.DtypeForDavinci.UINT32,
                                    utils.DtypeForDavinci.UINT64])
    shape1 = [x.value for x in ref.shape]
    shape2 = [x.value for x in val.shape]
    if shape1 != shape2:
        raise RuntimeError("assign operations need input shape equal!")
    utils.check_shape(shape2)
    ref_val = akg.tvm.compute(shape2, lambda *indice: val(*indice), name="ref_val")
    ref_val, binds_info = TensorUtils.inplace_set(ref, ref_val)
    attrs = {utils.BINDS: binds_info}
    fake_output = akg.tvm.compute(ref.shape, lambda *indice: ref_val(*indice), name="fake_output")

    return fake_output, ref_val, attrs