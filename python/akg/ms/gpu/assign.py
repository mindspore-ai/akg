# Copyright 2020 Huawei Technologies Co., Ltd
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

"""operator dsl function:Assign"""

import akg
import akg.topi as topi
import akg.tvm
from akg.utils import validation_check as vc_util
from akg.utils import kernel_exec as utils

@akg.schedule(topi.cuda.injective_single_kernel.schedule_injective)
def Assign(ref, val):
    """
    Assign val to ref.

    Args:
        ref: Tensor, which is mutable.
        val: Tensor, which will be assigned to ref.

    Returns:
        fake_output: Tensor, all zeros has the same shape as ref, needed by ME.
        ref_val: Tensor, ref assigned with val.
        attrs: Dictionary, indicates that ref and ref_val share the same buf.
    """
    dtype = val.dtype
    vc_util.ops_dtype_check(dtype, [vc_util.DtypeForDavinci.ALL_FLOAT, vc_util.DtypeForDavinci.INT8,
                                    vc_util.DtypeForDavinci.INT16, vc_util.DtypeForDavinci.INT32,
                                    vc_util.DtypeForDavinci.INT64, vc_util.DtypeForDavinci.UINT8,
                                    vc_util.DtypeForDavinci.UINT16, vc_util.DtypeForDavinci.UINT32,
                                    vc_util.DtypeForDavinci.UINT64])
    shape1 = [x.value for x in ref.shape]
    shape2 = [x.value for x in val.shape]
    if shape1 != shape2:
        raise RuntimeError("assign operations need input shape equal!")
    vc_util.check_shape(shape2)
    ref_val = akg.tvm.compute(shape2, lambda *indice: val(*indice), name="ref_val")
    ref_val, binds_info = utils.TensorUtils.inplace_set(ref, ref_val)
    attrs = {utils.BINDS: binds_info}
    fake_output = akg.tvm.compute(ref.shape, lambda *indice: ref_val(*indice), name="fake_output")

    return fake_output, ref_val, attrs
