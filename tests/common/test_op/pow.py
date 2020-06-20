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

"""pow"""
import akg
import akg.tvm
from akg.utils import validation_check as vc_util
from akg.utils.dsl_create import produce_shapes
from akg.utils import kernel_exec as utils
import akg.topi
from akg.ops.math.cast import cast


def pow_value(data, scale):
    shape1 = [x.value for x in data.shape]
    shape2 = [x.value for x in scale.shape]

    check_list = ["float16", "float32", "int32", "int8", "uint8"]
    dtype = data.dtype
    if not (dtype.lower() in check_list):
        raise RuntimeError("tile_cce only support %s while dtype is %s" % (",".join(check_list), dtype))

    shape = [x.value for x in data.shape]
    vc_util.check_shape(shape)
    vc_util.auto_broadcast_check(shape1, shape2)
    compute_dtype = "float32"
    if utils.product_is_mini():
        compute_dtype = "float16" 
    data = cast(data, compute_dtype)
    scale = cast(scale, compute_dtype)

    C = akg.topi.power(data, scale)
    C = cast(C, dtype)
    return C
