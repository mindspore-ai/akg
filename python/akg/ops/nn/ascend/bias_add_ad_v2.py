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

"""operator dsl function: bias_add_ad_v2"""

import akg.tvm
import akg
import akg.utils as utils
from .bias_add import bias_add


def bias_add_ad_v2(head, input_shape, data_format, target=utils.CCE):
    """Compute gradient for bias_add operator using automatic differentiate."""
    check_list = ["NHWC", "NC1HWC0", "DefaultFormat"]
    if data_format not in check_list:
        raise RuntimeError("bias_add_grad only support %s while dataformat is %s" %
                           (",".join(check_list), data_format))
    head_plh = akg.tvm.placeholder(head.shape, head.dtype, "head_plh")
    if data_format == "NC1HWC0":
        bias_shape = (1, head.shape[1], 1, 1, head.shape[4])
        bias_plh = akg.tvm.placeholder(bias_shape, head.dtype, "bias_plh")
    elif data_format == "NHWC":
        bias_shape = (input_shape[-1],)
        bias_plh = akg.tvm.placeholder(bias_shape, head.dtype, "bias_plh")
    else:
        bias_shape = (input_shape[1],)
        bias_plh = akg.tvm.placeholder(bias_shape, head.dtype, "bias_plh")
    bias_add_res = bias_add(head_plh, bias_plh, data_format)

    shape1 = [x.value for x in head_plh.shape]
    shape2 = [x.value for x in bias_plh.shape]

    def custom_bias_add_diff(out, input_data, head, ad_attrs, new_pld_array):
        if len(shape2) != 1:
            raise RuntimeError("Default Format needs Bias is a 1D Tensor!")
        if data_format == "NHWC":
            return [akg.tvm.compute(shape2, lambda l: head[0, 0, 0, l])]
        if data_format == "DefaultFormat":
            if len(shape1) == 2:
                return [akg.tvm.compute(shape2, lambda l: head[0, l])]
            if len(shape1) == 4:
                return [akg.tvm.compute(shape2, lambda l: head[0, l, 0, 0])]
            raise RuntimeError("bias_add only support 2D and 4D shape while dataformat is DefaultFormat")
        return None

    if data_format == "NC1HWC0":
        jacs = list(akg.differentiate(bias_add_res, [bias_plh], head))
    else:
        variables = akg.get_variables("reshape_diff")
        jacs = list(akg.differentiate(bias_add_res, [bias_plh], head, None, None,
                                      override={variables[0]: (variables[1], custom_bias_add_diff)}))

    return jacs[0]
