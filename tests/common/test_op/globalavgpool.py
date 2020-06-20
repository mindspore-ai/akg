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

"""operator dsl function: globalavgpool"""
import sys
import akg.topi
import akg.tvm
import akg
from akg import backend as cce


def globalavgpool(n, c, h, w, pool_type, attrs, kernel_name="global_pool"):
    """
    Performs the global average pooling on the input. For each feature map we can define the formula as:
    \f[
     res = \frac{1}{W * H} \\sum X_{i,j}
    \f]
    Note:
        The real input is create by akg.tvm.placeholder
    Args:
        n (int): input batchsize.
        c (int): input channel.
        h (int): input height.
        w (int): input weight.
        pool_type (str): pooling mode, default average.
        attrs (str): Default None.
        kernel_name (str): a str about kernel_name

    Returns:
            tvm.tensor.Tensor of shape n * c * 1 * 1
    """

    input = akg.tvm.placeholder((n, c, h, w), name='input', dtype="float16")
    output = akg.topi.nn.global_pool(input, pool_type=pool_type)
    s = akg.tvm.create_schedule(output.op)
    with akg.build_config(add_lower_pass=cce.debug_mode(0), dump_pass_ir=True):
        mod = akg.build(s, [input, output], "cce", name=kernel_name, attrs=attrs, polyhedral=True)
    return mod
