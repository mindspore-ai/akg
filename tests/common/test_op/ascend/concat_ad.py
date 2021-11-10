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

"""operator dsl function: concat_ad"""
import akg.tvm
import akg
from akg.ops.array.ascend import Concat


def concat_ad(data, axis, wrt_index=0, target="cce"):
    """
    autodiff of concat with one or more input data.

    Args:
        data (list[akg.tvm.tensor.Tensor]): input tensors.
        axis (int): concat axis
        wrt_index (int): derivative with respect to index (must be less than len(data)).

    Returns:
        concatenation result with the given data and axis.
    """

    output = Concat(data, axis, target=target)
    head = akg.tvm.placeholder(output.shape, output.dtype, name="head")
    jacs = list(akg.differentiate(output, [data[wrt_index]], head))
    return jacs[0], head
