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

"""operator dsl function:split"""
import akg
import akg.topi
import akg.tvm

def Split(data, num_or_size_splits, split_axis=0, num=None, target="cce"):
    """
    Splits a tensor into sub tensors.

    Args:
        data: Tensor.
        num_or_size_splits: Integer or list. Used to split data.
        split_axis: Integer. The dimension along which to split.

    Returns:
        Tuple of tensor, counts of which is determined by num_or_size_splits.
    """
    dtype = data.dtype
    shape = [x.value for x in data.shape]
    if isinstance(num_or_size_splits, (list, tuple)):
        if len(num_or_size_splits) >= 128:
            raise ValueError("Output tensors of split should not be more than 127. CCE can not support now.")
        if sum(num_or_size_splits) != shape[split_axis]:
            raise ValueError("Sum of size_split must be equal to the value of split axis.")
        if len(num_or_size_splits) == 1:
            res = akg.tvm.compute(data.shape, lambda *indice: data(*indice).astype(dtype), name='res')
            return res
        size_splits = [num_or_size_splits[0]]
        for i in range(len(num_or_size_splits) - 2):
            size_splits.append(num_or_size_splits[i + 1] + size_splits[i])
        res_tmp = akg.topi.split(data, size_splits, split_axis)
    else:
        if num_or_size_splits >= 128:
            raise ValueError("Output tensors of split should not be more than 127. CCE can not support now.")
        res_tmp = akg.topi.split(data, num_or_size_splits, split_axis)

    # add zero for each output to avoid same op.name
    zero = akg.tvm.const(0, dtype=data.dtype)
    res = []
    for item in res_tmp:
        item = akg.lang.ascend.vadds(item, zero)
        if item.dtype != dtype:
            item = akg.topi.cast(item, dtype)
        res.append(item)

    return res
