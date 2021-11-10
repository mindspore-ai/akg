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

import numpy as np
from tests.common.tensorio import compare_tensor
from akg.utils import kernel_exec as utils
from tests.common.test_op.ascend import scatter_
from akg.utils import validation_check as vc_util
from tests.common.gen_random import random_gaussian


def scatter__run(index_shape, src_shape, dim, output_shape, value, index_dtype, dtype, attrs):
    '''
    :param index_shape:  shape of 'index'
    :param src_shape: shape of 'src', when src_shape == [1,], regard src as float.
    :param dim: the axis along which to index
    :return:
    '''
    # check shapes
    vc_util.check_shape(index_shape)
    if not (index_dtype.lower() in "int32"):
        raise RuntimeError("indices_dtype only support int32 while dtype is %s" % index_dtype)

    support_list = {"float16": np.float16, "float32": np.float32, "int32": np.int32}
    if not (dtype.lower() in support_list):
        raise RuntimeError("scatter_nd_cce only support %s while dtype is %s" % (",".join(support_list.keys()), dtype))

    if src_shape is None:
        mod = utils.op_build_test(scatter_.scatter_, [index_shape], [index_dtype, dtype], op_attrs=[src_shape, dim, output_shape, value], kernel_name='scatter_', attrs=attrs)
    else:
        mod = utils.op_build_test(scatter_.scatter_, [index_shape, src_shape], [index_dtype, dtype], op_attrs=[dim, output_shape, value], kernel_name='scatter_', attrs=attrs)

    # Generate index_input. As all values in a row along the specified dimension dim must be unique and the values of index must be between 0 and shape.size(dim)-1,
    # we do some reshape operations here.
    # Concatenate dimensions before and after 'dim', set 'dim' as the last dimension and reshape the tensor into shape[mul_dim, index_shape[dim]
    reshape_temp_before = index_shape[:dim]
    reshape_temp_after = index_shape[dim + 1:]
    reshape_temp = reshape_temp_before + reshape_temp_after
    mul_dim = 1
    for item in reshape_temp:
        mul_dim = mul_dim * item

    # For last dimension, generate non-repetitive random number for each row.
    index_input = []
    for i in range(mul_dim):
        index_input.append(np.random.permutation(np.arange(0, output_shape[dim]))[:index_shape[dim]].astype(np.int32))
    index_input_reshape = np.array(index_input).reshape(reshape_temp + [index_shape[dim]])

    # move axis to original index_shape
    index_input = np.moveaxis(index_input_reshape, -1, dim)

    if src_shape is None:
        data_input = value

    else:
        if support_list[dtype] == np.int32:
            data_input = np.random.randint(100, size=src_shape, dtype=np.int32)
        else:
            data_input = random_gaussian(src_shape, miu=1, sigma=0.1).astype(support_list[dtype])

    expect = np.full(output_shape, 0.0, support_list[dtype])

    for index, i in np.ndenumerate(index_input):
        if isinstance(data_input, float):
            expect[index[:dim] + (i,) + index[dim + 1:]] = data_input
        else:
            expect[index[:dim] + (i,) + index[dim + 1:]] = data_input[index]

    output = np.full(output_shape, 0.0, support_list[dtype])
    if src_shape is None:
        output = utils.mod_launch(mod, (index_input, output), expect=expect)
    else:
        output = utils.mod_launch(mod, (index_input, data_input, output), expect=expect)

    return (index_input, data_input), output, expect, compare_tensor(output, expect, rtol=1e-03, equal_nan=True)
