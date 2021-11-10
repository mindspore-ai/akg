# Copyright 2020-2021 Huawei Technologies Co., Ltd
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

"""scatter_add_run"""
import numpy as np
from functools import reduce

from tests.common.tensorio import compare_tensor
from akg.utils import kernel_exec as utils
from tests.common.test_op.ascend import scatter_add
from tests.common.base import get_rtol_atol
from tests.common.gen_random import random_gaussian

def scatter_add_run(shape_ref, shape_indices, dtype_ref, dtype_indices, attrs=None):
    """scatter_add_run implementation"""
    args, exp_output, ref, indices, updates = gen_data(shape_ref, shape_indices, dtype_ref, dtype_indices)
    shape_updates = updates.shape
    mod = utils.op_build_test(scatter_add.scatter_add, [shape_ref, shape_indices, shape_updates],
                              [dtype_ref, dtype_indices, dtype_ref],
                              kernel_name='scatter_add', op_attrs=[], attrs=attrs)
    acu_output = utils.mod_launch(mod, args, outputs=(0, ), expect=exp_output)
    # compare result
    rtol, atol = get_rtol_atol("scatter_add", dtype_ref)
    testcase_result = compare_tensor(acu_output, exp_output, rtol=rtol, atol=atol, equal_nan=True)

    return [ref, indices, updates], acu_output, exp_output, testcase_result


def gen_data(shape_ref, shape_indices, dtype_ref, dtype_indices):
    ref = random_gaussian(shape_ref, miu=10, sigma=0.3).astype(dtype_ref)
    new_ref = ref.copy()

    # generate valid index
    indices = np.random.randint(low=0, high=shape_ref[0], size=shape_indices, dtype=dtype_indices)

    # reshape to a 1D tensor to index
    all_shape = np.prod(shape_indices).astype(dtype_indices)
    new_indices = np.reshape(indices, (all_shape, ))

    # according to indices shape and ref shape to make updates shape
    updates_shape = shape_indices + shape_ref[1:]
    updates = random_gaussian(updates_shape, miu=3, sigma=0.3).astype(dtype_ref)

    # according to new_indieces shape and ref shape to make new_update_shape, make sure to update base on new_indices
    new_updates_shape = new_indices.shape + shape_ref[1:]
    new_updates = np.reshape(updates, new_updates_shape)

    # get results by new_updates
    for i in range(new_indices.shape[0]):
        new_ref[new_indices[i], ] += new_updates[i, ]

    output = np.full(shape_ref, np.nan, dtype_ref)
    args = [ref, indices, updates, output]
    return args, new_ref, ref, indices, updates,
