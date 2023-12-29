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

"""
sqrt run define
"""

import numpy as np
import akg
import akg.lang.ascend
import akg.tvm
from akg.utils import kernel_exec as utils
from tests.common.test_op.ascend import confusion_matrix

from tests.common.tensorio import compare_tensor


def confusion_matrix_run(actual_shape, actual_dtype, predict_shape, predict_dtype, num_class,
                         kernel_name="confusion_matrix", attrs=None):
    # Create op
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(confusion_matrix.confusion_matrix, [actual_shape, predict_shape],
                                  [actual_dtype, predict_dtype], op_attrs=[num_class], kernel_name=kernel_name,
                                  attrs=attrs, tuning=t)
        if t:
            actual_data, expect_data, output_data, predict_data = gen_data(actual_dtype, actual_shape, num_class,
                                                                           predict_dtype, predict_shape)
            return mod, expect_data, (actual_data, predict_data, output_data)
        else:
            return mod
    else:
        mod = utils.op_build_test(confusion_matrix.confusion_matrix, [actual_shape, predict_shape],
                                  [actual_dtype, predict_dtype], op_attrs=[num_class], kernel_name=kernel_name,
                                  attrs=attrs)
        actual_data, expect_data, output_data, predict_data = gen_data(actual_dtype, actual_shape, num_class,
                                                                       predict_dtype, predict_shape)
        output_data = utils.mod_launch(mod, (actual_data, predict_data, output_data), expect=expect_data)

        return (actual_data, predict_data), output_data, expect_data, compare_tensor(expect_data, output_data)


def gen_data(actual_dtype, actual_shape, num_class, predict_dtype, predict_shape):
    # Generate data for testing the op
    np_types = {"int32": np.int32}
    actual_data = np.random.choice(a=num_class, size=actual_shape).astype(np_types[actual_dtype])
    predict_data = np.random.choice(a=num_class, size=predict_shape).astype(np_types[predict_dtype])
    expect_data = np.full([num_class, num_class], 0, dtype=np_types[actual_dtype])
    for i in range(actual_shape[0]):
        expect_data[actual_data[i]][predict_data[i]] += 1
    out_shape = expect_data.shape
    output_data = np.full(out_shape, 0, np.int32)
    return actual_data, expect_data, output_data, predict_data
