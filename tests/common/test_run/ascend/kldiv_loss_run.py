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

from akg.utils import kernel_exec as utils
from akg.utils.kernel_exec import product_is_mini
from tests.common.tensorio import compare_tensor
import numpy as np
from tests.common.test_op.ascend import kldiv_loss
from tests.common.gen_random import random_gaussian
from akg.utils.dsl_create import get_reduce_out_shape



def kldiv_loss_run(shape, dtype, reduction='none', kernel_name="kldiv_loss", attrs=None):
    input_shape = [shape, shape]
    input_dtype = [dtype, dtype]
    op_attrs = [reduction]

    if not product_is_mini():
        attrs['enable_multicore'] = True

    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(kldiv_loss.kldiv_loss, input_shape, input_dtype, op_attrs, kernel_name=kernel_name, attrs=attrs, tuning=t)
        if t:
            expect, output, prediction, target = gen_data(dtype, reduction, shape)
            return mod, expect, (prediction, target, output)
        else:
            return mod
    else:
        mod = utils.op_build_test(kldiv_loss.kldiv_loss, input_shape, input_dtype, op_attrs, kernel_name=kernel_name, attrs=attrs)
        expect, output, prediction, target = gen_data(dtype, reduction, shape)
        output = utils.mod_launch(mod, (prediction, target, output), expect=expect)
        return (prediction, target), output, expect, compare_tensor(output, expect, rtol=0.005, atol=0.005)


def gen_data(dtype, reduction, shape):
    # support_list = {"float16": np.float16, "float32": np.float32}
    target = random_gaussian(shape, miu=0, sigma=1).astype(dtype)
    target = np.abs(target)
    prediction = random_gaussian(shape, miu=0, sigma=1).astype(dtype)
    prediction = np.abs(prediction)
    # off_set = np.full(prediction.shape, 0.05, dtype)
    off_set = np.full(prediction.shape, 2, dtype)
    prediction = np.add(prediction, off_set)
    target = np.add(target, off_set)
    pre_log = np.log(prediction).astype(dtype)
    tar_log = np.log(target).astype(dtype)
    sub_log = np.subtract(tar_log, pre_log)
    expect = np.multiply(target, sub_log).astype(dtype)
    if reduction == 'sum':
        expect = np.sum(expect)
    if reduction == 'mean':
        expect = np.mean(expect)
    if reduction == 'batchmean':
        reduce_axis = tuple(np.arange(1, len(shape)))
        expect = np.mean(expect, axis=reduce_axis, keepdims=False)
    if reduction == 'sum' or reduction == 'mean':
        out_shape = []
        out_shape.append(1)
        output = np.full(out_shape, 0, dtype)
    if reduction == 'batchmean':
        reduce_axis = tuple(np.arange(1, len(shape)))
        out_shape = get_reduce_out_shape(shape, axis=reduce_axis, keepdims=False)
        output = np.full(expect.shape, np.nan, dtype)
    if reduction == 'none':
        output = np.full(expect.shape, np.nan, dtype)
    return expect, output, prediction, target
