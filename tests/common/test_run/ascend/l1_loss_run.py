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
from tests.common.tensorio import compare_tensor
import numpy as np
from tests.common.test_op.ascend import l1_loss
import os


def l1_loss_run(prediction_shape, prediction_dtype, target_shape, target_dtype, reduction='none', kernel_name="l1_loss",
                attrs=None):
    input_shape = [prediction_shape, target_shape]
    input_dtype = [prediction_dtype, target_dtype]
    op_attrs = [reduction]

    if not utils.product_is_mini():
        attrs['enable_multicore'] = True
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(l1_loss.l1_loss, input_shape, input_dtype, op_attrs, kernel_name=kernel_name,
                                  attrs=attrs, tuning=t)
        if t:
            expect, output, prediction, target = gen_data(prediction_dtype, prediction_shape, reduction, target_dtype,
                                                          target_shape)
            return mod, expect, (prediction, target, output)
        else:
            return mod
    else:
        mod = utils.op_build_test(l1_loss.l1_loss, input_shape, input_dtype, op_attrs, kernel_name=kernel_name,
                                  attrs=attrs)
        expect, output, prediction, target = gen_data(prediction_dtype, prediction_shape, reduction, target_dtype,
                                                      target_shape)
        output = utils.mod_launch(mod, (prediction, target, output), expect=expect)
        # ret = (prediction,target),output,expect,compare_tensor(output,expect,rtol=0.001,atol=0.0001)
        # print("compare result is :", ret)
        # return ret
        return (prediction, target), output, expect, compare_tensor(output, expect, rtol=0.001, atol=0.0001)


def gen_data(prediction_dtype, prediction_shape, reduction, target_dtype, target_shape):
    support_list = {"float16": np.float16, "float32": np.float32}
    prediction = np.random.random(prediction_shape).astype(support_list[prediction_dtype])
    target = np.random.random(target_shape).astype(support_list[target_dtype])
    error = np.subtract(prediction, target)
    expect = np.abs(error)
    if reduction == 'sum':
        expect = np.sum(expect)
    if reduction == 'mean':
        expect = np.mean(expect)
    if reduction == 'sum' or reduction == 'mean':
        out_shape = []
        out_shape.append(1)
        output = np.full(out_shape, 0, prediction_dtype)
    else:
        output = np.full(expect.shape, np.nan, prediction_dtype)
    return expect, output, prediction, target
