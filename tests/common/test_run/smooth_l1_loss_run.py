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

from akg.utils import kernel_exec as utils
from tests.common.tensorio import compare_tensor
import numpy as np
from tests.common.test_op import smooth_l1_loss


def smooth_l1_loss_run(prediction_shape, prediction_dtype, target_shape, target_dtype, anchor_samples_shape,
                       anchor_samples_dtype, anchor_sample_correct=0, delta=1.0, kernel_name="smooth_l1_loss",
                       attrs=None):
    input_shape = [prediction_shape, target_shape, anchor_samples_shape]
    input_dtype = [prediction_dtype, target_dtype, anchor_samples_dtype]
    op_attrs = [anchor_sample_correct, delta]

    if not utils.product_is_mini():
        attrs['enable_multicore'] = True

    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(smooth_l1_loss.smooth_l1_loss, input_shape, input_dtype, op_attrs,
                                  kernel_name=kernel_name, attrs=attrs, tuning=t)
        if t:
            anchor_samples, expect, output, prediction, target = gen_data(anchor_sample_correct, anchor_samples_dtype,
                                                                          anchor_samples_shape, delta, prediction_dtype,
                                                                          prediction_shape, target_dtype, target_shape)
            return mod, expect, (prediction, target, anchor_samples, output)
        else:
            return mod
    else:
        anchor_samples, expect, output, prediction, target = gen_data(anchor_sample_correct, anchor_samples_dtype,
                                                                      anchor_samples_shape, delta, prediction_dtype,
                                                                      prediction_shape, target_dtype, target_shape)
        mod = utils.op_build_test(smooth_l1_loss.smooth_l1_loss, input_shape, input_dtype, op_attrs,
                                  kernel_name=kernel_name, attrs=attrs)
        output = utils.mod_launch(mod, (prediction, target, anchor_samples, output), expect=expect)
        return (prediction, target, anchor_samples), output, expect, compare_tensor(output, expect, rtol=0.01,
                                                                                    atol=0.0001)


def gen_data(anchor_sample_correct, anchor_samples_dtype, anchor_samples_shape, delta, prediction_dtype,
             prediction_shape, target_dtype, target_shape):
    prediction = np.random.random(prediction_shape).astype(prediction_dtype)
    target = np.random.random(target_shape).astype(target_dtype)
    anchor_samples = np.random.randint(0, anchor_sample_correct + 1, anchor_samples_shape).astype(
        anchor_samples_dtype)
    anchor_samples_expand = np.expand_dims(anchor_samples, axis=2)
    error = np.subtract(prediction, target)
    abs_error = np.abs(error)
    quadratic = np.minimum(abs_error, delta)
    linear = np.subtract(abs_error, quadratic)
    loss = np.add(np.multiply(0.5, np.multiply(quadratic, quadratic)), np.multiply(delta, linear))
    loss = np.where(anchor_samples_expand == anchor_sample_correct, 0, loss)
    expect = np.sum(loss, axis=2)
    output = np.full(expect.shape, np.nan, prediction_dtype)
    return anchor_samples, expect, output, prediction, target
