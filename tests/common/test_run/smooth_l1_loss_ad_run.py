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
from tensorio import compare_tensor
import numpy as np
from test_op import smooth_l1_loss_ad
import os


def smooth_l1_loss_ad_run(prediction_shape, prediction_dtype, target_shape, target_dtype, anchor_samples_shape,
                          anchor_samples_dtype, anchor_sample_correct=0, delta=1.0, kernel_name="smooth_l1_loss_ad",
                          attrs=None):
    input_shape = [prediction_shape[:-1], prediction_shape, target_shape, anchor_samples_shape]
    input_dtype = [prediction_dtype, prediction_dtype, target_dtype, anchor_samples_dtype]
    op_attrs = [anchor_sample_correct, delta]

    if not utils.product_is_mini():
        attrs['enable_multicore'] = True

    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(smooth_l1_loss_ad.smooth_l1_loss_ad, input_shape, input_dtype, op_attrs,
                                  kernel_name=kernel_name, attrs=attrs, tuning=t)
        if t:
            anchor_samples, expect, head, output, prediction, target = gen_data(anchor_sample_correct,
                                                                                anchor_samples_dtype,
                                                                                anchor_samples_shape, delta,
                                                                                prediction_dtype,
                                                                                prediction_shape, target_dtype,
                                                                                target_shape)
            return mod, expect, (head, prediction, target, anchor_samples, output)
        else:
            return mod
    else:
        anchor_samples, expect, head, output, prediction, target = gen_data(anchor_sample_correct, anchor_samples_dtype,
                                                                            anchor_samples_shape, delta,
                                                                            prediction_dtype,
                                                                            prediction_shape, target_dtype,
                                                                            target_shape)
        mod = utils.op_build_test(smooth_l1_loss_ad.smooth_l1_loss_ad, input_shape, input_dtype, op_attrs,
                                  kernel_name=kernel_name, attrs=attrs)
        # source_code = mod.imported_modules[0].get_source()
        # print ("==========Generated cce code:==========")
        # print (source_code)
        output = utils.mod_launch(mod, (head, prediction, target, anchor_samples, output), expect=expect)

        return (head, prediction, target, anchor_samples), output, expect, compare_tensor(output, expect, rtol=0.01,
                                                                                          atol=0.001)


def gen_data(anchor_sample_correct, anchor_samples_dtype, anchor_samples_shape, delta, prediction_dtype,
             prediction_shape, target_dtype, target_shape):
    support_list = {"float16": np.float16, "float32": np.float32}
    prediction = np.random.random(prediction_shape).astype(support_list[prediction_dtype])
    target = np.random.random(target_shape).astype(support_list[target_dtype])
    head = np.random.random(prediction_shape[:-1]).astype(support_list[prediction_dtype])
    support_list_int = {"int16": np.int16, "int32": np.int32}
    # generate the random binary numbers
    anchor_samples = np.random.randint(0, anchor_sample_correct + 2, anchor_samples_shape).astype(
        support_list_int[anchor_samples_dtype])
    anchor_samples_expand = np.expand_dims(anchor_samples, axis=2)
    error = np.subtract(prediction, target)
    abs_error = np.abs(error)
    abs_derror = np.minimum(abs_error, delta)
    derror = np.sign(error) * abs_derror
    derror_cor = np.where(anchor_samples_expand == anchor_sample_correct, 0, derror)
    expect = np.repeat(head, prediction_shape[-1]).reshape(prediction_shape) * derror_cor
    output = np.full(prediction.shape, np.nan, prediction_dtype)
    return anchor_samples, expect, head, output, prediction, target
