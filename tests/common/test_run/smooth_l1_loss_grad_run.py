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

import numpy as np
from tensorio import compare_tensor
from akg.utils import kernel_exec as utils
from test_op import smooth_l1_loss_grad
from gen_random import random_gaussian


def smooth_l1_loss_grad_run(shape, dtype, attrs=None, kernel_name="smooth_l1_loss_grad"):
    assert len(shape) >= 2, "last dimension of the shape will be reduced, so the shape length should be >= 2"
    sample_shape = shape[:-1]

    anchor_samples_dtype = "int32"
    # sigma is a constant parameter
    sigma = 1.0
    anchor_sample_correct = 0

    if not utils.product_is_mini():
        attrs['enable_align_fix'] = True
        attrs['enable_multicore'] = True

    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(smooth_l1_loss_grad.smooth_l1_loss_grad, [sample_shape, shape, shape, sample_shape],
                                  [dtype, dtype, dtype, anchor_samples_dtype], op_attrs=[sigma, anchor_sample_correct],
                                  attrs=attrs, kernel_name=kernel_name, dump_cce=True, tuning=t)
        if t:
            anchor_samples, dloss, expect, output, prediction, prediction_, target, target_ = gen_data(
                anchor_sample_correct, anchor_samples_dtype, dtype, sample_shape, shape, sigma)
            return mod, expect, (dloss, prediction, target, anchor_samples, output)
        else:
            return mod
    else:
        anchor_samples, dloss, expect, output, prediction, prediction_, target, target_ = gen_data(
            anchor_sample_correct, anchor_samples_dtype, dtype, sample_shape, shape, sigma)
        mod = utils.op_build_test(smooth_l1_loss_grad.smooth_l1_loss_grad,
                                  [sample_shape, shape, shape, sample_shape],
                                  [dtype, dtype, dtype, anchor_samples_dtype], op_attrs=[sigma, anchor_sample_correct],
                                  attrs=attrs, kernel_name=kernel_name, dump_cce=True)
        output = utils.mod_launch(mod, (dloss, prediction, target, anchor_samples, output), expect=expect)
        return (dloss, prediction, target, anchor_samples), output, expect, compare_tensor(output, expect, atol=5e-3,
                                                                                           rtol=5e-3)


def gen_data(anchor_sample_correct, anchor_samples_dtype, dtype, sample_shape, shape, sigma):
    target = random_gaussian(shape, miu=0, sigma=5).astype(dtype)
    diff = random_gaussian(shape, miu=0, sigma=1).astype(dtype)
    prediction = np.add(target, diff)
    # anchor_samples == 16 indicates prediction == target, derivative should be zero.
    anchor_samples_in = random_gaussian(sample_shape, miu=0, sigma=0.2).astype(dtype)
    # random 16 or 0, anchor_samples = 16 with a small probability
    anchor_samples = np.where(anchor_samples_in > 0.4, anchor_sample_correct, anchor_sample_correct + 1).astype(
        anchor_samples_dtype)
    dloss = random_gaussian(sample_shape, miu=0, sigma=2).astype(dtype)
    diff = np.subtract(prediction, target)
    abs_diff = np.abs(diff)
    second_branch = np.where(0 < diff, 1, -1)
    dloss_broadcast = np.expand_dims(dloss, axis=len(shape)) + np.zeros(shape[-1])
    anchor_samples_broadcast = np.expand_dims(anchor_samples, axis=len(shape)) + np.zeros(shape[-1])
    gradient = np.where(abs_diff <= (1 / (sigma * sigma)), diff * sigma * sigma, second_branch) * dloss_broadcast
    expect = np.where(anchor_samples_broadcast == anchor_sample_correct, 0, gradient)

    prediction_ = None
    target_ = None
    output = np.full(expect.shape, np.nan, dtype)
    return anchor_samples, dloss, expect, output, prediction, prediction_, target, target_
