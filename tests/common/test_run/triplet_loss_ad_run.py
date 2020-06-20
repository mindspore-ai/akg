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
from test_op import triplet_loss_grad
from test_op import triplet_loss_ad
from gen_random import random_gaussian

def triplet_loss_grad_run(shape, dtype, margin=12.0, kernel_name="triplet_loss_grad", attrs={}):
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(triplet_loss_grad.triplet_loss_naive_grad,
                                  [shape, shape, shape, shape[:-1]], [dtype, dtype, dtype, dtype],
                                  op_attrs=[margin], kernel_name=kernel_name, attrs=attrs, tuning=t)
        if t:
            anchor, expect_d_anchor, expect_d_negative, expect_d_positive,\
            grad, neg, output_d_anchor, output_d_negative, output_d_positive, pos =\
                gen_grad_data(dtype, margin, shape)
            return mod, (expect_d_anchor, expect_d_positive, expect_d_negative), {
                "args": (anchor, pos, neg, grad, output_d_anchor, output_d_positive, output_d_negative),
                'outputs': (-3, -2, -1), 'tuning': False}
        else:
            return mod
    else:
        mod = utils.op_build_test(triplet_loss_grad.triplet_loss_naive_grad,
                                  [shape, shape, shape, shape[:-1]], [dtype, dtype, dtype, dtype],
                                  op_attrs=[margin], kernel_name='triplet_loss_grad', attrs=attrs)
        anchor, expect_d_anchor, expect_d_negative, expect_d_positive,\
        grad, neg, output_d_anchor, output_d_negative, output_d_positive, pos =\
            gen_grad_data(dtype, margin, shape)
        output_d_anchor, \
            output_d_positive, \
            output_d_negative = utils.mod_launch(mod, (
                anchor, pos, neg, grad, output_d_anchor, output_d_positive, output_d_negative),
                outputs=(-3, -2, -1), expect=(expect_d_anchor, expect_d_positive, expect_d_negative))

        assert_res = compare_tensor(output_d_anchor, expect_d_anchor, rtol=5e-03, atol=5e-2, equal_nan=True)
        assert_res &= compare_tensor(output_d_positive, expect_d_positive, rtol=5e-03, atol=5e-2, equal_nan=True)
        assert_res &= compare_tensor(output_d_negative, expect_d_negative, rtol=5e-03, atol=5e-2, equal_nan=True)
        return grad, (output_d_anchor, output_d_positive, output_d_negative), \
            (expect_d_anchor, expect_d_positive, expect_d_negative), assert_res


def gen_grad_data(dtype, margin, shape):
    anchor = np.arange(np.prod(shape)).reshape(shape).astype(dtype)
    pos = anchor + 0.5
    neg = anchor + 2.0
    d_pos = np.sum((anchor - pos) * (anchor - pos), -1)
    d_neg = np.sum((anchor - neg) * (anchor - neg), -1)
    output_forward = margin + d_pos - d_neg
    output_forward[output_forward < 0.0] = 0.0
    output_forward[output_forward > 0.0] = 1.0
    support_list = {"float16": np.float16, "float32": np.float32}
    grad = random_gaussian(shape[:-1], miu=1, sigma=0.1).astype(support_list[dtype])
    d_pos1 = anchor - pos
    d_neg1 = anchor - neg
    expect_d_anchor = 2.0 * np.transpose(grad * output_forward * np.transpose((d_pos1 - d_neg1), (1, 0)), (1, 0))
    expect_d_positive = -2.0 * np.transpose(grad * output_forward * np.transpose(d_pos1, (1, 0)), (1, 0))
    expect_d_negative = 2.0 * np.transpose(grad * output_forward * np.transpose(d_neg1, (1, 0)), (1, 0))
    output_d_anchor = np.full(expect_d_anchor.shape, np.nan, dtype)
    output_d_positive = np.full(expect_d_positive.shape, np.nan, dtype)
    output_d_negative = np.full(expect_d_negative.shape, np.nan, dtype)
    return anchor, expect_d_anchor, expect_d_negative, expect_d_positive, grad, neg, output_d_anchor, output_d_negative, output_d_positive, pos


def triplet_loss_ad_run(shape, dtype, margin=12.0, kernel_name="triplet_loss_grad", attrs={}):
    support_list = {"float16": np.float16, "float32": np.float32}
    anchor = np.arange(np.prod(shape)).reshape(shape).astype(dtype)
    pos = anchor + 0.5
    neg = anchor + 2.0
    d_pos = np.sum((anchor - pos) * (anchor - pos), -1)
    d_neg = np.sum((anchor - neg) * (anchor - neg), -1)
    output_forward = margin + d_pos - d_neg
    output_forward[output_forward < 0.0] = 0.0
    output_forward[output_forward > 0.0] = 1.0

    d_pos1 = anchor - pos
    d_neg1 = anchor - neg
    assert_res = True
    output_all = list()
    expect_all = list()

    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        grad = random_gaussian(shape[:-1], miu=1, sigma=0.1).astype(support_list[dtype])
        for input_id in range(3):
            mod = utils.op_build(triplet_loss_ad.triplet_loss_ad,
                                 [grad.shape, shape, shape, shape], [dtype, dtype, dtype, dtype],
                                 op_attrs=[margin, input_id], kernel_name=kernel_name, attrs=attrs, tuning=t)
            if t:
                expect, output = gen_data(d_neg1, d_pos1, dtype, grad, input_id, output_forward)
                return mod, expect, (grad, anchor, pos, neg, output)
            else:
                return mod
    else:
        grad = random_gaussian(shape[:-1], miu=1, sigma=0.1).astype(support_list[dtype])
        # Testing AD for 3 inputs of the triplet_loss op:
        # 0 - for "anchor_output"
        # 1 - for "positive_output"
        # 2 - for "negative_output"
        for input_id in range(3):
            mod = utils.op_build(triplet_loss_ad.triplet_loss_ad,
                                 [grad.shape, shape, shape, shape], [dtype, dtype, dtype, dtype],
                                 op_attrs=[margin, input_id], kernel_name='triplet_loss_ad', attrs=attrs)
            expect, output = gen_data(d_neg1, d_pos1, dtype, grad, input_id, output_forward)
            output = utils.mod_launch(mod, [grad, anchor, pos, neg, output])
            assert_res &= compare_tensor(output, expect, rtol=5e-03, atol=5e-2, equal_nan=True)
            output_all.append(output)
            expect_all.append(expect)

        return grad, tuple(output), tuple(expect), assert_res


def gen_data(d_neg1, d_pos1, dtype, grad, input_id, output_forward):
    if (input_id == 0):
        print("Testing Autodiff for gradient wrt. anchor_output")
        expect = 2.0 * np.transpose(grad * output_forward * np.transpose((d_pos1 - d_neg1), (1, 0)), (1, 0))
    elif (input_id == 1):
        print("Testing Autodiff for gradient wrt. positive_output")
        expect = -2.0 * np.transpose(grad * output_forward * np.transpose(d_pos1, (1, 0)), (1, 0))
    else:
        print("Testing Autodiff for gradient wrt. negative_output")
        expect = 2.0 * np.transpose(grad * output_forward * np.transpose(d_neg1, (1, 0)), (1, 0))
    output = np.full(expect.shape, np.nan, dtype)
    return expect, output
