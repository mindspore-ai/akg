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
import akg
import akg.tvm
from akg.utils import kernel_exec as utils
from akg.utils.kernel_exec import product_is_mini
from tests.common.test_op.ascend import focalloss_grad
from tests.common.test_run.ascend.focal_loss_run import softmax as softmax_np
from tests.common.test_run.ascend.focal_loss_run import logsoftmax as logsoftmax_np
from tests.common.test_op.ascend.focalloss_ad import focalloss_ad
from tests.common.tensorio import compare_tensor
from tests.common.gen_random import random_gaussian

def softmax(x):
    mv = np.max(x, axis=-1, keepdims=True)
    v = x - mv
    s = np.exp(v) / np.sum(np.exp(v), axis=-1, keepdims=True)
    return s


def logsoftmax(x):
    mv = np.max(x, axis=-1, keepdims=True)
    v = x - mv
    exp_x = np.exp(v)
    Z = np.sum(exp_x, axis=-1, keepdims=True)
    return v - np.log(Z)


def benchmark(x, y, gamma):
    y_pred = softmax(x)
    expect = -y * ((1 - y_pred) ** gamma) * logsoftmax(x)
    res = np.sum(expect, axis=-1)
    return res


def RANGEFILL(shape, offset=0):
    size = np.prod([d for d in shape])
    return np.arange(0 + offset, size + offset, dtype).reshape(shape)


def focalloss_ad_run(shape, p_dtype, t_dtype, gamma, kernel_name, attrs):
    head = np.random.rand(*shape[:2]).astype(p_dtype)

    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(focalloss_ad, [head.shape, shape, shape], [p_dtype, p_dtype, t_dtype],
                                  op_attrs=[gamma], kernel_name=kernel_name, attrs=attrs, tuning=t)
        if t:
            expect, output, pred, targ = gen_data(gamma, head, p_dtype, shape, t_dtype)
            return mod, expect, (head, pred, targ, output)
        else:
            return mod
    else:
        mod = utils.op_build_test(focalloss_ad, [head.shape, shape, shape], [p_dtype, p_dtype, t_dtype],
                                  op_attrs=[gamma], kernel_name=kernel_name, attrs=attrs)
        expect, output, pred, targ = gen_data(gamma, head, p_dtype, shape, t_dtype)
        output = utils.mod_launch(mod, [head, pred, targ, output], expect=expect)
        return [head, pred, targ, gamma], output, expect, compare_tensor(output, expect, rtol=5e-03, atol=5e-03,
                                                                         equal_nan=True)


def gen_data(gamma, head, p_dtype, shape, t_dtype):
    pred = np.random.rand(*shape).astype(p_dtype)
    targ = (np.eye(shape[-1])[np.random.randint(0, shape[-1], size=shape[:-1])]).astype(t_dtype)
    y = benchmark(pred, targ, gamma)
    expect_labels, expect_logits = get_expect(pred, targ, head, gamma)
    expect = expect_logits
    out_shape = expect.shape
    output = np.full(out_shape, 0, p_dtype)
    return expect, output, pred, targ


def focalloss_ad_run2(shape, dtype, attrs):
    logits_pld = akg.tvm.placeholder(shape, dtype=dtype, name='logits')
    labels_pld = akg.tvm.placeholder(shape, dtype='int32', name='labels')
    d_labels, d_logits, head = focalloss_ad.focalloss_ad(labels_pld, logits_pld)
    print("autodiff d_logits:\n", akg.tvm.PrintTensorRecursively(d_logits))
    print("autodiff d_labels:\n", akg.tvm.PrintTensorRecursively(d_labels))

    # build autodiff kernels
    io = [labels_pld, logits_pld, head, d_labels, d_logits]
    s = akg.tvm.create_schedule([e.op for e in io])
    kernel_name = utils.gen_name_kernel("focalloss_ad", dtype, (shape[0], shape[1],))
    with akg.build_config(add_lower_pass=utils.debug_mode(0), dump_pass_ir=True):
        mod = akg.build(s, io, "cce", name=kernel_name, attrs=attrs, polyhedral=True)

    labels_np = RANGEFILL((batchsize,))
    logits_np = RANGEFILL((batchsize,), 2)
    head_np = RANGEFILL((batchsize,), 2)
    output = np.full(expect.shape, np.nan, dtype)
    output = utils.mod_launch(mod, (labels_np, logits_np, head_np, output), expect=output)
    expect = output  # hack

    return (input_np, head_np), output, expect, compare_tensor(output, expect, atol=0.1)


def get_expect(prediction, labels, grad, gamma):
    if str(prediction.dtype) == "float16":
        eps = 1e-4
    else:
        eps = 1e-8
    pred = softmax_np(prediction).astype(prediction.dtype)
    log_p = logsoftmax_np(prediction).astype(pred.dtype)
    neg_pred_pow = np.exp(gamma * np.log(1 - pred + eps)).astype(pred.dtype)

    d_labels = -neg_pred_pow * log_p * np.expand_dims(grad, -1).astype(pred.dtype)

    d_logits1 = (-labels * (-log_p * gamma * neg_pred_pow * pred + neg_pred_pow * (1 - pred))).astype(pred.dtype)

    d_logits2 = np.zeros(pred.shape, dtype=pred.dtype)
    for k in range(pred.shape[2]):
        d_logits2[:, :, k] += np.sum(-labels * (gamma * neg_pred_pow / (1 - pred + eps) *
                                                np.expand_dims(pred[:, :, k], -1) * pred *
                                                log_p - neg_pred_pow * np.expand_dims(pred[:, :, k], -1)),
                                     axis=2)
    d_logits = np.expand_dims(grad, -1) * (labels * d_logits1 + (1 - labels) * d_logits2)
    return d_labels, d_logits


def focalloss_grad_run(shape, dtype, label_dtype, gamma, attrs):
    kernel_name = utils.gen_name_kernel("focalloss_grad", dtype, shape)
    attrs["pragma_disable_whole_component"] = False
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(focalloss_grad.focal_loss_bwd,
                                  input_shapes=[shape, shape, shape[:2]],
                                  input_types=[dtype, label_dtype, dtype],
                                  op_attrs=[gamma], kernel_name=kernel_name, attrs=attrs, tuning=t)
        if t:
            d_logits, expect_logits, grad_np, labels_np, logits_np = gen_grad_data(attrs, dtype, gamma, label_dtype,
                                                                                   shape)
            return mod, expect_logits, (logits_np, labels_np, grad_np, d_logits)
        else:
            return mod
    else:
        mod = utils.op_build_test(focalloss_grad.focal_loss_bwd,
                                  input_shapes=[shape, shape, shape[:2]],
                                  input_types=[dtype, label_dtype, dtype],
                                  op_attrs=[gamma], kernel_name=kernel_name, attrs=attrs)
        d_logits, expect_logits, grad_np, labels_np, logits_np = gen_grad_data(attrs, dtype, gamma, label_dtype, shape)
        d_logits = utils.mod_launch(mod, (logits_np, labels_np, grad_np, d_logits), expect=expect_logits)
        result_logits = compare_tensor(d_logits, expect_logits, rtol=2e-2, atol=1e-4)

        return (labels_np, logits_np, grad_np), d_logits, expect_logits, result_logits


def gen_grad_data(attrs, dtype, gamma, label_dtype, shape):
    if not product_is_mini():
        attrs['enable_align_fix'] = True
        attrs['enable_multicore'] = True

    logits_np = random_gaussian(shape, miu=50, sigma=80).astype(dtype)
    labels_np = (np.eye(shape[-1])[np.random.randint(0, shape[-1], size=shape[:-1])]).astype(label_dtype)
    grad_np = np.ones(shape[:2]).astype(dtype)
    expect_labels, expect_logits = get_expect(logits_np, labels_np, grad_np, gamma)
    d_logits = np.full(expect_logits.shape, np.nan, dtype)
    return d_logits, expect_logits, grad_np, labels_np, logits_np
