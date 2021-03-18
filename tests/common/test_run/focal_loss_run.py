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
from tests.common.tensorio import compare_tensor
from akg.utils import kernel_exec as utils
from tests.common.test_op import focal_loss
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


def focal_loss_run(shape, p_dtype, t_dtype, gamma, kernel_name, attrs):
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(focal_loss.focal_loss, [shape, shape], [p_dtype, t_dtype], op_attrs=[gamma],
                                  kernel_name=kernel_name, attrs=attrs, tuning=t)
        if t:
            expect, pred, targ = gen_data(attrs, gamma, p_dtype, shape, t_dtype)
            output = np.full(expect.shape, 0.0, p_dtype)
            return mod, expect, (pred, targ, output)
        else:
            return mod
    else:
        mod = utils.op_build_test(focal_loss.focal_loss, [shape, shape], [p_dtype, t_dtype], op_attrs=[gamma],
                                  kernel_name=kernel_name, attrs=attrs)
        expect, pred, targ = gen_data(attrs, gamma, p_dtype, shape, t_dtype)
        output = np.full(expect.shape, 0.0, p_dtype)
        output = utils.mod_launch(mod, (pred, targ, output), expect=expect)

        return (pred, targ), output, expect, compare_tensor(output, expect, rtol=5e-2, atol=1e-4)


def gen_data(attrs, gamma, p_dtype, shape, t_dtype):
    #  if mode=='rpc_cloud': # 4\6\7\12\13 fail in multi core
    if not utils.product_is_mini():
        #  if (mode=='rpc_cloud' or mode=='aic_cloud') and p_dtype=='float16':
        attrs['enable_multicore'] = True
    pred = abs(random_gaussian(shape, miu=1, sigma=0.1).astype(p_dtype) * 10.0).astype(p_dtype)
    targ = (np.eye(shape[-1])[np.random.randint(0, shape[-1], size=shape[:-1])]).astype(t_dtype)
    expect = benchmark(pred, targ, gamma)
    return expect, pred, targ
