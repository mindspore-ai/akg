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
from akg.utils import kernel_exec as utils
from tests.common.test_op.ascend import fc
from tests.common.tensorio import compare_tensor
from tests.common.gen_random import random_gaussian

def fc_run(fmapshape, weightshape, fc_dtype, block_size, random_type, attrs):
    # Result_Numpy
    if random_type:
        fMapBatch = random_gaussian(fmapshape, miu=1, sigma=0.1).astype(np.float16)
        weight = random_gaussian(weightshape, miu=1, sigma=0.1).astype(np.float16)
    else:
        fMapBatch = np.ones(fmapshape, dtype=np.float16)
        weight = np.ones(weightshape, dtype=np.float16)

    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(fc.fc, [fMapBatch, weight], [fc_dtype, fc_dtype], op_attrs=[block_size],
                                  kernel_name=kernel_name, attrs=attrs, tuning=t)
        if t:
            args, exp_output, inputs = gen_data(block_size, fMapBatch, weight)
            return mod, exp_output, args
        else:
            return mod
    else:
        # Result_fc
        mod = fc.fc(fMapBatch, weight, fc_dtype, block_size, attrs)
        args, exp_output, inputs = gen_data(block_size, fMapBatch, weight)
        acu_output = utils.mod_launch(mod, args, expect=exp_output)

        # compare result
        TestCase_Result = compare_tensor(acu_output, exp_output, rtol=5e-03, equal_nan=True)

        return inputs, acu_output, exp_output, TestCase_Result


def gen_data(block_size, fMapBatch, weight):
    out = None
    N, C, H, W = fMapBatch.shape
    F, _, HH, WW = weight.shape
    Ho = 1 + H - HH
    Wo = 1 + W - WW
    assert Ho == 1 and Wo == 1
    out = np.zeros((N, F, Ho, Wo))
    for f in range(F):
        for i in range(Ho):
            for j in range(Wo):
                out[:, f, i, j] = np.sum(fMapBatch[:, :, i: i + HH, j: j + WW] * weight[f, :, :, :], axis=(1, 2, 3))
    exp_output = out.reshape(N, F // block_size, block_size, Ho, Wo).transpose(0, 1, 3, 4, 2).copy()
    # inputs and output to hold the data
    # >>>>NCHW
    f_n, f_c, f_h, f_w = fMapBatch.shape
    w_n, w_c, w_h, w_w = weight.shape
    # >>>>inputs
    fmap_data = fMapBatch.reshape(f_n, f_c // block_size, block_size, f_h, f_w).transpose(0, 1, 3, 4, 2).copy()
    filter_data = weight.reshape(w_n, w_c // block_size, block_size, w_h, w_w).transpose(1, 3, 4, 0, 2).copy()
    inputs = [fmap_data, filter_data]
    # >>>>output
    out_shape_nc1hwc0 = (f_n, w_n // block_size, 1, 1, block_size)
    out_data = np.full(out_shape_nc1hwc0, 0, 'float16')
    # >>>>args
    args = [fmap_data, filter_data, out_data]
    return args, exp_output, inputs
