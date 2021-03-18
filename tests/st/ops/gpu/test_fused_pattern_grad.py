# Copyright 2020 Huawei Technologies Co., Ltd
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
# limitations under the License

import numpy as np
import akg.tvm as tvm
import akg.topi as topi

def relu_grad_np(head, in_data):
    return np.where(in_data > 0, head, 0)

def bn_beta_grad_np(head, layout='NHWC'):
    if layout == 'NCHW':
        head = topi.transpose(head, (0, 2, 3, 1))
    elif layout != "NHWC":
        raise NotImplementedError(
            'layout is not supported {} '.format(layout)
        ) 
    
    bn_beta_grad = np.sum(head, axis=(0, 1, 2))
    return bn_beta_grad

def bn_gamma_grad_np(head, in_data, data_sum, layout='NHWC'):
    if layout == 'NCHW':
        head = topi.transpose(head, (0, 2, 3, 1))
    elif layout != "NHWC":
        raise NotImplementedError(
            'layout is not supported {} '.format(layout)
        )

    n, h, w, c = head.shape
    mean = np.divide(data_sum, n * h * w)
    x_hat = np.subtract(in_data, mean)
    x_hat_mul = np.multiply(x_hat, head)
    bn_gamma_grad = np.sum(x_hat_mul, axis=(0, 1, 2))
    return bn_gamma_grad

    