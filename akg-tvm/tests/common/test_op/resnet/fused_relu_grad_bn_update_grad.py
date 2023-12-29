# Copyright 2020-2021 Huawei Technologies Co., Ltd
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

"""
fused operator dsl function: fused_relu_grad_bn_update_grad
ResNet50 fused computation. 216 in XLA patterns
"""
from __future__ import absolute_import
import akg.topi as topi
import akg.utils as utils
from tests.common.test_op.resnet.fused_pattern_grad import relu_grad, bn_gamma_grad, bn_beta_grad

def fused_relu_grad_bn_update_grad(data_sum, in_bn, head_active, in_active, layout='NHWC',
                                    out_dtype="float32", target=utils.CUDA):

    if layout != 'NHWC' and layout!= "NCHW":
        raise NotImplementedError(
            'Layout not supported {} '.format(layout))

    inter_dtype = "float32"
    relugrad = relu_grad(head_active, in_active)
    relugrad_cast = topi.cast(relugrad, inter_dtype) 
    inbn_cast = topi.cast(in_bn, inter_dtype)

    bn_gamma_ad = bn_gamma_grad(relugrad_cast, inbn_cast, data_sum, layout)
    bn_beta_ad = bn_beta_grad(relugrad_cast, layout)
    
    if bn_gamma_ad.dtype != out_dtype:
        bn_gamma_ad = topi.cast(bn_gamma_ad, out_dtype)
    if bn_beta_ad.dtype != out_dtype:
        bn_beta_ad = topi.cast(bn_beta_ad, out_dtype)

    return [bn_gamma_ad, bn_beta_ad]



