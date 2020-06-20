#!/usr/bin/env python3
# coding: utf-8
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

"""gradient for fused batchnorm"""
from akg.ops.nn import fused_batch_norm_grad
from akg.ms.utils import DEFAULT

def FusedBatchNormGrad(dy, x, scale, save_mean, save_inv_variance, data_format=None):
    """gradient for fused batchnorm"""
    if data_format is None:
        data_format = [DEFAULT]
    if isinstance(data_format, list):
        data_format = data_format[0]
    eps = 1e-3
    return fused_batch_norm_grad.fused_batch_norm_grad(dy, x, save_mean, save_inv_variance,
                                                       scale, eps=eps, data_format=data_format, axis=1)
