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

"""Third part operator of splited Grad function of BN"""
from akg.ops.nn import fused_batch_norm_grad_split

def BNGrad3(dy, rs, dgamma_dx, dbeta_dx, data_minus_mean):
    """BNGrad3"""
    return fused_batch_norm_grad_split.fused_bn_grad_3(dy, rs, dgamma_dx, dbeta_dx, data_minus_mean)
