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

"""apply momentum"""
from akg.ops.optimizers import apply_momentum

def ApplyMomentum(variable, accumulation, learning_rate, gradient, momentum, use_nesterov=False, gradient_scale=1.0):
    """apply momentum"""
    return apply_momentum.apply_momentum(variable, gradient, accumulation, learning_rate,
                                         momentum, use_nesterov=use_nesterov, grad_scale=gradient_scale)
