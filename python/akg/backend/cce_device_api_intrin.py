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

"""CCE configuration constants"""

from __future__ import absolute_import as _abs

import akg.tvm


def cc_device_exp(input_data, p_scale, p_shift, p_base, p_shape):
    """
    Take exp of input_data by cc_device_api.(cc_device_exp).

    Args:
    input_data (tvm.tensor.Tensor): Input argument.
    p_scale : default [1].
    p_shift: default [0].
    p_base: default [-1].
    p_shape: default [1].

    Returns:
        tvm.expr.Expr. The result.
    """
    return akg.tvm.call_pure_intrin(input_data.dtype, "cc_device_exp", input_data, p_scale, p_shift, p_base, p_shape)


def cc_device_log(input_data, p_scale, p_shift, p_base, p_shape):
    """
    Take log of input_data by cc_device_api(cc_device_log).

    Args:
        input_data (tvm.tensor.Tensor): Input argument.
        p_scale: default [1].
        p_shift: default [0].
        p_base: default [-1].
        p_shape: default [1].

    Returns:
        tvm.Expr.expr. The result.
    """
    return akg.tvm.call_pure_intrin(input_data.dtype, "cc_device_log", input_data, p_scale, p_shift, p_base, p_shape)
