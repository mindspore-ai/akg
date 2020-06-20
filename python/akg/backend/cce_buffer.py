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

"""register the cce buffer info"""
from __future__ import absolute_import as _abs

import akg.tvm
from akg.utils import kernel_exec
from .cce_conf import cce_product_

# add default product, default value is 200
# get the CceProductParams instance
mode = kernel_exec.get_runtime_mode()

# currently we have 5 kinds of runtime modes:ca/aic/rpc/aic_cloud/rpc_cloud
# aic means aic_mini;rpc means rpc_mini
# the default target is mini
if mode in ('aic', 'rpc', 'compile_mini', 'air'):
    cur_cce_product_params = cce_product_("1.1.xxx.xxx")
    target = akg.tvm.target.cce("mini")
elif mode in ('aic_cloud', 'rpc_cloud', 'compile_cloud', 'air_cloud'):
    cur_cce_product_params = cce_product_("1.6.xxx.xxx")
    target = akg.tvm.target.cce("cloud")
elif mode == 'ca':
    cur_cce_product_params = cce_product_("1.1.xxx.xxx")
    target = akg.tvm.target.cce("mini")
else:
    cur_cce_product_params = cce_product_("1.1.xxx.xxx")
    target = akg.tvm.target.cce("mini")
target.__enter__()
