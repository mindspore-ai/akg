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

# def the buffer var
scope_cbuf = "local.L1"
scope_ubuf = "local.UB"
scope_ca = "local.L0A"
scope_cb = "local.L0B"
scope_cc = "local.L0C"
scope_reg = "local.REG"
scope_aicpu = "local_aicpu"

dma_copy = "dma_copy"
dma_copy_global = "global"

f = akg.tvm.get_global_func("cce_util.GetCceAxis")

# def the cce thread axis for sync
CCE_AXIS = f()

# def the gemm const
WGT_WIDTH = 16
INP_WIDTH = 16
OUT_WIDTH = 16
BLOCK_IN = 16
BLOCK_OUT = 16
BLOCK_REDUCE = 16

INP_ELEM_BYTES = (BLOCK_IN * BLOCK_REDUCE * INP_WIDTH // 8)
WGT_ELEM_BYTES = (BLOCK_OUT * BLOCK_REDUCE * WGT_WIDTH // 8)
OUT_ELEM_BYTES = (BLOCK_IN * BLOCK_OUT * OUT_WIDTH // 8)
GLB_ELEM_BYTES = (16 * OUT_WIDTH // 8)
