#!/usr/bin/env python3
# coding: utf-8
# Copyright 2025 Huawei Technologies Co., Ltd
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

import os
if not os.getenv("ENABLE_SWFT_JIT", 0):
    exit()
import mindspore as ms
import numpy as np
import sys
import swft
from pathlib import Path
from swft.core import *
from swft.api import *


parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
from verify_result import verify_result


@swft.jit(core_num=8)
def ms_tanh_kernel(x, out):
    x_ub = move_to_ub(x)
    tanh_ub = tanh(x_ub)
    out.load(tanh_ub)


class Net(ms.nn.Cell):
    def __init__(self) -> None:
        super().__init__()

    def construct(self, x, out):
        ms_tanh_kernel(x, out)
        return ms.ops.add(out, 1)


if __name__ == '__main__':
    set_context("310P")
    ms.set_context(mode=ms.GRAPH_MODE)
    x_np = np.random.uniform(-1, 1, [512]).astype(np.float16)
    x = ms.Tensor(x_np)
    out = ms.Tensor(np.zeros([512]).astype(np.float16))
    Net = compile_ms_cell(Net)
    net = Net()
    actual = net(x, out)
    np_expect = np.tanh(x_np) + 1
    sys.exit(verify_result(output=actual.numpy(), golden=np_expect) >> 8)
