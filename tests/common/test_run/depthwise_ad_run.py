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

import os

import numpy as np
import akg.topi
import akg.topi.testing
from akg.utils import kernel_exec as utils
from test_op import group_conv_ad
import time
from . import group_conv_ad_run


def depthwise_ad_run(N, H, W, CI, k_ch, KH, KW, PAD_H, PAD_W, SH, SW, cutH, cutCo, cutM, cutK, cutN, attrs):
    block_size = 16
    LOG_FILE_NAME = "log.txt"
    CO = k_ch * CI
    group = CI // block_size

    return group_conv_ad_run.group_conv_ad_run(N, H, W, CI, CO, group, KH, KW, PAD_H, PAD_W, SH, SW, cutH, cutCo, cutM, cutK, cutN, attrs)
