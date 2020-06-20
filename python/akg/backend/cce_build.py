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

"""Runtime function related hooks."""
from __future__ import absolute_import as _abs
from akg import build_module


def debug_mode(debug_flag):
    """
    Pass to enable tpu debug mode.

    Args:
        debug_flag (int): The dbeug flag to be passed.

    Returns:
        list of function, the pass to set to build_config(add_lower_pass=tpu.debug_mode(mode)).
    """
    # the number in pass_list such as 0,1,2,3 represents the order of the pass called
    pass_list = []
    if debug_flag == 1:
        pass_list.append((0, ir_pass.inject_dma_intrin))
    return pass_list


# Add a lower pass to sync uop
build_config = build_module.build_config(add_lower_pass=debug_mode(0), dump_pass_ir=True)
