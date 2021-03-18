#!/usr/bin/env python3
# coding: utf-8
# Copyright 2019-2021 Huawei Technologies Co., Ltd
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

"""__init__"""
from __future__ import absolute_import as _abs
import logging
from akg.tvm._ffi.function import _init_api
if __name__ == "platform":
    import sys
    import os
    logging.info("Using python build-in 'platform'")
    tp_ = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))

    bak_path = sys.path[:]
    for item in bak_path:
        if (item == '' or os.path.realpath(item) == tp_) and item in sys.path:
            sys.path.remove(item)

    sys.modules.pop('platform')
    sys.path.insert(0, '')
    sys.path.append(tp_)
_init_api("akg.build_module")
