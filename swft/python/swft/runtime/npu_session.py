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
from swft.core.c_expression import NPUSession as CNPUSession
from swft.core.c_expression import set_context, get_context

class NPUSession(CNPUSession):
    _instance = None

    def __init__(self, device_id=0, context="310P"):
        self.cann_path = os.getenv("ASCEND_HOME_PATH")
        set_context(context)
        if not self.cann_path:
            raise ValueError("ASCEND_HOME_PATH not set!")
        CNPUSession.__init__(self, device_id)
    
    @property
    def stream(self):
        return self._get_stream()

    @property
    def current_device(self):
        return self._get_current_device()
    
    @property
    def context(self):
        return get_context()
    
    def sync_stream(self):
        return self._sync_stream()
    
    @classmethod
    def create(cls, device_id=0, context="310P"):
        if cls._instance is not None:
            raise RuntimeError(
                "npu session already created, please use instance()")
        cls._instance = cls(device_id, context)
        cls._is_default = False
        return cls._instance
    
    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls(0)
        return cls._instance

    
    