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

"""dim"""
DIM = 'dim'
checkname = ["index", "axis", "tilel1", "tilel0"]


class Dim():
    """class Dim"""
    def __init__(self):
        self.dim = ""

    def setdim(self, **kwargs):
        sorted_keys = sorted([x for x in kwargs], key=checkname.index)
        for key in sorted_keys:
            if key not in checkname:
                raise ValueError("Set dim error!")
            self.dim += ' ' + str(kwargs[key])

    def __str__(self):
        return self.dim
