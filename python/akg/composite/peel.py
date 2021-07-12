#!/usr/bin/env python3
# coding: utf-8
# Copyright 2021 Huawei Technologies Co., Ltd
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

import akg.tvm


class CompositePeel(object):
    """Provide interface for C++ DimensionPeeler"""

    def __init__(self, desc):
        self.desc = desc
        self.stmt = None
        self.build_info = None
        self.peeling_space = None

    def analyze(self):
        func = akg.tvm.get_global_func("composite_peel_analyze")
        ret = func(self.desc)
        self.stmt = ret["stmt"]
        self.build_info = ret["build_info"]
        self.peeling_space = ret["peeling_space"]

    def get_peeling_space(self):
        if self.peeling_space is None:
            return None
        peeling_space = [s.value for s in self.peeling_space]
        return peeling_space

    def get_peeled_desc(self, peeling):
        """
        Returns a peeled json str using the give peeling, peeling is str composed of axis value pairs,
          e.g. "0 1024 1 1024", "0 1024", "1, 1024" are valid ones
        """
        func = akg.tvm.get_global_func("get_peeled_body")
        peeled_body = func(self.stmt, peeling)
        dump_func = akg.tvm.get_global_func("dump_to_json")
        return dump_func(peeled_body, self.build_info)


def composite_peel_analyze(desc):
    """
    Analyzes the peeling space for a give json str.
    Args:
       desc: json str

    Returns:
       CompositePeel.
    """
    peel = CompositePeel(desc)
    peel.analyze()
    return peel
