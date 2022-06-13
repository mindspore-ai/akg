# Copyright 2022 Huawei Technologies Co., Ltd
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

"""
Info version adapt

History:
======================================================================
v0(before r1.8):
The kernel info jsons without "version" field are treated as version 0.

v1(r1.8):
1. Add a "version" field, starts from 1.
2. Add a "target_info" field:
   * on cpu, it contains "arch", "system" and "feature" fields.
   * on gpu, it contains "compute_capability" and "sm_count" fields,
     the "compute_capability" already exists from v0.
"""

import logging


def _adapt_version_0(kernel_info: dict):
    if "compute_capability" in kernel_info:
        # move the "compute_capability" into "target_info" field.
        if kernel_info.get("process", None) == "cuda":
            kernel_info["target_info"] = {"compute_capability": kernel_info["compute_capability"]}
        kernel_info.pop("compute_capability")


_adapt_func_list = [
    _adapt_version_0,
]


class InfoVersionAdapt:
    """Info version adapt"""
    CURRENT_VERSION = len(_adapt_func_list)

    def __init__(self, kernel_info):
        self.kernel_info = kernel_info
        self.msg = ""

    def run(self):
        version = self.kernel_info.get("version", 0)
        if version > self.CURRENT_VERSION:
            self.msg = "The akg only supports kernel info of version up to {}, but got kernel info of version {}, please upgrade mindspore for akg.".format(
                self.CURRENT_VERSION, version)
            logging.error(self.msg)
            return False
        if version < self.CURRENT_VERSION:
            for i in range(version, self.CURRENT_VERSION):
                func = _adapt_func_list[i]
                func(self.kernel_info)
        return True
