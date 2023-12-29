# Copyright 2022-2023 Huawei Technologies Co., Ltd
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

v2(r2.2):
1. For all the ops, the number of inputs in akg info is equal to the number of
real inputs, which means that no inputs are converted to attributes in the info.
AKG needs to handle this process by itself from now on.
2. For those inputs need to be converted to attributes, a key in "input_desc" of
this op named "value" is guaranteed, which contains the value of this input.
"""

import logging


def convert_input_to_attr(kernel_info:dict):
    op_info = {
        "Reshape": [(1, "shape")],
        "ReduceMax": [(1, "axis")],
        "ReduceMin": [(1, "axis")],
        "ReduceSum": [(1, "axis")],
        "Transpose": [(1, "perm")],
        "ExpandDims": [(1, "axis")],
        "Tile": [(1, "multiples")],
        "StridedSlice": [(3, "strides"), (2, "end"), (1, "begin")],
        "OneHot": [(1, "depth")],
        "Gather": [(2, "axis")],
        "UnsortedSegmentSum": [(2, "num_segments")],
        "CumSum": [(1, "axis")],
    }

    int_input_required_ops = {
        "OneHot",
        "UnsortedSegmentSum",
    }

    ops = kernel_info["op_desc"]
    for op in ops:
        op_name = op["name"]
        if op_name in op_info:
            attr = []
            if op["attr"]:
                attr = op["attr"]
            for input_info in op_info[op_name]:
                input_index = input_info[0]
                input_name = input_info[1]
                if input_index >= len(op["input_desc"]):
                    continue
                input_desc_i = op["input_desc"].pop(input_index)
                input_value = input_desc_i[0]["value"]
                input_dtype = "listInt"
                if op_name not in int_input_required_ops and isinstance(input_value, int):
                    input_value = [input_value]
                if isinstance(input_value, int):
                    input_dtype = "int"
                attr.append(
                    {"name": input_name, "dtype": input_dtype, "value": input_value}
                )
            op["attr"] = attr


def _adapt_version_0(kernel_info: dict):
    if "compute_capability" in kernel_info:
        # move the "compute_capability" into "target_info" field.
        if kernel_info.get("process", None) == "cuda":
            kernel_info["target_info"] = {"compute_capability": kernel_info["compute_capability"]}
        kernel_info.pop("compute_capability")


def _adapt_version_1(kernel_info:dict):
    pass


def _adapt_version_2(kernel_info:dict):
    if kernel_info.get("version", 0) < 2:
        return
    else:
        convert_input_to_attr(kernel_info)


_adapt_func_list = [
    _adapt_version_0,
    _adapt_version_1,
    _adapt_version_2,
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
