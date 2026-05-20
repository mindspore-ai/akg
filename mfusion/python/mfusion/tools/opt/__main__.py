# Copyright 2026 Huawei Technologies Co., Ltd
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

"""MFusion modular optimizer driver

Typically, when installed from a wheel, this can be invoked as:

  mfusion-opt [options] <input file>

To see available passes, dialects, and options, run:

  mfusion-opt --help
"""
import os
import subprocess
import sys


def _get_builtin_tool(exe_name: str) -> str:
    this_path = os.path.dirname(__file__)
    tool_path = os.path.join(this_path, "..", "..", "_mlir_libs", exe_name)
    return tool_path


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    exe = _get_builtin_tool("mfusion-opt")
    return subprocess.call(args=[exe] + args)


if __name__ == "__main__":
    sys.exit(main())
