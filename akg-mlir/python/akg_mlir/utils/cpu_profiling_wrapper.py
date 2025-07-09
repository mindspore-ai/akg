# Copyright 2023 Huawei Technologies Co., Ltd
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
# ============================================================================

"""CPU Profiling Wrapper Function"""
import argparse
import os
import pathlib


def wrap_timer_func(file, kernel_name, profiling_trails):
    """generate the file for cpu profiling

    Args:
        file (str): mlir file name
        kernel_name (str): kernel name
        profiling_trails (int): the number of profiling trails

    Returns:
        str: timer file name
    """
    if not file.endswith(".mlir"):
        return file
    template_src = ""
    with open(os.path.join(str(pathlib.Path(__file__).absolute().parent), "cpu_profiling_template.txt"), 'r') as f:
        template_src = f.read()
    inputs_name = ""
    inputs_ptr = ""
    file_src = ""
    kernel_func_line_id = 0
    with open(file, 'r') as f:
        file_src = f.readlines()
        for idx, line in enumerate(file_src):
            if "llvm.func @Fused_" in line:
                kernel_func_line_id = idx
                break
    inputs_name, inputs_ptr = file_src[kernel_func_line_id].split("(")[1].split(")")[0].split(": ")
    template_src = template_src.replace("KERNEL_NAME", kernel_name)
    template_src = template_src.replace("INPUTS_NAME", inputs_name)
    template_src = template_src.replace("INPUTS_PTR", inputs_ptr)
    template_src = template_src.replace("CTIMES", str(profiling_trails))
    module_func = "".join(file_src[:kernel_func_line_id])
    kernel_func = "".join(file_src[kernel_func_line_id:])
    wrapped_timer_src = "\n".join([module_func, template_src, kernel_func])
    timer_file = file.split(".")[0] + "_wrapped_timer." + file.split(".")[1]
    with open(timer_file, "wt") as f:
        f.writelines(wrapped_timer_src)
    return timer_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run cpu profiling wrapper")
    parser.add_argument("-f", "--file", type=str, help="Run single file.")
    parser.add_argument("-kn", "--kernel_name", type=str, help="kernel name")
    parser.add_argument("-tr", "--prof_trails", type=int,
                        required=False, default=0)
    args = parser.parse_args()

    _ = wrap_timer_func(args.file, args.kernel_name, args.prof_trails)
