# Copyright 2023-2026 Huawei Technologies Co., Ltd
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
from pathlib import Path
from .code_template import CPU_PROFILING_TEMPLATE


def _find_kernel_func_line(file_src):
    """Find the line index of the kernel function in the MLIR source."""
    for idx, line in enumerate(file_src):
        if "llvm.func @Fused_" in line:
            return idx
    return 0


def _apply_template(template_src, kernel_name, inputs_name, inputs_ptr, profiling_trails):
    """Apply replacements to the template source."""
    template_src = template_src.replace("KERNEL_NAME", kernel_name)
    template_src = template_src.replace("INPUTS_NAME", inputs_name)
    template_src = template_src.replace("INPUTS_PTR", inputs_ptr)
    template_src = template_src.replace("CTIMES", str(profiling_trails))
    return template_src


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
    with open(file, 'r', encoding='utf-8') as f:
        file_src = f.readlines()
    kernel_func_line_id = _find_kernel_func_line(file_src)
    inputs_name, inputs_ptr = file_src[kernel_func_line_id].split("(")[1].split(")")[0].split(": ")
    template_src = _apply_template(CPU_PROFILING_TEMPLATE, kernel_name, inputs_name, inputs_ptr,
                                   profiling_trails)
    module_func = "".join(file_src[:kernel_func_line_id])
    kernel_func = "".join(file_src[kernel_func_line_id:])
    wrapped_timer_src = "\n".join([module_func, template_src, kernel_func])
    file_path = Path(file)
    timer_file = file_path.with_name(file_path.stem + "_wrapped_timer" + file_path.suffix)
    with open(timer_file, "wt", encoding='utf-8') as f:
        f.writelines(wrapped_timer_src)
    return timer_file
