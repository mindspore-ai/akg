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

"""Runtime code generator for testing."""

import os
import json
import pathlib
from functools import reduce
from akg_mlir.utils.dynamic_utils import get_gpu_setting_by_input, get_device_shape
from akg_mlir.utils.composite_op_helper import get_cpptype_from_pytype


def get_cur_dir():
    """Get parent path of this file"""
    return pathlib.Path(__file__).absolute().parent


class ProfilingParams:
    """Collect profiling parameters"""

    def __init__(self, number=1, repeat=1, min_repeat_ms=0):
        self.number = number
        self.repeat = repeat
        self.min_repeat_ms = min_repeat_ms

    def get_data(self, ):
        """Get data"""
        data = [self.number, self.repeat, self.min_repeat_ms]
        return data


def get_shape_args_list(device_shape, is_dyn_shape, fake_output_indices):
    """Get shape_args list"""
    shape_args_list = list()
    if not is_dyn_shape:
        for idx in range(len(device_shape)):
            shape_args_list.append(["pointer"])
        return shape_args_list

    for idx, data_shape in enumerate(device_shape):
        shape_list = list()
        if idx not in fake_output_indices:
            shape_list.append("remove")
            shape_list.append("pointer")
            shape_list.append(0)
            shape_list += list(data_shape)
            stride_list = [1] * len(data_shape)
            for i, _ in enumerate(data_shape[1:]):
                stride_list[-i -
                            2] = stride_list[-i - 1] * data_shape[-i - 1]
            shape_list += stride_list
        shape_args_list.append(shape_list)
    return shape_args_list


def gen_cuda_runtime_code(kernel_name,
                          input_for_mod,
                          output_indexes,
                          is_dyn_shape,
                          fake_output_indices,
                          path="./tmp_files/"):
    """Generate cuda runtime code"""
    template_src = ""
    akg_kernel_meta_path = os.path.join(path, "akg_kernel_meta")

    with open(os.path.join(str(get_cur_dir()), "mlir_cuda_runtime_template.txt"), 'r') as file:
        template_src = file.read()
    device_shape, symbol_map, support_info = get_device_shape(
        input_for_mod, kernel_name, is_dyn_shape)
    mapping_file = os.path.join(akg_kernel_meta_path, kernel_name + ".json")
    runtime_arg_file = os.path.join(
        akg_kernel_meta_path, kernel_name + "_runtime_arg.txt")
    dim = get_gpu_setting_by_input(symbol_map, mapping_file, support_info)
    dyn_tiling_args = dict()
    if is_dyn_shape:
        with open(runtime_arg_file, "r") as file:
            dyn_tiling_args = json.loads(file.read())

    rt_code_ptx_path = "\"" + akg_kernel_meta_path + "/" + kernel_name + ".ptx\""
    rt_code_kernel_name = "\"" + kernel_name + "_kernel\""

    shape_args_list = get_shape_args_list(device_shape, is_dyn_shape,
                                          fake_output_indices)

    rt_code_params_list = ""
    rt_code_mem_alloc = ""
    rt_code_mem_copy_htod = ""
    rt_code_set_grid_params = ""
    rt_code_set_block_params = ""
    rt_code_set_args_params = ""
    rt_code_mem_copy_dtoh = ""
    rt_code_free_d_mem = ""
    if is_dyn_shape:
        rt_code_init_memref_params = "CUdeviceptr dev_ptr_fake;\n"
    else:
        rt_code_init_memref_params = ""

    for idx, d in enumerate(input_for_mod):
        if idx in fake_output_indices:
            continue
        for j, param in enumerate(shape_args_list[idx]):
            if param == "remove":
                rt_code_set_args_params += "&dev_ptr_fake, "
            elif param == "pointer":
                rt_code_set_args_params += "&" + "dev_ptr_" + str(idx) + ", "
            else:
                param_name = "param_" + str(idx) + "_" + str(j)
                rt_code_init_memref_params += "size_t " + param_name + " = " + str(
                    param) + ";\n"
                rt_code_set_args_params += "&" + param_name + ", "

        dtype = get_cpptype_from_pytype(str(d.dtype))
        size = str(reduce(lambda x, y: x * y, d.shape))
        rt_code_params_list += dtype + "* " + "data_" + str(idx) + ", "
        rt_code_mem_alloc += "    CUdeviceptr " + "dev_ptr_" + str(idx) + ";\n"
        rt_code_mem_alloc += "    checkCudaDrvErrors(cuMemAlloc(&" + "dev_ptr_" + str(
            idx) + ", " + size + " * sizeof(" + dtype + "))); \n"
        rt_code_mem_copy_htod += "    checkCudaDrvErrors(cuMemcpyHtoD(" + "dev_ptr_" + str(
            idx) + ", " + "data_" + str(
                idx) + ", " + size + " * sizeof(" + dtype + ")));\n"
        if (idx in output_indexes) or ((idx - len(input_for_mod))
                                       in output_indexes):
            rt_code_mem_copy_dtoh += "    checkCudaDrvErrors(cuMemcpyDtoH(" + "data_" + str(
                idx) + ", " + "dev_ptr_" + str(
                    idx) + ", " + size + " * sizeof(" + dtype + ")));\n"
        rt_code_free_d_mem += "checkCudaDrvErrors(cuMemFree(" + \
            "dev_ptr_" + str(idx) + "));\n"

    if is_dyn_shape:
        keys = [int(item) for item in dyn_tiling_args.keys()]
        for i, k in enumerate(sorted(keys)):
            arg = dyn_tiling_args[str(k)]
            param_name = "dyn_tile_" + str(i)
            rt_code_init_memref_params += "int64_t " + param_name + " = " + str(
                arg) + ";\n"
            rt_code_set_args_params += "&" + param_name + ", "

    rt_code_params_list = rt_code_params_list.strip(" ").strip(",")
    rt_code_set_grid_params = "    const int gx = " + str(
        dim["blockIdx.x"]) + ", gy = " + str(
            dim["blockIdx.y"]) + ", gz = " + str(dim["blockIdx.z"]) + ";"
    rt_code_set_block_params = "    const int bx = " + str(
        dim["threadIdx.x"]) + ", by = " + str(
            dim["threadIdx.y"]) + ", bz = " + str(dim["threadIdx.z"]) + ";"
    rt_code_set_args_params = rt_code_set_args_params.strip(" ").strip(",")

    template_src = template_src.replace("rt_code_ptx_path", rt_code_ptx_path)
    template_src = template_src.replace(
        "rt_code_kernel_name", rt_code_kernel_name)
    template_src = template_src.replace(
        "rt_code_params_list", rt_code_params_list)
    template_src = template_src.replace("rt_code_mem_alloc", rt_code_mem_alloc)
    template_src = template_src.replace(
        "rt_code_mem_copy_htod", rt_code_mem_copy_htod)
    template_src = template_src.replace(
        "rt_code_set_grid_params", rt_code_set_grid_params)
    template_src = template_src.replace(
        "rt_code_set_block_params", rt_code_set_block_params)
    template_src = template_src.replace(
        "rt_code_set_args_params", rt_code_set_args_params)
    template_src = template_src.replace(
        "rt_code_mem_copy_dtoh", rt_code_mem_copy_dtoh)
    template_src = template_src.replace(
        "rt_code_free_d_mem", rt_code_free_d_mem)
    template_src = template_src.replace("rt_code_init_memref_params",
                                        rt_code_init_memref_params)
    with open(os.path.join(path, "tmp_files", "gen_func_" + kernel_name + ".cu"), "wt") as file:
        file.writelines(template_src)
