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
from functools import reduce
from .dynamic_utils import get_gpu_setting_by_input, get_device_shape
from .composite_op_helper import get_cpptype_from_pytype
from .code_template import CUDA_RUNTIME_TEMPLATE


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
    if not is_dyn_shape:
        return [["pointer"]] * len(device_shape)

    shape_args_list = []
    for idx, data_shape in enumerate(device_shape):
        shape_list = []
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


def _build_input_code(input_for_mod, output_indexes, fake_output_indices, shape_args_list):
    """Build code fragments for each input tensor."""
    code = {
        "params_list": [],
        "mem_alloc": [],
        "mem_copy_htod": [],
        "mem_copy_dtoh": [],
        "free_d_mem": [],
        "set_args_params": [],
        "init_memref_params": [],
    }

    for idx, d in enumerate(input_for_mod):
        if idx in fake_output_indices:
            continue

        for j, param in enumerate(shape_args_list[idx]):
            if param == "remove":
                code["set_args_params"].append("&dev_ptr_fake")
            elif param == "pointer":
                code["set_args_params"].append(f"&dev_ptr_{idx}")
            else:
                param_name = f"param_{idx}_{j}"
                code["init_memref_params"].append(f"size_t {param_name} = {param};")
                code["set_args_params"].append(f"&{param_name}")

        dtype = get_cpptype_from_pytype(str(d.dtype))
        size = reduce(lambda x, y: x * y, d.shape)

        code["params_list"].append(f"{dtype}* data_{idx}")
        code["mem_alloc"].extend([
            f"    CUdeviceptr dev_ptr_{idx};",
            f"    checkCudaDrvErrors(cuMemAlloc(&dev_ptr_{idx}, {size} * sizeof({dtype})));"
        ])
        code["mem_copy_htod"].append(
            f"    checkCudaDrvErrors(cuMemcpyHtoD(dev_ptr_{idx}, data_{idx}, {size} * sizeof({dtype})));"
        )

        if idx in output_indexes or (idx - len(input_for_mod)) in output_indexes:
            code["mem_copy_dtoh"].append(
                f"    checkCudaDrvErrors(cuMemcpyDtoH(data_{idx}, dev_ptr_{idx}, {size} * sizeof({dtype})));"
            )

        code["free_d_mem"].append(f"    checkCudaDrvErrors(cuMemFree(dev_ptr_{idx}));")

    return code


def _build_dyn_tiling_code(dyn_tiling_args, set_args_params, init_memref_params):
    """Build code fragments for dynamic tiling args."""
    if not dyn_tiling_args:
        return
    keys = sorted(int(k) for k in dyn_tiling_args.keys())
    for i, k in enumerate(keys):
        arg = dyn_tiling_args[str(k)]
        param_name = f"dyn_tile_{i}"
        init_memref_params.append(f"int64_t {param_name} = {arg};")
        set_args_params.append(f"&{param_name}")


def _build_grid_block_params(dim):
    """Build grid and block parameter strings."""
    set_grid_params = f"    const int gx = {dim['blockIdx.x']}, gy = {dim['blockIdx.y']}, gz = {dim['blockIdx.z']};"
    set_block_params = f"    const int bx = {dim['threadIdx.x']}, by = {dim['threadIdx.y']}, bz = {dim['threadIdx.z']};"
    return set_grid_params, set_block_params


def _load_dyn_tiling_args(runtime_arg_file, is_dyn_shape):
    """Load dynamic tiling args from the runtime arg file."""
    if not is_dyn_shape:
        return {}
    with open(os.path.realpath(runtime_arg_file), "r", encoding='utf-8') as file:
        return json.loads(file.read())


def _apply_replacements(template_src, replacements):
    """Apply replacement mappings to the template source."""
    for old, new in replacements.items():
        template_src = template_src.replace(old, new)
    return template_src


def _write_runtime_code(output_file, template_src):
    """Write the generated runtime code to file."""
    with open(os.path.realpath(output_file), "wt", encoding='utf-8') as file:
        file.writelines(template_src)


def gen_cuda_runtime_code(kernel_name,
                          input_for_mod,
                          output_indexes,
                          is_dyn_shape,
                          fake_output_indices,
                          path=os.path.expanduser("~/.akg")):
    """Generate cuda runtime code"""
    device_info = get_device_shape(input_for_mod, kernel_name, is_dyn_shape)
    dim = get_gpu_setting_by_input(
        device_info[1], os.path.join(path, f"{kernel_name}.json"), device_info[2])

    dyn_tiling_args = _load_dyn_tiling_args(
        os.path.join(path, f"{kernel_name}_runtime_arg.txt"), is_dyn_shape)

    shape_args_list = get_shape_args_list(device_info[0], is_dyn_shape, fake_output_indices)

    input_code = _build_input_code(input_for_mod, output_indexes, fake_output_indices, shape_args_list)

    if is_dyn_shape:
        input_code["init_memref_params"].append("CUdeviceptr dev_ptr_fake;")

    _build_dyn_tiling_code(dyn_tiling_args, input_code["set_args_params"], input_code["init_memref_params"])

    grid_block = _build_grid_block_params(dim)

    template_src = _apply_replacements(CUDA_RUNTIME_TEMPLATE, {
        "rt_code_ptx_path": f'"{path}/{kernel_name}.ptx"',
        "rt_code_kernel_name": f'"{kernel_name}_kernel"',
        "rt_code_params_list": ", ".join(input_code["params_list"]),
        "rt_code_mem_alloc": "\n".join(input_code["mem_alloc"]),
        "rt_code_mem_copy_htod": "\n".join(input_code["mem_copy_htod"]),
        "rt_code_set_grid_params": grid_block[0],
        "rt_code_set_block_params": grid_block[1],
        "rt_code_set_args_params": ", ".join(input_code["set_args_params"]),
        "rt_code_mem_copy_dtoh": "\n".join(input_code["mem_copy_dtoh"]),
        "rt_code_free_d_mem": "\n".join(input_code["free_d_mem"]),
        "rt_code_init_memref_params": "\n".join(input_code["init_memref_params"]),
    })

    output_file = os.path.join(path, "tmp_files", f"gen_func_{kernel_name}.cu")
    _write_runtime_code(output_file, template_src)
