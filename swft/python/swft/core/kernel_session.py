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
import re
from functools import reduce
from swft.utils.util import TYPE_SIZE, SCALARTYPE_CTYPE, is_scalar, is_tensor
from .header import *

prefix_dir = "."


def extract_parameters(func_call):
    match = re.search(r'\((.*?)\)', func_call)
    if not match:
        return []
    params_str = match.group(1)
    params = [p.strip() for p in params_str.split(',') if p.strip()]
    return params


def gen_size_str(tensor_name, local_vars):
    lines = []
    for name in tensor_name:
        tensor = local_vars[name]
        if is_tensor(tensor):
            tensor_size = reduce(lambda x, y: x*y, tensor.shape)
            tensor_size = (tensor_size * TYPE_SIZE[tensor.dtype] + 7) // 8
            lines.append(f'    size_t {name}_size = {str(tensor_size)};')
        elif is_scalar(tensor):
            lines.append(
                f'    size_t {name}_size = {TYPE_SIZE[tensor.dtype] // 8};')
    return lines


def gen_allocate_addr(input_names, var_names, kernel_name, local_vars):
    lines = []
    for name in var_names:
        if name in local_vars.keys() and is_scalar(local_vars[name]):
            curr_scalar = local_vars[name]
            lines.append(
                f'    {SCALARTYPE_CTYPE[curr_scalar.dtype]} *{name}_host;')
            lines.append(
                f'    aclrtMallocHost((void **)(&{name}_host), {TYPE_SIZE[curr_scalar.dtype] // 8});')
            if name in input_names:
                lines.append(
                    f'    ReadFile("{prefix_dir}/{kernel_name}/input/{name}.bin", {name}_size, {name}_host, {name}_size);')
            continue
        lines.append(f'    uint8_t *{name}_host;')
        lines.append(f'    uint8_t *{name}_device;')
        lines.append(
            f'    aclrtMallocHost((void **)(&{name}_host), {name}_size);')
        lines.append(
            f'    aclrtMalloc((void **)(&{name}_device), {name}_size, ACL_MEM_MALLOC_HUGE_FIRST);')
        if name in input_names:
            lines.append(
                f'    ReadFile("{prefix_dir}/{kernel_name}/input/{name}.bin", {name}_size, {name}_host, {name}_size);')
            lines.append(
                f'    aclrtMemcpy({name}_device, {name}_size, {name}_host, {name}_size, ACL_MEMCPY_HOST_TO_DEVICE);')
    return lines


def gen_write_to_file(output_names, kernel_name):
    lines = ['    aclrtSynchronizeStream(stream);']
    for name in output_names:
        lines.append(
            f'    aclrtMemcpy({name}_host, {name}_size, {name}_device, {name}_size, ACL_MEMCPY_DEVICE_TO_HOST);')
        lines.append(
            f'    WriteFile("{prefix_dir}/{kernel_name}/output/{name}_actual.bin", {name}_host, {name}_size);')
    lines.append('    aclrtDestroyStream(stream);')
    lines.append('    aclrtResetDevice(deviceId);')
    lines.append('    aclFinalize();')
    lines.append('    return 0;')
    lines.append('}')
    return lines


def gen_func_call(func_sig, var_names, block_dim, local_vars):
    func_call = ""
    match = re.search(r'\bvoid\s+(\w+)\s*\(', func_sig)
    func_name = match.group(1)
    func_call += (func_name + f'({block_dim}, nullptr, stream,')
    for name in var_names:
        if name in local_vars.keys() and is_tensor(local_vars[name]):
            func_call += f'{name}_device, '
        elif name in local_vars.keys() and is_scalar(local_vars[name]):
            func_call += f'*{name}_host, '
    func_call = func_call[:-2]
    func_call += ");"
    return func_call


def gen_profiling(iter_time, func_sig, var_names, block_dim, local_vars):
    func_call = gen_func_call(func_sig, var_names, block_dim, local_vars)
    lines = ['    auto start = std::chrono::high_resolution_clock::now();']
    lines.append('    for (int i = 0; i < {}; ++i) '.format(iter_time) + "{")
    lines.append('        ' + func_call)
    lines.append('}')
    lines.append('    aclrtSynchronizeStream(stream);')
    lines.append('    auto end = std::chrono::high_resolution_clock::now();')
    lines.append('    std::chrono::duration<double> duration = end - start;')
    lines.append(
        '    std::chrono::nanoseconds totalDurationInUs = std::chrono::duration_cast<std::chrono::nanoseconds>(duration);')
    lines.append(
        '    auto avgDuration = totalDurationInUs.count() / {};'.format(iter_time))
    lines.append(
        '    std::cout << "program avg duration " << (double)avgDuration / 1000.0 << " us" << std::endl;')
    return lines


def exec_kernel(kernel_name, local_vars, prefix_path=None, inputs=[], outputs=[], block_dim=8, device_id=0, profiling=0):
    global prefix_dir
    if prefix_path is not None:
        prefix_dir = prefix_path
    os.system(f"mkdir -p {prefix_dir}/{kernel_name}")
    cann_path = os.getenv("ASCEND_HOME_PATH")
    if not cann_path:
        raise ValueError("ASCEND_HOME_PATH not set!")
    compile_kernel = f'{cann_path}/toolkit/tools/ccec_compiler/bin/ccec -xcce --cce-aicore-arch=dav-m200 -mllvm -cce-aicore-function-stack-size=16000 -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm --cce-aicore-jump-expand=true  -fPIC -pthread -o {prefix_dir}/{kernel_name}/{kernel_name}.o -c {prefix_dir}/{kernel_name}/{kernel_name}.cce'
    compile_main = f'c++ -I{cann_path}/acllib/include -fvisibility-inlines-hidden -std=c++17 -fmessage-length=0 -ftree-vectorize -fPIC -fstack-protector-strong -fno-plt -O3 -pipe -isystem -std=gnu++17 -fPIC -O2 -std=c++17 -MD -MT {prefix_dir}/{kernel_name}/main.cpp.o -MF {prefix_dir}/{kernel_name}/main.cpp.o.d -o {prefix_dir}/{kernel_name}/main.cpp.o -c {prefix_dir}/{kernel_name}/main.cpp'
    link_opt = f'{cann_path}/toolkit/tools/ccec_compiler/bin/ccec --cce-fatobj-link  -Wl,-O2 -Wl,--sort-common -Wl,--as-needed -Wl,-z,relro -Wl,-z,now -Wl,--allow-shlib-undefined {prefix_dir}/{kernel_name}/main.cpp.o -o {prefix_dir}/{kernel_name}/{kernel_name}_kernel -L{cann_path}/lib64  -L{cann_path}/tools/simulator/Ascend310P3/lib {prefix_dir}/{kernel_name}/{kernel_name}.o -lstdc++ -lm -lgcc_s -lgcc -lc -lgcc_s -lgcc  -lstdc++ -lruntime -lprofapi -lascendcl'
    fname = f'{prefix_dir}/{kernel_name}/{kernel_name}.cce'
    with open(fname, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        var_names = extract_parameters(lines[-2])
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].startswith("extern"):
                func_sig = lines[i][:-2] + ";"
    main_cpp_lines = []
    main_cpp_lines.append(data_utils)
    main_cpp_lines.append(func_sig)
    main_cpp_lines.append(main_start)
    main_cpp_lines.extend(gen_size_str(var_names, local_vars))
    main_cpp_lines.extend(gen_allocate_addr(
        inputs, var_names, kernel_name, local_vars))
    main_cpp_lines.append(
        '    ' + gen_func_call(func_sig, var_names, block_dim, local_vars))
    if (profiling > 0):
        main_cpp_lines.extend(gen_profiling(
            profiling, func_sig, var_names, block_dim, local_vars))
    main_cpp_lines.extend(gen_write_to_file(outputs, kernel_name))
    with open(f'{prefix_dir}/{kernel_name}/main.cpp', "w") as f:
        f.write("\n".join(main_cpp_lines))
    os.system(compile_kernel)
    os.system(compile_main)
    os.system(link_opt)
    os.system(f'{prefix_dir}/{kernel_name}/{kernel_name}_kernel {device_id}')
