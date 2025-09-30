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
from swft.utils import gen_func_sig, gen_size_str, gen_allocate_addr, gen_func_call, gen_profiling, gen_write_to_file, data_utils, main_start


def exec_kernel(kernel_name, local_vars, prefix_path=".", inputs=[], outputs=[], block_dim=8, device_id=0, profiling=0):
    os.system(f"mkdir -p {prefix_path}/{kernel_name}")
    cann_path = os.getenv("ASCEND_HOME_PATH")
    if not cann_path:
        raise ValueError("ASCEND_HOME_PATH not set!")
    compile_kernel = f'{cann_path}/toolkit/tools/ccec_compiler/bin/ccec -xcce -O3 \
        -I{cann_path}/compiler/tikcpp/tikcfw/ \
        -I{cann_path}/aarch64-linux/ascendc/include/basic_api/impl/ \
        -I{cann_path}/aarch64-linux/ascendc/include/basic_api/interface/ \
        --cce-aicore-arch=dav-m200 -mllvm -cce-aicore-function-stack-size=16000 -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm --cce-aicore-jump-expand=true -std=c++20 -fPIC -pthread -o {prefix_path}/{kernel_name}/{kernel_name}.o -c {prefix_path}/{kernel_name}/{kernel_name}.cce'
    compile_main = f'c++ -I{cann_path}/acllib/include -fvisibility-inlines-hidden -std=c++17 -fmessage-length=0 -ftree-vectorize -fPIC -fstack-protector-strong -fno-plt -O3 -pipe -isystem -std=gnu++17 -fPIC -O2 -std=c++17 -MD -MT {prefix_path}/{kernel_name}/main.cpp.o -MF {prefix_path}/{kernel_name}/main.cpp.o.d -o {prefix_path}/{kernel_name}/main.cpp.o -c {prefix_path}/{kernel_name}/main.cpp'
    link_opt = f'{cann_path}/toolkit/tools/ccec_compiler/bin/ccec --cce-fatobj-link  -Wl,-O2 -Wl,--sort-common -Wl,--as-needed -Wl,-z,relro -Wl,-z,now -Wl,--allow-shlib-undefined ./{prefix_path}/{kernel_name}/main.cpp.o -o ./{prefix_path}/{kernel_name}/{kernel_name}_kernel -L{cann_path}/lib64  -L{cann_path}/tools/simulator/Ascend310P3/lib ./{prefix_path}/{kernel_name}/{kernel_name}.o -lstdc++ -lm -lgcc_s -lgcc -lc -lgcc_s -lgcc  -lstdc++ -lruntime -lprofapi -lascendcl'
    fname = f'./{prefix_path}/{kernel_name}/{kernel_name}.cce'
    func_sig, var_names = gen_func_sig(fname)
    main_cpp_lines = []
    main_cpp_lines.append(data_utils)
    main_cpp_lines.append(func_sig)
    main_cpp_lines.append(main_start)
    main_cpp_lines.extend(gen_size_str(var_names, local_vars))
    main_cpp_lines.extend(gen_allocate_addr(
        inputs, var_names, kernel_name, local_vars, prefix_path))
    main_cpp_lines.append(
        '    ' + gen_func_call(func_sig, var_names, block_dim, local_vars))
    if (profiling > 0):
        main_cpp_lines.extend(gen_profiling(
            profiling, func_sig, var_names, block_dim, local_vars))
    main_cpp_lines.extend(gen_write_to_file(outputs, kernel_name, prefix_path))
    with open(f'{prefix_path}/{kernel_name}/main.cpp', "w") as f:
        f.write("\n".join(main_cpp_lines))
    os.system(compile_kernel)
    os.system(compile_main)
    os.system(link_opt)
    os.system(
        f'./{prefix_path}/{kernel_name}/{kernel_name}_kernel {device_id}')

