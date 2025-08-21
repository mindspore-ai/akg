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

import enum
import re
from functools import reduce
from swft.utils.util import TYPE_SIZE, SCALARTYPE_CTYPE, is_scalar, is_tensor

data_utils = """#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <arm_neon.h>
#include <cassert>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <chrono>
#include "acl/acl.h"


bool ReadFile(const std::string &filePath, size_t &fileSize, void *buffer, size_t bufferSize)
{
    struct stat sBuf;
    int fileStatus = stat(filePath.data(), &sBuf);
    if (fileStatus == -1) {
        printf("failed to get file, file path: %s", filePath.c_str());
        std::exit(0);
        return false;
    }
    if (S_ISREG(sBuf.st_mode) == 0) {
        printf("%s is not a file, please enter a file", filePath.c_str());
        std::exit(0);
        return false;
    }

    std::ifstream file;
    file.open(filePath, std::ios::binary);
    if (!file.is_open()) {
        printf("Open file failed. path = %s", filePath.c_str());
        std::exit(0);
        return false;
    }

    std::filebuf *buf = file.rdbuf();
    size_t size = buf->pubseekoff(0, std::ios::end, std::ios::in);
    if (size == 0) {
        printf("file size is 0");
        file.close();
        std::exit(0);
        return false;
    }
    if (size > bufferSize) {
        printf("file size is larger than buffer size");
        file.close();
        std::exit(0);
        return false;
    }
    buf->pubseekpos(0, std::ios::in);
    buf->sgetn(static_cast<char *>(buffer), size);
    fileSize = size;
    file.close();
    return true;
}

bool WriteFile(const std::string &filePath, const void *buffer, size_t size)
{
    if (buffer == nullptr) {
        printf("Write file failed. buffer is nullptr");
        std::exit(0);
        return false;
    }

    int fd = open(filePath.c_str(), O_RDWR | O_CREAT | O_TRUNC, S_IRUSR | S_IWRITE);
    if (fd < 0) {
        printf("Open file failed. path = %s", filePath.c_str());
        std::exit(0);
        return false;
    }

    auto writeSize = write(fd, buffer, size);
    (void)close(fd);
    if (writeSize != size) {
        printf("Write file Failed.");
        std::exit(0);
        return false;
    }

    return true;
}"""

main_start = """int main(int argc, char *argv[]) {
    aclInit(nullptr);
    uint32_t devCount = 0;
    aclrtGetDeviceCount(&devCount);
    int deviceId = 0;
    if (argc > 1) {
        deviceId = std::stoi(argv[1]);
    }
    aclrtSetDevice(deviceId);
    aclrtContext context;
    aclrtCreateContext(&context, deviceId);
    aclrtStream stream = nullptr;
    aclrtCreateStreamWithConfig(&stream, 0, ACL_STREAM_FAST_LAUNCH | ACL_STREAM_FAST_SYNC);
"""

op_bind_start = """#include <pybind11/pybind11.h>
#include "acl/acl.h"
namespace py = pybind11;
"""

module_registry_class_def = '''
using ModuleRegisterFunction = std::function<void(pybind11::module_ &)>;
class ModuleRegistry {
public:
  // Get the singleton instance
  static ModuleRegistry &Instance() {
    static ModuleRegistry instance;
    return instance;
  }

  // Register a module function
  void Register(ModuleRegisterFunction func, bool pynative_node = true) {
    auto &target =
        pynative_node ? pynative_reg_functions_ : graph_reg_functions_;
    target.emplace_back(std::move(func));
  }

  // Call all registered module functions
  void RegisterAll(pybind11::module_ &m) {
    for (const auto &func : pynative_reg_functions_) {
      func(m);
    }
    for (const auto &func : graph_reg_functions_) {
      func(m);
    }
  }
private:
  ModuleRegistry() = default;
  ~ModuleRegistry() = default;

  // Disable copy and assignment
  ModuleRegistry(const ModuleRegistry &) = delete;
  ModuleRegistry &operator=(const ModuleRegistry &) = delete;

  // Store all registered functions
  std::vector<ModuleRegisterFunction> pynative_reg_functions_;
  std::vector<ModuleRegisterFunction> graph_reg_functions_;
};

#define REG_GRAPH_MODE_OP(op, OpFuncImplClass, KernelClass)                    \\
  MS_CUSTOM_OPS_REGISTER(op, OpFuncImplClass, KernelClass);                    \\
  static void op##_func() {}                                                   \\
  static void op##_register(pybind11::module_ &m) {                            \\
    if (!pybind11::hasattr(m, #op)) {                                          \\
      m.def(#op, &op##_func);                                                  \\
    }                                                                          \\
  }                                                                            \\
  struct op##_registrar {                                                      \\
    op##_registrar() {                                                         \\
      ModuleRegistry::Instance().Register(op##_register, false);               \\
    }                                                                          \\
  };                                                                           \\
  static op##_registrar registrar_instance
'''

ms_run_src_start = '''
#include "ms_extension.h"
#include "plugin/device/ascend/kernel/custom/custom_kernel_factory.h"
#include "common/kernel.h"
#include <iostream>
#include <dlfcn.h>
#include <pybind11/pybind11.h>
#include <vector>
using namespace mindspore;
using namespace mindspore::kernel;
using namespace mindspore::device::ascend;
using namespace mindspore::ops;
typedef void (*kernel_func_t)(uint32_t, void *, void *, ...);
static std::unordered_map<std::string, kernel_func_t> kernel_cache;
'''

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
            tensor_size = (int(tensor_size.value) * int(TYPE_SIZE[tensor.dtype]) + 7) // 8
            lines.append(f'    size_t {name}_size = {tensor_size};')
        elif is_scalar(tensor):
            lines.append(
                f'    size_t {name}_size = {TYPE_SIZE[tensor.dtype] // 8};')
    return lines


def gen_allocate_addr(input_names, var_names, kernel_name, local_vars, prefix_dir="."):

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
                    f'    ReadFile("./{prefix_dir}/{kernel_name}/input/{name}.bin", {name}_size, {name}_host, {name}_size);')
            continue
        lines.append(f'    uint8_t *{name}_host;')
        lines.append(f'    uint8_t *{name}_device;')
        lines.append(
            f'    aclrtMallocHost((void **)(&{name}_host), {name}_size);')
        lines.append(
            f'    aclrtMalloc((void **)(&{name}_device), {name}_size, ACL_MEM_MALLOC_HUGE_FIRST);')
        if name in input_names:
            lines.append(
                f'    ReadFile("./{prefix_dir}/{kernel_name}/input/{name}.bin", {name}_size, {name}_host, {name}_size);')
            lines.append(
                f'    aclrtMemcpy({name}_device, {name}_size, {name}_host, {name}_size, ACL_MEMCPY_HOST_TO_DEVICE);')
    return lines


def gen_write_to_file(output_names, kernel_name, prefix_dir="."):

    lines = ['    aclrtSynchronizeStream(stream);']
    for name in output_names:
        lines.append(
            f'    aclrtMemcpy({name}_host, {name}_size, {name}_device, {name}_size, ACL_MEMCPY_DEVICE_TO_HOST);')
        lines.append(
            f'    WriteFile("./{prefix_dir}/{kernel_name}/output/{name}_actual.bin", {name}_host, {name}_size);')
    lines.append('    aclrtDestroyStream(stream);')
    lines.append('    aclrtResetDevice(deviceId);')
    lines.append('    aclFinalize();')
    lines.append('    return 0;')
    lines.append('}')
    return lines


def gen_func_sig(ccef_name):
    func_sig = ""
    with open(ccef_name, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        var_names = extract_parameters(lines[-2])
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].startswith("extern"):
                func_sig = lines[i][:-2] + ";"
    return func_sig, var_names


def gen_pybind_define(kernal_name, args):
    pybind_define = []
    pybind_call = "void "
    pybind_call += f"{kernal_name}_launch(uint32_t blockDim, "
    kernel_call = f"    {kernal_name}(blockDim, nullptr, stream, "
    for arg in args:
        if is_tensor(arg):
            pybind_call += f"void *{arg.__name__}, "
            kernel_call += f"static_cast<uint8_t *>({arg.__name__}), "
        elif is_scalar(arg):
            if arg.dtype in ["INT32", "INT16"]:
                pybind_call += f"int {arg.__name__}, "
            elif arg.dtype in ["FP32", "FP16"]:
                pybind_call += f"float {arg.__name__}, "
            elif arg.dtype == "BOOL":
                pybind_call += f"bool {arg.__name__}, "
            kernel_call += f"{arg.__name__}, "
        elif isinstance(arg, (int, bool, float)):
            if isinstance(arg, int):
                pybind_call += f"int {arg.__name__}, "
            elif isinstance(arg, bool):
                pybind_call += f"bool {arg.__name__}, "
            elif isinstance(arg, float):
                pybind_call += f"float {arg.__name__}, "
            kernel_call += f"{arg.__name__}, "
        else:
            raise TypeError(f"unparsed argument:{arg.__name__}")
    pybind_call += " void *stream) {"
    kernel_call = kernel_call[:-2] + ");"
    pybind_define.append(pybind_call)
    pybind_define.append(kernel_call)
    pybind_define.append("}")
    return pybind_define


def gen_pybind_binding(kernal_name):
    pybind_binding = []
    pybind_binding.append(f"PYBIND11_MODULE({kernal_name}, m)")
    pybind_binding.append("{")
    pybind_binding.append(
        f"    m.def(\"{kernal_name}\", &{kernal_name}_launch);")
    pybind_binding.append("}")
    return pybind_binding


def gen_opbind_cpp(ccef_name, kernel_name, args):
    opbind_cpp = []
    opbind_cpp.append(op_bind_start)
    func_sig, _ = gen_func_sig(ccef_name)
    opbind_cpp.append(func_sig)
    opbind_cpp.extend(gen_pybind_define(kernel_name, args))
    opbind_cpp.extend(gen_pybind_binding(kernel_name))
    return opbind_cpp


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
    lines.append('    }')
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


class ArgType(enum.Enum):
    TENSOR = 1
    INT = 2
    FLOAT = 3
    BOOL = 4


def get_load_kernel_src(lib_path, kernel_name):
    lines = ['    int32_t block_dim = 8;',
             '    void *l2ctrl = nullptr;',
             '    kernel_func_t func;',
             f'    std::string op_name{{"{kernel_name}"}};',
             '    auto iter = kernel_cache.find(op_name);',
             '    if (iter == kernel_cache.end()) {',
             f'        std::string lib_name = "{lib_path}";',
             '        auto lib_handle = dlopen(lib_name.c_str(), RTLD_LAZY);',
             '        if (lib_handle == nullptr) {',
             '            std::cout << "[ERROR] failed to dlopen " << lib_name << std::endl;',
             '        }',
             '        auto kernel_handle = dlsym(lib_handle, op_name.c_str());',
             '        if (kernel_handle == nullptr) {',
             '            std::cout << "[ERROR] failed to dlsym " << op_name << std::endl;',
             '        }',
             '        func = reinterpret_cast<kernel_func_t>(kernel_handle);',
             '    } else {',
             '        func = iter->second;',
             '    }']
    return lines


def get_launch_src_for_kernel_mod(args_order):
    lines = []
    args = []
    for i, arg_type in enumerate(args_order):
        if arg_type == ArgType.TENSOR:
            tensor_arg = f"tensor_arg_{i}"
            lines.append(f'        void *{tensor_arg} = inputs[{i}]->device_ptr();')
            args.append(tensor_arg)
        elif arg_type == ArgType.INT:
            int_arg = f"int_arg_{i}"
            lines.append(f'        int {int_arg} = static_cast<int>(inputs.at({i})->GetValue<int64_t>().value());')
            args.append(int_arg)
        elif arg_type == ArgType.FLOAT:
            float_arg = f"float_arg_{i}"
            lines.append(f'        float {float_arg} = static_cast<float>(inputs.at({i})->GetValue<float>().value());')
            args.append(float_arg)
        elif arg_type == ArgType.BOOL:
            bool_arg = f"bool_arg_{i}"
            lines.append(f'        bool {bool_arg} = static_cast<bool>(inputs.at({i})->GetValue<bool>().value());')
            args.append(bool_arg)
    launch_line = '        func(block_dim, l2ctrl, stream_ptr'
    for arg in args:
        launch_line += f', {arg}'
    launch_line += ');'
    lines.append(launch_line)
    return lines


def get_launch_line_for_pyboost_runner(args_order):
    launch_line = '        func(block_dim, l2ctrl, stream()'
    tensor_idx = 0
    int_idx = 0
    float_idx = 0
    bool_idx = 0
    for arg_type in args_order:
        if arg_type == ArgType.TENSOR:
            launch_line += f', tensor_args[{tensor_idx}]'
            tensor_idx += 1
        elif arg_type == ArgType.INT:
            launch_line += f', int_args_[{int_idx}]'
            int_idx += 1
        elif arg_type == ArgType.FLOAT:
            launch_line += f', float_args_[{float_idx}]'
            float_idx += 1
        elif arg_type == ArgType.BOOL:
            launch_line += f', bool_args_[{bool_idx}]'
            bool_idx += 1
    launch_line += ');'
    return launch_line


def gen_ms_run_src(src_file, lib_path, kernel_name, args_order):
    lines = [ms_run_src_start]

    op_func_class_name = f'{kernel_name}_OpFuncImpl'
    lines.append('namespace ms_custom_ops {')
    lines.append(f'class OPS_API {op_func_class_name} : public OpFuncImpl')
    lines.append('{')
    lines.append('public:')
    lines.append('  ShapeArray InferShape(const PrimitivePtr &primitive,')
    lines.append('                        const InferInfoPtrList &input_infos) const override {')
    lines.append('    return {input_infos[0]->GetShape()};')
    lines.append('  }')
    lines.append('  std::vector<TypeId>')
    lines.append('  InferType(const PrimitivePtr &primitive,')
    lines.append('            const InferInfoPtrList &input_infos) const override {')
    lines.append('    return {input_infos[0]->GetType()};')
    lines.append('  }')
    lines.append('')
    lines.append('  bool GeneralInferRegistered() const override { return true; }')
    lines.append('};')

    kernel_mod_class_name = f'{kernel_name}_KernelMod'
    load_kernel_src = get_load_kernel_src(lib_path, kernel_name)
    lines.append(f'class {kernel_mod_class_name} : public KernelMod')
    lines.append('{')
    lines.append('public:')
    lines.append('    bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override {')
    lines.append('        return true;')
    lines.append('    }')
    lines.append('    int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override {')
    lines.append('        return KernelMod::Resize(inputs, outputs);')
    lines.append('    }')
    lines.append('    bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,')
    lines.append('                const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {')
    lines.extend(load_kernel_src)
    launch_src_for_kernel_mod = get_launch_src_for_kernel_mod(args_order)
    lines.extend(launch_src_for_kernel_mod)
    lines.append('        return true;')
    lines.append('    }')
    lines.append('    std::vector<KernelAttr> GetOpSupport() override {')
    lines.append('      MS_LOG(EXCEPTION) << "This interface is not support in swft kernel.";')
    lines.append('    }')
    lines.append('    void set_fullname(const std::string &fullname) override {')
    lines.append('        fullname_ = fullname;')
    lines.append('    }')
    lines.append('')
    lines.append('private:')
    lines.append('  std::string fullname_;')
    lines.append('};')
    lines.append('} // namespace ms_custom_ops')
    lines.append(module_registry_class_def)
    lines.append(f'REG_GRAPH_MODE_OP({kernel_name}, ms_custom_ops::{op_func_class_name},')
    lines.append(f'                  ms_custom_ops::{kernel_mod_class_name});')

    pyboost_runner_class_name = f'{kernel_name}_PyboostRunner'
    lines.append(f'class {pyboost_runner_class_name} : public ms::pynative::PyboostRunner')
    lines.append('{')
    lines.append('public:')
    lines.append('  using PyboostRunner::PyboostRunner;')
    lines.append('    void LaunchKernel() override {')
    lines.append('        std::vector<uint8_t *> tensor_args;')
    lines.append('        for (auto input : inputs()) {')
    lines.append('            tensor_args.push_back(static_cast<uint8_t *>(input.GetDataPtr()));')
    lines.append('        }')
    lines.extend(load_kernel_src)
    launch_line_for_pyboost_runner = get_launch_line_for_pyboost_runner(args_order)
    lines.append(launch_line_for_pyboost_runner)

    lines.append('    }')
    lines.append('    static void Eval(std::vector<ms::Tensor> tensor_args, std::vector<int> int_args,')
    lines.append('                     std::vector<float> float_args, std::vector<bool> bool_args)')
    lines.append('    {')
    lines.append(f'        auto runner = std::make_shared<{pyboost_runner_class_name}>("{kernel_name}");')
    lines.append('        runner->int_args_ = int_args;')
    lines.append('        runner->float_args_ = float_args;')
    lines.append('        runner->bool_args_ = bool_args;')
    lines.append('        runner->Run(tensor_args, {});')
    lines.append('    }')
    lines.append('private:')
    lines.append('    std::vector<int> int_args_;')
    lines.append('    std::vector<float> float_args_;')
    lines.append('    std::vector<bool> bool_args_;')
    lines.append('};')
    apiName = f'{kernel_name}_runner'
    lines.append(f'auto {apiName}(std::vector<ms::Tensor> tensor_args, std::vector<int> int_args,')
    lines.append('    std::vector<float> float_args, std::vector<bool> bool_args)')
    lines.append('{')
    lines.append('    return ms::pynative::PyboostRunner::Call<0>(')
    lines.append(f'        {pyboost_runner_class_name}::Eval, tensor_args, int_args, float_args, bool_args);')
    lines.append('}')
    lines.append('PYBIND11_MODULE(MS_EXTENSION_NAME, m)')
    lines.append('{')
    lines.append(f'    m.def("{kernel_name}", &{apiName});')
    lines.append(f'    ModuleRegistry::Instance().RegisterAll(m);')
    lines.append('}')
    src = '\n'.join(lines)
    with open(src_file, "w") as f:
        f.write(src)


def gen_ms_run_op_yaml(kernel_name, args_order, op_yaml_file):
    lines = [f'#operator {kernel_name}',
             f'{kernel_name}:',
             '  args:']
    tensor_args = []
    for i, arg_type in enumerate(args_order):
        if arg_type == ArgType.TENSOR:
            tensor_arg = f"tensor_arg_{i}"
            tensor_args.append(tensor_arg)
            lines.append(f'    {tensor_arg}:')
            lines.append('      dtype: tensor')
        elif arg_type == ArgType.INT:
            int_arg = f"int_arg_{i}"
            lines.append(f'    {int_arg}:')
            lines.append('      dtype: int')
        elif arg_type == ArgType.FLOAT:
            float_arg = f"float_arg_{i}"
            lines.append(f'    {float_arg}:')
            lines.append('      dtype: float')
        elif arg_type == ArgType.BOOL:
            bool_arg = f"bool_arg_{i}"
            lines.append(f'    {bool_arg}:')
            lines.append('      dtype: bool')
    lines.append('  args_signature:')
    rw_write_line = '    rw_write: '
    for i, arg in enumerate(tensor_args):
        if i > 0:
            rw_write_line += ', '
        rw_write_line += arg
    lines.append(rw_write_line)
    lines.append('  labels:')
    lines.append('    side_effect_mem: True')
    lines.append('  returns:')
    lines.append(f'    out:')
    lines.append('      dtype: tensor')
    lines.append('  class:')
    lines.append(f'    name: {kernel_name}')
    src = '\n'.join(lines)
    with open(op_yaml_file, "w") as f:
        f.write(src)


def gen_ms_run_doc_yaml(kernel_name, op_doc_file):
    lines = [f'{kernel_name}:',
             '    description: |']
    src = '\n'.join(lines)
    with open(op_doc_file, "w") as f:
        f.write(src)


def gen_ms_run_build(build_dir, build_file, src_file, module_name, op_yaml_file, op_doc_file):
    lines = ['import mindspore as ms',
             'ms.ops.CustomOpBuilder(']
    lines.append(f'    name="{module_name}",')
    lines.append(f'    sources="{src_file}",')
    lines.append('    backend="Ascend",')
    lines.append(f'    build_dir="{build_dir}",')
    lines.append(f'    op_def="{op_yaml_file}",')
    lines.append(f'    op_doc="{op_doc_file}",')
    lines.append(').build()')
    src = '\n'.join(lines)
    with open(build_file, "w") as f:
        f.write(src)
