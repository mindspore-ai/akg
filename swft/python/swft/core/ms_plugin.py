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

try:
  from mindspore.ops import CustomOpBuilder
  del CustomOpBuilder
except ImportError:
  exit()
import os
import mindspore as ms
from functools import wraps
from inspect import signature
from .c_expression import new_subkernel
from .compile import compile_kernel
from .scalar import Scalar
from .sub_kernel import kernel_list, load_pybind_op
from .tensor import Tensor as SwftTensor
from swft.utils import gen_ms_run_src, gen_ms_run_build, ArgType


ms_type_map = {
    ms.float16: "FP16",
    ms.float32: "FP32",
    ms.int8: "INT8",
    ms.int16: "INT16",
    ms.int32: "INT32",
    ms.bool_: "BOOL",
}
ms_arg_idx = 0


def call_ms_op(build_path, module_name, ms_run_func_name,
               tensor_args, int_args, float_args, bool_args):
    ms_run_func = load_pybind_op(module_name, ms_run_func_name, f"{build_path}/{module_name}.so")
    return ms_run_func(tensor_args, int_args, float_args, bool_args)


def get_swft_tensor_from_ms_tensor(tensor):
    if tensor.dtype not in ms_type_map:
        print("[ERROR]: ms tensor type not supported")
    dtype = ms_type_map[tensor.dtype]
    shape = [*tensor.shape]
    return SwftTensor("GM", dtype, shape, format="ND", multi_core=False)


def ms_jit(core_num=1):
    def logging_decorator(func):
        @wraps(func)
        def wrapped_function(*args, **kwargs):
            prefix_path = "temp"
            kernel_name = func.__name__
            module_name = f"ms_expression_{kernel_name}"
            ms_run_func_name = f"ms_run_{kernel_name}"
            build_path = f'./{prefix_path}/{kernel_name}'

            tensor_args = []
            int_args = []
            float_args = []
            bool_args = []
            args_order = []
            for i, v in enumerate(args):
                if isinstance(v, ms.Tensor):
                    tensor_args.append(v)
                    args_order.append(ArgType.TENSOR)
                elif isinstance(v, int):
                    int_args.append(v)
                    args_order.append(ArgType.INT)
                elif isinstance(v, float):
                    float_args.append(v)
                    args_order.append(ArgType.FLOAT)
                elif isinstance(v, bool):
                    bool_args.append(v)
                    args_order.append(ArgType.BOOL)
                else:
                    print("[ERROR]: unrecognized arg type for ms_jit")
            if os.path.exists(f"{build_path}/{module_name}.so"):
                return call_ms_op(build_path, module_name, ms_run_func_name,
                                  tensor_args, int_args, float_args, bool_args)
            os.system(f"mkdir -p {build_path}")

            # @sub_kernel
            kernel_id = len(kernel_list)
            var_names = list(signature(func).parameters.keys())
            pos = 0
            swft_args = []
            for idx in range(len(var_names)):
                # TODO change to scalar
                if isinstance(args[idx], int):
                    scalar_int = Scalar("INT32", args[idx])
                    swft_args.append(scalar_int)
                elif isinstance(args[idx], bool):
                    scalar_bool = Scalar("BOOL", args[idx])
                    swft_args.append(scalar_bool)
                elif isinstance(args[idx], float):
                    scalar_float = Scalar("FP32", args[idx])
                    swft_args.append(scalar_float)
                elif isinstance(args[idx], ms.Tensor):
                    swft_tensor = get_swft_tensor_from_ms_tensor(args[idx])
                    swft_args.append(swft_tensor)
                    global ms_arg_idx
                    swft_args[idx].update_name(f"ms_arg_{ms_arg_idx}")
                    ms_arg_idx += 1
                    swft_args[idx].update_position(kernel_id, pos)
                    pos += 1
                else:
                    print("[ERROR]: unsupported args for ms_jit!")
            new_subkernel(core_num, kernel_name)
            kernel_list.append(kernel_id)
            func(*swft_args, **kwargs)
            
            # compile cce
            compile_kernel(
                f"{build_path}/{kernel_name}.cce", kernel_name)
            cann_path = os.getenv("ASCEND_HOME_PATH")
            lib_path = f"{build_path}/{kernel_name}.so"
            compile_opt = f'{cann_path}/toolkit/tools/ccec_compiler/bin/ccec -xcce --cce-aicore-arch=dav-m200 -mllvm \
                            -cce-aicore-function-stack-size=16000 -mllvm -cce-aicore-record-overflow=false -mllvm \
                            -cce-aicore-addr-transform -mllvm --cce-aicore-jump-expand=true -fPIC -pthread -o \
                         {build_path}/{kernel_name}.o -c {build_path}/{kernel_name}.cce'
            link_opt = f"{cann_path}/toolkit/tools/ccec_compiler/bin/ccec --cce-fatobj-link -O3 -Wall -shared \
                         -std=c++11  -fPIC -o {lib_path}  {build_path}/{kernel_name}.o -I {cann_path}/acllib/include \
                         -L {cann_path}/lib64/ -L {cann_path}/aarch64-linux/lib64/ -Wl,-Bdynamic -lstdc++ -lruntime \
                         -lprofapi -lascendcl"
            os.system(compile_opt)
            os.system(link_opt)

            # compile pybind
            src_file = f'{build_path}/ms_run.cpp'
            ms_run_func_name = f"ms_run_{kernel_name}"
            gen_ms_run_src(src_file, lib_path, kernel_name, ms_run_func_name, args_order)
            build_file = f'{build_path}/build_custom_with_ms.py'
            gen_ms_run_build(build_path, build_file, src_file, module_name)
            os.system(f"python {build_file}")
            return call_ms_op(build_path, module_name, ms_run_func_name,
                              tensor_args, int_args, float_args, bool_args)
        return wrapped_function
    return logging_decorator
