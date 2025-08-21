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
if not os.getenv("ENABLE_SWFT_JIT", 1):
    exit()
import inspect
import sys
import types
import mindspore as ms
from functools import wraps
from .c_expression import new_subkernel
from .compile import compile_kernel
from .compile_func import compile_func
from .scalar import Scalar
from .sub_kernel import kernel_list, load_func_from_module
from .tensor import Tensor as SwftTensor
from swft.utils import ArgType, gen_ms_run_src, gen_ms_run_build, gen_ms_run_op_yaml, gen_ms_run_doc_yaml


GEN_OPS_PRIM = "gen_ops_prim"
AUTO_GENERATE = "auto_generate"


ms_type_map = {
    ms.float16: "FP16",
    ms.float32: "FP32",
    ms.int8: "INT8",
    ms.int16: "INT16",
    ms.int32: "INT32",
    ms.bool_: "BOOL",
}


def is_graph_mode():
    return ms.get_context("mode") == ms.GRAPH_MODE


def replace_func_with_prim(func, prim, namespace, visited):
    for k, v in list(namespace.items()):
        if inspect.isfunction(v) and k == func.__name__:
            namespace[k] = prim
        elif isinstance(v, types.ModuleType):
            if v not in visited:
                visited.add(v)
                replace_func_with_prim(func, prim, v.__dict__, visited)


def call_ms_op(build_path, module_name, kernel_name, func,
               tensor_args, int_args, float_args, bool_args):
    if is_graph_mode():
        cwd_dir = os.getcwd()
        sys.path.append(str(cwd_dir) + f"/{build_path}")
        sys.path.append(str(cwd_dir) + f"/{build_path}/{AUTO_GENERATE}")
        graph_mode_prim = load_func_from_module(
            GEN_OPS_PRIM, f"{kernel_name}_op", f"{build_path}/{AUTO_GENERATE}/{GEN_OPS_PRIM}.py")
        caller_globals = inspect.stack()[2][0].f_globals
        replace_func_with_prim(func, graph_mode_prim, caller_globals, set())
        return
    func = load_func_from_module(module_name, kernel_name, f"{build_path}/{module_name}.so")
    return func(tensor_args, int_args, float_args, bool_args)


def get_nd_shape_from_nz_shape(nz_shape):
    ndim = len(nz_shape)
    if ndim < 3:
        print("[ERROR]: only nz format with no less than 3 dimensions is supported")
    perm_shape = list(nz_shape)
    perm_shape[ndim - 3], perm_shape[ndim - 2] = perm_shape[ndim - 2], perm_shape[ndim - 3]
    nd_shape = perm_shape[:-2]
    nd_shape.append(perm_shape[-2] * perm_shape[-1])
    return nd_shape


def get_swft_tensor_from_ms_tensor(tensor, is_nz_arg):
    if isinstance(tensor, ms.Tensor):
        if tensor.dtype not in ms_type_map:
            print("[ERROR]: ms tensor type not supported")
        dtype = ms_type_map[tensor.dtype]
    shape = [*tensor.shape]
    if is_nz_arg:
        format = "NZ"
        shape = [*get_nd_shape_from_nz_shape(shape)]
    else:
        format = "ND"
    return SwftTensor("GM", dtype, shape, format=format, multi_core=False)


def jit(core_num=1, nz_args=None):
    def logging_decorator(func):
        @wraps(func)
        def wrapped_function(*args, **kwargs):
            prefix_path = "temp"
            kernel_name = func.__name__
            module_name = f"ms_expression_{kernel_name}"
            build_path = f'./{prefix_path}/{kernel_name}'

            tensor_args = []
            int_args = []
            float_args = []
            bool_args = []
            args_order = []
            for v in args:
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
                    print("[ERROR]: unrecognized arg type for jit")
            if os.path.exists(f"{build_path}/{module_name}.so"):
                return call_ms_op(build_path, module_name, kernel_name, func,
                                  tensor_args, int_args, float_args, bool_args)
            os.system(f"mkdir -p {build_path}")

            # @sub_kernel
            kernel_id = len(kernel_list)
            var_names = list(inspect.signature(func).parameters.keys())
            pos = 0
            swft_args = []
            for idx, name in enumerate(var_names):
                if isinstance(args[idx], int):
                    scalar_int = Scalar("INT32")
                    swft_args.append(scalar_int)
                elif isinstance(args[idx], bool):
                    scalar_bool = Scalar("BOOL")
                    swft_args.append(scalar_bool)
                elif isinstance(args[idx], float):
                    scalar_float = Scalar("FP32")
                    swft_args.append(scalar_float)
                elif isinstance(args[idx], ms.Tensor):
                    is_nz_arg = hasattr(nz_args, "__contains__") and idx in nz_args
                    swft_tensor = get_swft_tensor_from_ms_tensor(args[idx], is_nz_arg)
                    swft_args.append(swft_tensor)
                else:
                    print("[ERROR]: unsupported args for jit!")
                swft_args[idx].update_name(name)
                swft_args[idx].update_position(kernel_id, pos)
                pos += 1
            new_subkernel(core_num, kernel_name)
            kernel_list.append(kernel_id)
            compile_func(func, func.__globals__)(*swft_args)

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
            gen_ms_run_src(src_file, lib_path, kernel_name, args_order)
            op_yaml_file = f'{build_path}/{kernel_name}_op.yaml'
            gen_ms_run_op_yaml(kernel_name, args_order, op_yaml_file)
            op_doc_file = f'{build_path}/{kernel_name}_doc.yaml'
            gen_ms_run_doc_yaml(kernel_name, op_doc_file)
            build_file = f'{build_path}/build_custom_with_ms.py'
            gen_ms_run_build(build_path, build_file, src_file, module_name, op_yaml_file, op_doc_file)
            os.system(f"python {build_file}")
            return call_ms_op(build_path, module_name, kernel_name, func,
                              tensor_args, int_args, float_args, bool_args)
        return wrapped_function
    return logging_decorator


@ms.ops.constexpr
def get_const_tensor(shape, dtype):
    return ms.Tensor(shape=shape, dtype=dtype)


def cell_construct_wrapper(cell, func):
    origin_construct = cell.construct
    def construct(self, *args, **kwargs):
        const_args = []
        for arg in args:
            if isinstance(arg, ms.Tensor):
                const_tensor = get_const_tensor(arg.shape, arg.dtype)
                const_args.append(const_tensor)
            else:
                const_args.append(arg)
        func(*const_args, **kwargs)
        return origin_construct(*args, **kwargs)
    cell.construct = types.MethodType(construct, cell)


def compile_ms_cell(cell):
    if is_graph_mode():
        original_init = cell.__init__
        def __init__(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            self.swft_graph_mode_hook = ms.ops.constexpr(self.construct)
            cell_construct_wrapper(self, self.swft_graph_mode_hook)
        cell.__init__ = __init__
    return cell
