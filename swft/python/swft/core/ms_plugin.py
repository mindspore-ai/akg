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
if not os.getenv("ENABLE_SWFT_JIT", 0):
    exit()
import inspect
import subprocess
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
CONST_DIM = 128


ms_type_map = {
    ms.float16: "FP16",
    ms.float32: "FP32",
    ms.int8: "INT8",
    ms.int16: "INT16",
    ms.int32: "INT32",
    ms.bool_: "BOOL",
}


tensor_type_map = {
    "float16": "FP16",
    "float32": "FP32",
    "int8": "INT8",
    "int16": "INT16",
    "int32": "INT32",
    "bool": "BOOL",
}


arg_type_map = {
    "tensor": ArgType.TENSOR,
    "int": ArgType.INT,
    "float": ArgType.FLOAT,
    "bool": ArgType.BOOL,
}


IN_GRAPH = False


def is_graph_mode():
    return IN_GRAPH or ms.get_context("mode") == ms.GRAPH_MODE


def replace_func_with_primitive(func, prim, namespace, visited):
    for k, v in list(namespace.items()):
        if inspect.isfunction(v) and k == func.__name__:
            namespace[k] = prim
        elif isinstance(v, types.ModuleType):
            if v not in visited:
                visited.add(v)
                replace_func_with_primitive(func, prim, v.__dict__, visited)


def get_kernel_name(kernel):
    return kernel.__name__


def get_module_name(kernel):
    kernel_name = get_kernel_name(kernel)
    return f"ms_expression_{kernel_name}"


def get_build_path(kernel):
    prefix_path = "temp"
    kernel_name = get_kernel_name(kernel)
    return f"./{prefix_path}/{kernel_name}"


def get_op_primtive(kernel):
    kernel_name = get_kernel_name(kernel)
    module_name = get_module_name(kernel)
    build_path = get_build_path(kernel)
    cwd_dir = os.getcwd()
    sys.path.append(str(cwd_dir) + f"/{build_path}")
    sys.path.append(str(cwd_dir) + f"/{build_path}/{module_name}_{AUTO_GENERATE}")
    return load_func_from_module(
        GEN_OPS_PRIM, f"{kernel_name}_op", f"{build_path}/{module_name}_{AUTO_GENERATE}/{GEN_OPS_PRIM}.py")


def call_ms_op(kernel, args):
    op_primitive = get_op_primtive(kernel)
    if is_graph_mode():
        caller_globals = inspect.stack()[2][0].f_globals
        replace_func_with_primitive(kernel, op_primitive, caller_globals, set())
        return
    return op_primitive(*args)


def get_nd_shape_from_nz_shape(nz_shape):
    ndim = len(nz_shape)
    if ndim < 3:
        print("[ERROR]: only nz format with no less than 3 dimensions is supported")
    perm_shape = list(nz_shape)
    perm_shape[ndim - 3], perm_shape[ndim - 2] = perm_shape[ndim - 2], perm_shape[ndim - 3]
    nd_shape = perm_shape[:-2]
    if isinstance(perm_shape[-2], Scalar) or isinstance(perm_shape[-1], Scalar):
        print("[ERROR]: dynamic shape not supported for -3 and -1 dimensions of nz-shape")
    nd_shape.append(perm_shape[-2] * perm_shape[-1])
    return nd_shape


def get_swft_tensor_from_ms_tensor(arg_idx, tensor, is_nz_arg, dynamic_dims):
    if tensor.dtype not in ms_type_map:
        print("[ERROR]: ms tensor type not supported")
    dtype = ms_type_map[tensor.dtype]
    shape = []
    for dim_idx, dim_size in enumerate(tensor.shape):
        if arg_idx in dynamic_dims and dim_idx in dynamic_dims[arg_idx]:
            # placeholder for dynamic dim, updated after function args
            shape.append(Scalar("INT32"))
        else:
            shape.append(dim_size)
    if is_nz_arg:
        format = "NZ"
        shape = [*get_nd_shape_from_nz_shape(shape)]
    else:
        format = "ND"
    return SwftTensor("GM", dtype, shape, format=format, multi_core=False)


def run_cmd(cmd):
    try:
        subprocess.run(cmd, text=True, shell=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Command {cmd} execute failed with {e}")


def get_link_cmd(kernel, src_file, cann_path):
    kernel_name = get_kernel_name(kernel)
    module_name = get_module_name(kernel)
    build_path = get_build_path(kernel)
    flags = [f'{cann_path}/toolkit/tools/ccec_compiler/bin/ccec']
    flags.append(f'{build_path}/{kernel_name}.o {build_path}/gen_custom_ops_def.o')
    flags.append(f'{src_file.replace(".cpp", ".o")}')
    flags.append(f'-o {build_path}/{module_name}.so')
    flags.append('--cce-fatobj-link -O3 -Wall -shared -std=c++11 -fPIC -s')
    flags.append('-Wl,-Bdynamic -Wl,-z,relro,-z,now,-z,noexecstack -Wl,--disable-new-dtags,--rpath')
    flags.append(f'-I {cann_path}/acllib/include')
    flags.append(f'-L{cann_path}/lib64 -L{cann_path}/aarch64-linux/lib64')
    flags.append('-lstdc++ -lruntime -lprofapi -lascendcl')

    ms_path = os.path.dirname(os.path.abspath(ms.__file__))
    flags.append(f'-L{os.path.abspath(os.path.join(ms_path, "lib"))}')
    flags.append('-lmindspore_core -lmindspore_ms_backend -lmindspore_pynative -lmindspore_extension')
    plugin_path = os.path.abspath(os.path.join(ms_path, 'lib', 'plugin'))
    flags.append(f"-L{plugin_path}")
    flags.append(f"-L{os.path.join(plugin_path, 'ascend')}")
    flags.append('-l:libmindspore_ascend.so.2 -lmindspore_extension_ascend_aclnn')
    return " ".join(flags)


def compile_kernel_and_link_custom_op(kernel, core_num, swft_args, dynamic_dims,
                                      arg_to_name_pos_map, args_order, arg_attrs):
    kernel_name = get_kernel_name(kernel)
    module_name = get_module_name(kernel)
    build_path = get_build_path(kernel)
    kernel_id = len(kernel_list)
    pos = len(swft_args)
    for arg_idx, dim_indices in dynamic_dims.items():
        for dim_idx in dim_indices:
            scalar = Scalar("INT32")
            scalar.update_name(f"dynamic_shape_arg_{arg_idx}_dim_{dim_idx}")
            scalar.update_position(kernel_id, pos)
            swft_args.append(scalar)
            orig_tensor = swft_args[arg_idx]
            shape = orig_tensor.shape
            shape[dim_idx] = scalar
            tensor = SwftTensor(orig_tensor.mem_type, orig_tensor.dtype, shape,
                                orig_tensor.format, orig_tensor.multi_core)
            name, pos = arg_to_name_pos_map[arg_idx]
            tensor.update_name(name)
            tensor.update_position(kernel_id, pos)
            swft_args[arg_idx] = tensor
            pos += 1
    new_subkernel(core_num, kernel_name)
    kernel_list.append(kernel_id)
    compile_func(kernel, kernel.__globals__, dynamic_dims)(*swft_args)

    # compile cce
    compile_kernel(
        f"{build_path}/{kernel_name}.cce", kernel_name, idx=kernel_id)
    cann_path = os.getenv("ASCEND_HOME_PATH")
    compile_opt = f'{cann_path}/toolkit/tools/ccec_compiler/bin/ccec -xcce -O3 \
                    -I{cann_path}/compiler/tikcpp/tikcfw/ \
                    -I{cann_path}/aarch64-linux/ascendc/include/basic_api/impl/ \
                    -I{cann_path}/aarch64-linux/ascendc/include/basic_api/interface/ \
                    --cce-aicore-arch=dav-m200 \
                    -mllvm -cce-aicore-function-stack-size=16000 \
                    -mllvm -cce-aicore-record-overflow=false -mllvm \
                    -cce-aicore-addr-transform -mllvm --cce-aicore-jump-expand=true  \
                    -std=c++20 -fPIC -pthread -o \
                    {build_path}/{kernel_name}.o -c {build_path}/{kernel_name}.cce'
    run_cmd(compile_opt)

    # compile pybind
    src_file = f'{build_path}/ms_run.cpp'
    gen_ms_run_src(src_file, kernel_name, args_order, dynamic_dims, arg_attrs)
    op_yaml_file = f'{build_path}/{kernel_name}_op.yaml'
    gen_ms_run_op_yaml(kernel_name, args_order, op_yaml_file)
    op_doc_file = f'{build_path}/{kernel_name}_doc.yaml'
    gen_ms_run_doc_yaml(kernel_name, op_doc_file)
    build_file = f'{build_path}/build_custom_with_ms.py'
    gen_ms_run_build(build_path, build_file, src_file, module_name, op_yaml_file, op_doc_file)
    run_cmd(f"python {build_file}")

    link_cmd = get_link_cmd(kernel, src_file, cann_path)
    run_cmd(link_cmd)


def check_arg_attrs_valid(arg_attrs, accept_none=False):
    if accept_none and arg_attrs is None:
        return
    if not isinstance(arg_attrs, dict):
        print("[ERROR]: only dictionary type is acccepted for arg_attrs")
    for arg, attr in arg_attrs.items():
        if not isinstance(arg, str):
            print("[ERROR]: only string is accepted as key for arg_attrs")
        if not isinstance(attr, dict):
            print("[ERROR]: only dictionary type is accepted as value for arg_attrs")
        if "type" not in attr:
            print("[ERROR]: argument attribute must contain 'type'")
        arg_type = attr["type"]
        if arg_type not in ("tensor", "int", "float", "bool"):
            print(f"[ERROR]: unsupported argument type {arg_type}")
        if arg_type == "tensor":
            if "shape" not in attr:
                print("[ERROR]: for argument of type tensor, shape attr must be provided")
            shape = attr["shape"]
            if not isinstance(shape, (list, tuple)):
                print("[ERROR]: only list or tuple is accepted for argument shape")
            for dim in shape:
                if not isinstance(dim, int):
                    print("[ERROR]: only int values are supported for argument shape")
                if dim < -1:
                    print("[ERROR]: argument shape should be non-negative, or -1 for dynamic dim")
            if "dtype" not in attr:
                print("[ERROR]: for argument of type tensor, dtype attr must be provided")
            dtype = attr["dtype"]
            if dtype not in ("float16", "float32", "int8", "int16", "int32", "bool"):
                print(f"[ERROR]: unsupported tensor dtype {dtype}")
            if "format" in attr:
                format = attr["format"]
                if format not in ("ND", "NZ"):
                    print(f"[ERROR]: unsupported tensor format {format}")


def check_jit_args_valid(nz_args, dynamic_dims, arg_attrs):
    if nz_args is not None:
        if not isinstance(nz_args, (set, list, tuple)) or any(not isinstance(arg_idx, int) for arg_idx in nz_args):
            print("[ERROR]: only set, list or tuple containing int values is accepted as arg indices "
                  "for nz_args")
    if dynamic_dims is not None:
        if not isinstance(dynamic_dims, dict):
            print("[ERROR]: only dictionary type is acccepted for dynamic_dims")
        for arg_idx, dim_indices in dynamic_dims.items():
            if not isinstance(arg_idx, int):
                print("[ERROR]: only int values are accepted as argument indices for dynamic_dims")
            if (not isinstance(dim_indices, (set, list, tuple)) or
                any(not isinstance(dim_idx, int) for dim_idx in dim_indices)):
                print("[ERROR]: only set, list or tuple containing int values is accepted as dim indices "
                      "for dynamic_dims")
    check_arg_attrs_valid(arg_attrs, accept_none=True)


def jit(core_num=1, nz_args=None, dynamic_dims=None, arg_attrs=None):
    check_jit_args_valid(nz_args, dynamic_dims, arg_attrs)
    if nz_args is None:
        nz_args = []
    if dynamic_dims is None:
        dynamic_dims = {}
    if arg_attrs is None:
        arg_attrs = {}

    def logging_decorator(kernel):
        @wraps(kernel)
        def wrapped_function(*args, **kwargs):
            module_name = get_module_name(kernel)
            build_path = get_build_path(kernel)
            for v in kwargs.values():
                args.append(v)

            args_order = []
            for v in args:
                if isinstance(v, ms.Tensor):
                    args_order.append(ArgType.TENSOR)
                elif isinstance(v, int):
                    args_order.append(ArgType.INT)
                elif isinstance(v, float):
                    args_order.append(ArgType.FLOAT)
                elif isinstance(v, bool):
                    args_order.append(ArgType.BOOL)
                else:
                    print("[ERROR]: unrecognized arg type for jit")
            if os.path.exists(f"{build_path}/{module_name}.so"):
                return call_ms_op(kernel, args)
            os.system(f"mkdir -p {build_path}")

            # @sub_kernel
            kernel_id = len(kernel_list)
            var_names = list(inspect.signature(kernel).parameters.keys())
            pos = 0
            swft_args = []
            arg_to_name_pos_map = {}
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
                    is_nz_arg = idx in nz_args
                    swft_tensor = get_swft_tensor_from_ms_tensor(idx, args[idx], is_nz_arg, dynamic_dims)
                    swft_args.append(swft_tensor)
                else:
                    print("[ERROR]: unsupported args for jit!")
                swft_args[idx].update_name(name)
                swft_args[idx].update_position(kernel_id, pos)
                arg_to_name_pos_map[idx] = (name, pos)
                pos += 1
            compile_kernel_and_link_custom_op(
                kernel, core_num, swft_args, dynamic_dims, arg_to_name_pos_map, args_order, arg_attrs)
            return call_ms_op(kernel, args)
        return wrapped_function
    return logging_decorator


def get_swft_tensor_from_attr(tensor_attr):
    dtype = tensor_type_map[tensor_attr["dtype"]]
    shape = []
    for dim in tensor_attr["shape"]:
        if dim == -1:
            shape.append(Scalar("INT32"))
        else:
            shape.append(dim)
    format = tensor_attr.get("format", "ND")
    if format == "NZ":
        shape = [*get_nd_shape_from_nz_shape(shape)]
    return SwftTensor("GM", dtype, shape, format=format, multi_core=False)


def aot(kernel, core_num=1, arg_attrs=None):
    check_arg_attrs_valid(arg_attrs)
    module_name = get_module_name(kernel)
    build_path = get_build_path(kernel)
    if os.path.exists(f"{build_path}/{module_name}.so"):
        return get_op_primtive(kernel)
    os.system(f"mkdir -p {build_path}")

    # @sub_kernel
    kernel_id = len(kernel_list)
    pos = 0
    swft_args = []
    arg_to_name_pos_map = {}
    args_order = []
    for idx, (name, attr) in enumerate(arg_attrs.items()):
        arg_type = attr["type"]
        args_order.append(arg_type_map[arg_type])
        if arg_type == "int":
            scalar_int = Scalar("INT32")
            swft_args.append(scalar_int)
        elif arg_type == "bool":
            scalar_bool = Scalar("BOOL")
            swft_args.append(scalar_bool)
        elif arg_type == "float":
            scalar_float = Scalar("FP32")
            swft_args.append(scalar_float)
        elif arg_type == "tensor":
            swft_tensor = get_swft_tensor_from_attr(attr)
            swft_args.append(swft_tensor)
        else:
            print("[ERROR]: unsupported args for jit!")
        swft_args[idx].update_name(name)
        swft_args[idx].update_position(kernel_id, pos)
        arg_to_name_pos_map[idx] = (name, pos)
        pos += 1
    dynamic_dims = {i: [] for i in range(len(arg_attrs))}
    for arg_idx, (_, attr) in enumerate(arg_attrs.items()):
        if attr["type"] == "tensor":
            for dim_idx, dim_size in enumerate(attr["shape"]):
                if dim_size == -1:
                    dynamic_dims[arg_idx].append(dim_idx)
    compile_kernel_and_link_custom_op(
        kernel, core_num, swft_args, dynamic_dims, arg_to_name_pos_map, args_order, arg_attrs)
    return get_op_primtive(kernel)


@ms.ops.constexpr
def get_const_tensor(shape, dtype):
    shape = [CONST_DIM if dim is None else dim for dim in shape]
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


def compile_ms_func(func, *jit_args, **jit_kwargs):
    @wraps(func)
    def wrapped_function(*args, **kwargs):
        global IN_GRAPH
        IN_GRAPH = True
        func(*args, **kwargs)
        IN_GRAPH = False
        return ms.jit(*jit_args, **jit_kwargs)(func)(*args, **kwargs)
    return wrapped_function
