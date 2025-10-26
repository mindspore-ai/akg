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
"""Module for kernel test and profiling"""
import argparse
import ctypes
import json
import logging
import multiprocessing
import os
import pathlib
import shutil
import subprocess
import time

import numpy as np
from akg import AkgMlirDriver
from ..utils.composite_op_helper import compare_tensor, gen_json_data
from ..utils.dynamic_utils import dump_shape_arg_list, get_device_shape
from ..utils.gen_runtime_code import (ProfilingParams,
                                      gen_cuda_runtime_code)
from ..utils.result_analysis import get_compare_tolerance
from ..ascend_profilier.cann_file_parser import CANNFileParser
from ..ascend_profilier.op_summary_parser import OpSummaryParser
from ..ascend_profilier.op_summary_headers import OpSummaryHeaders
from bfloat16 import bfloat16

import akg.akgAscendLaunch as akgAscendLaunch
import akg.akgProfileMgr as akgProfileMgr

PROF_ERROR_CODE = 9999999999

def validate_and_normalize_path(
        path,
        check_absolute_path=False,
        allow_parent_dir=True,
):
    """
    Validates path and returns its normalized form.

    If path has a valid scheme, treat path as url, otherwise consider path a
    unix local path.

    Note:
        File scheme (rfc8089) is currently not supported.

    Args:
        path (str): Path to be normalized.
        check_absolute_path (bool): Whether check path scheme is supported.
        allow_parent_dir (bool): Whether allow parent dir in path.

    Returns:
        str, normalized path.
    """
    if not path:
        raise RuntimeError("The path is invalid!")

    path_str = str(path)
    if not allow_parent_dir:
        path_components = path_str.split("/")
        if ".." in path_components:
            raise RuntimeError("The parent path is not allowed!")

    # path does not have valid schema, treat it as unix local path.
    if check_absolute_path:
        if not path_str.startswith("/"):
            raise RuntimeError("The path is invalid!")
    try:
        # most unix systems allow
        normalized_path = os.path.realpath(path)
    except ValueError:
        raise RuntimeError("The path is invalid!")

    return normalized_path

def _get_json_dict(desc):
    return json.loads(desc) if isinstance(desc, str) else desc


def _get_kernel_name(desc):
    json_obj = _get_json_dict(desc)
    return json_obj["op"]


def _transform_data_to_ctypes(data,
                              kernel_name,
                              is_dyn_shape=False,
                              backend="gpu",
                              is_profile_params=False,
                              ):
    data_ctypes = list()
    if len(data) == 0:
        # dynamic shape info cannot generate inputs while compilation
        return data_ctypes
    shape_arg_list = list()
    int_p = ctypes.POINTER(ctypes.c_int)
    device_shape, _, _ = get_device_shape(
        data, kernel_name, is_dyn_shape and not is_profile_params)

    for data_idx, d in enumerate(data):
        shape_list = [0]
        data_shape = device_shape[data_idx]
        if isinstance(d, int):
            data_ctypes.append(ctypes.c_int(d))
        elif isinstance(d, np.ndarray):
            data_ctypes.append(d.ctypes.data_as(int_p))
            shape_list += list(data_shape)
            # for tensor (m, n, k), strides is [n*k, k, 1]
            stride_list = [1] * len(data_shape)
            for idx, _ in enumerate(data_shape[1:]):
                stride_list[-idx - 2] = stride_list[-idx - 1] * \
                    data_shape[-idx - 1]
            shape_list += stride_list
        else:
            raise TypeError("wrong data to cytpes, current type is '", type(d),
                            "'")
        shape_list += [0] * (1 + 2 * 0 if is_profile_params else max([len(shape)
                             for shape in device_shape]) - len(shape_list))
        shape_arg_list.append(shape_list)
    if is_profile_params or backend == "gpu":
        return data_ctypes
    # pack parameters into an array of pointers
    # static shape: array of pointers of data
    packed_tensors = (int_p * len(data))()
    packed_tensors[:] = [
        ctypes.cast(data_ctype, int_p) for data_ctype in data_ctypes
    ]

    if backend == "cpu" and not is_dyn_shape:
        return [packed_tensors]
    else:
        # dynamic shape: array of pointers of data, array of [0, shape, stride] of data
        # tensor_num * [0, shape_list 1,2,...,n, strides 0,1,2,...,n]
        packed_shape_lists = (int_p * len(data))()
        for idx, shape_list in enumerate(shape_arg_list):
            packed_shapes = (int_p * len(shape_list))()
            packed_shapes[:] = [
                ctypes.cast(shape, int_p) for shape in shape_list
            ]
            packed_shape_lists[idx] = ctypes.cast(packed_shapes, int_p)
        return [packed_tensors, packed_shape_lists]

def _transform_data_to_ctypes_ascend(data,
                              kernel_name,
                              output_indexes,
                              is_dyn_shape=False,
                              backend="ascend",
                              is_profile_params=False,
                              ):
    data_ctypes = list()
    if len(data) == 0:
        # dynamic shape info cannot generate inputs while compilation
        return data_ctypes
    shape_arg_list = list()

    device_shape, _, _ = get_device_shape(
        data, kernel_name, is_dyn_shape and not is_profile_params)

    output_idx_set = list()
    for output_idx in output_indexes:
        if output_idx >= 0:
            output_idx_set.append(output_idx)
        else:
            output_idx_set.append(output_idx + len(data))
    output_idx_set = set(output_idx_set)
    for data_idx, d in enumerate(data):
        data_shape = np.array(device_shape[data_idx])
        data_bytes = d.nbytes
        is_bf16 = False
        if (d.dtype.name == "bfloat16"):
            d = d.astype(np.float32)
            data[data_idx] = d
            is_bf16 = True
        if isinstance(d, int):
            data_ctypes.append(ctypes.c_int(d))
        elif isinstance(d, np.ndarray):
            ascend_tensor_obj = akgAscendLaunch.AscendTensorObjStructPyTorch()
            data_addr = d.ctypes.data_as(ctypes.c_void_p)
            shape_addr = data_shape.ctypes.data_as(ctypes.c_void_p)
            is_output = data_idx in output_idx_set
            is_dynamic = False #now we consider static shape first;
            ascend_tensor_obj.tensor_info = d
            ascend_tensor_obj.shape_info = data_shape
            ascend_tensor_obj.nbytes = data_bytes
            ascend_tensor_obj.is_output = is_output
            ascend_tensor_obj.is_dynamic = is_dynamic
            ascend_tensor_obj.is_bf16 = is_bf16
            data_ctypes.append(ascend_tensor_obj)

    return data_ctypes

def _compile_lib(kernel_name, file_path="./tmp_files/"):
    so_file = os.path.join(file_path, "gen_func_" + kernel_name + ".so")
    gen_lib_file = os.path.join(file_path, "gen_func_" + kernel_name + ".cu")

    cmd = ["nvcc", "-o", so_file, gen_lib_file, "--shared",
           "-Xcompiler", "-fPIC", "-lcudart", "-lcuda", "-O3"]

    subprocess.call(cmd)


def _compare_func(output, expect, compare_tolerance=None):
    return compare_tensor(output, expect, rtol=compare_tolerance, atol=compare_tolerance)


def create_executable(kernel_name,
                      input_for_mod,
                      output_indexes,
                      is_dyn_shape):
    """Generate executable files"""
    cur_path = _get_kernel_meta_dir()
    tmp_file_path = _get_tmp_dir()
    tmp_file_name = os.path.join(
        tmp_file_path, "gen_func_" + kernel_name + ".so")
    fake_output_indices = list()
    gen_cuda_runtime_code(kernel_name,
                          input_for_mod,
                          output_indexes,
                          is_dyn_shape,
                          fake_output_indices,
                          path=cur_path)
    try:
        _compile_lib(kernel_name, file_path=tmp_file_path)
    except (Exception,):
        raise RuntimeError("Compile cuda runtime lib fail")
    try:
        lib = ctypes.cdll.LoadLibrary(tmp_file_name)
    except (Exception,):
        raise RuntimeError("Load cuda runtime lib fail")
    return lib


def compare_results(kernel_name, desc, input_for_mod, output_indexes, expect):
    """Helper function to compare result"""
    output = list(input_for_mod[i] for i in output_indexes)
    if isinstance(output, tuple) and len(output) > 0 and isinstance(
            output[-1], dict):
        output = output[0]
    output = output if isinstance(output, (list, tuple)) else [output]
    expect = expect if isinstance(expect, (list, tuple)) else [expect]
    output = list(output)
    expect = list(expect)
    print("output:", output)
    print("expect:", expect)
    compare_tolerance = get_compare_tolerance(desc, output_indexes)
    compare_res = list(map(_compare_func, output, expect, compare_tolerance))
    if not all(compare_res):
        print(kernel_name + " precision error")
    else:
        print(kernel_name + " precision correct")

def _get_kernel_meta_dir():
    kernel_meta_dir = os.getenv("KERNEL_META_DIR", default="akg_kernel_meta")
    return kernel_meta_dir

def _get_tmp_dir():
    return os.path.join(_get_kernel_meta_dir(), "tmp_files")

def _create_dirs():
    dir_paths = [_get_kernel_meta_dir(), _get_tmp_dir()]
    for file_path in dir_paths:
        if not os.path.exists(file_path):
            os.makedirs(file_path)


def _clear_tmp_dirs(kernel_name):
    dir_paths = [_get_kernel_meta_dir(), _get_tmp_dir()]
    for file_path in dir_paths:
        if not os.path.exists(file_path):
            continue
        for file_name in os.listdir(file_path):
            if kernel_name not in file_name:
                continue
            target = os.path.join(file_path, file_name)
            if os.path.isfile(target):
                os.remove(target)
            elif os.path.isdir(target):
                shutil.rmtree(target)


def _auto_get_target(desc):
    desc_d = json.loads(desc)
    process = desc_d.get("process", None)
    if process is None:
        raise RuntimeError("Can't get process in the json desc")
    return "gpu" if process == "cuda" else "cpu"


def _run_gpu_kernel(akg_mlir_driver, is_dyn_shape, input_for_mod, kernel_name,
                    output_indexes, desc, profiling_trails, expect):
    if is_dyn_shape:
        dump_shape_arg_list(input_for_mod, kernel_name, str(
            pathlib.Path(__file__).absolute().parent))

    akg_mlir_driver.run_gpu()
    input_for_mod_ctypes = _transform_data_to_ctypes(
        input_for_mod,
        kernel_name,
        is_dyn_shape,
        "gpu")
    if profiling_trails == 0:
        # Run executable
        lib = create_executable(kernel_name, input_for_mod, output_indexes,
                                is_dyn_shape)
        lib.cuda_runtime_exec(*input_for_mod_ctypes)
        compare_results(kernel_name, desc, input_for_mod, output_indexes,
                        expect)
    else:
        # Profiling
        prof_params = ProfilingParams(number=10,
                                      repeat=profiling_trails,
                                      min_repeat_ms=0)
        prof_params_ctypes = _transform_data_to_ctypes(
            prof_params.get_data(),
            kernel_name,
            is_dyn_shape,
            "gpu",
            is_profile_params=True
        )
        lib = create_executable(kernel_name, input_for_mod, output_indexes,
                                is_dyn_shape)
        lib.cuda_runtime_profiling(*input_for_mod_ctypes,
                                   *prof_params_ctypes)


def _run_cpu_kernel(akg_mlir_driver, is_dyn_shape, input_for_mod, kernel_name,
                    output_indexes, desc, profiling_trails, expect, replace_dso):
    akg_mlir_driver.run_cpu()
    # Run executable and profiling
    if replace_dso:
        dso_path = os.path.join(_get_kernel_meta_dir(), kernel_name + "_custom.so")
        if os.path.exists(dso_path):
            logging.info(
                "Try to use the customized dso file : %s", dso_path)
        else:
            raise ValueError(
                "Failed to find the customized dso file : " + dso_path)
    else:
        dso_path = os.path.join(_get_kernel_meta_dir(), kernel_name + ".so")
    cur = ctypes.cdll.LoadLibrary(dso_path)
    input_for_mod_ctypes = _transform_data_to_ctypes(
        input_for_mod, kernel_name, is_dyn_shape, "cpu")
    # Profiling
    if profiling_trails > 0:
        func = cur.__getattr__("main")
        np_timers_ns = np.array([0], dtype=np.int64)
        input_for_mod_ctypes.append(
            np_timers_ns.ctypes.data_as(
                ctypes.POINTER(ctypes.c_longlong)))
        func(*input_for_mod_ctypes)
        print(kernel_name, ": Running ", profiling_trails,
              " times, the average execution time is ",
              np_timers_ns / 1000000 / profiling_trails, " ms.")
    else:
        func = cur.__getattr__(kernel_name)
        # Run executable and compare results
        func(*input_for_mod_ctypes)
        compare_results(kernel_name, desc, input_for_mod,
                        output_indexes, expect)

def profiling_analyse(arch):
    public_path = os.getenv('PROFILING_DIR')
    if public_path is None:
        raise RuntimeError("Environment PROFILING_DIR not set!")
    public_path = validate_and_normalize_path(public_path)
    CANNFileParser(public_path).export_cann_profiling()
    cann_file_parser = OpSummaryParser(public_path, arch)
    task_duration = cann_file_parser.generate_op_summary_data()
    return task_duration

def _run_ascend_kernel(akg_mlir_driver, is_dyn_shape, input_for_mod, kernel_name,
                    output_indexes, desc, profiling_trails, expect, replace_dso):
    ascend_path = os.getenv("ASCEND_HOME_PATH", "")
    if ascend_path == "":
        raise Exception("ASCEND_HOME_PATH is not sett, source <ascend-toolkit>/set_env.sh first")
    os.environ['RT_LIB'] = os.path.join(ascend_path, 'lib64')
    akg_mlir_driver.run_ascend()
    # Run executable and profiling
    if replace_dso:
        dso_path = os.path.join(_get_kernel_meta_dir(), kernel_name + "_custom.so")
        if os.path.exists(dso_path):
            logging.info(
                "Try to use the customized dso file : %s", dso_path)
        else:
            raise ValueError(
                "Failed to find the customized dso file : " + dso_path)
    else:
        dso_path = os.path.join(_get_kernel_meta_dir(), "lib" + kernel_name + ".so")
    # load ascend_run func
    launch_func_name = "asend_run"

    cur = ctypes.cdll.LoadLibrary(dso_path)
    input_for_mod_ctypes = _transform_data_to_ctypes_ascend(
        input_for_mod, kernel_name, output_indexes, is_dyn_shape, "ascend")
    # Run executable and compare results
    device_id = int(os.environ.get("DEVICE_ID", 0))
    dso_path = _get_kernel_meta_dir()
    if profiling_trails == 0:
        akgAscendLaunch.akg_ascend_run(dso_path, kernel_name, device_id, is_dyn_shape, *input_for_mod_ctypes)
        for idx, d in enumerate(expect):
            expect[idx] = d.astype(np.float32) if d.dtype == bfloat16 else d
        compare_results(kernel_name, desc, input_for_mod,
            output_indexes, expect)
    else:
        akgProfileMgr.ascend_start_profiling(device_id)
        for i in range(5):
            akgAscendLaunch.akg_ascend_run(dso_path, kernel_name, device_id, is_dyn_shape, *input_for_mod_ctypes)
        akgProfileMgr.ascend_stop_profiling()
        # analysis
        cycle = profiling_analyse(None)
        logging.info('=====Task Duration(us)==============================')
        if cycle != PROF_ERROR_CODE:
            logging.info(cycle)
        else:
            logging.error("OOPS, can't correctly Task Duration!")
        logging.info('=====Task Duration(us)==============================')
        for idx, d in enumerate(expect):
            expect[idx] = d.astype(np.float32) if d.dtype == bfloat16 else d
        compare_results(kernel_name, desc, input_for_mod,
            output_indexes, expect)
        print(kernel_name + " Task Duration(us) : " + str(cycle))
    return

def run_a_kernel(desc,
                 file_path,
                 backend="gpu",
                 profiling_trails=0,
                 static_desc=None,
                 clear_tmp=False,
                 dump_ir=False,
                 replace_dso=False,
                 repo_path="",
                 enable_akg_loop_fusion=False):
    """function to run a single kernel"""
    is_dyn_shape = (static_desc is not None)
    if not backend:
        backend = _auto_get_target(desc)

    kernel_name = _get_kernel_name(desc)
    # Generate data
    input_for_mod, expect, output_indexes = gen_json_data(
        static_desc if is_dyn_shape else desc, with_compute=True)
    # Init AkgMlirDriver
    akg_mlir_driver = AkgMlirDriver(input_file=file_path,
                                output_dir=_get_kernel_meta_dir(),
                                llvm_tools_dir=os.getenv("LLVM_HOME", ""),
                                dynamic_shape=is_dyn_shape,
                                dump_ir=dump_ir,
                                repo_path=repo_path,
                                profiling_trails=profiling_trails,
                                runtime_provider="MLIR",
                                enable_akg_loop_fusion=enable_akg_loop_fusion)

    if backend == "gpu":
        _run_gpu_kernel(akg_mlir_driver, is_dyn_shape, input_for_mod, kernel_name,
                        output_indexes, desc, profiling_trails, expect)
    elif backend == "cpu":
        _run_cpu_kernel(akg_mlir_driver, is_dyn_shape, input_for_mod, kernel_name,
                        output_indexes, desc, profiling_trails, expect, replace_dso)
    elif backend == "ascend":
        _run_ascend_kernel(akg_mlir_driver, is_dyn_shape, input_for_mod, kernel_name,
                        output_indexes, desc, profiling_trails, expect, replace_dso)
    else:
        TypeError("only support gpu, cpu, ascend backend currently")
    if clear_tmp:
        _clear_tmp_dirs(kernel_name)


def _get_compute(desc):
    desc_d = json.loads(desc)
    compute = []
    for op in desc_d.get("op_desc", []):
        op_name = op.get("name", "")
        compute.append(op_name)
    return compute


def _get_input_shape(desc):
    is_dyn_shape = False
    desc_d = json.loads(desc)
    input_desc = desc_d.get("input_desc", [])
    if input_desc is None:
        return [], False
    input_shape = []
    for inputs in input_desc:
        shape = inputs[0].get("shape", [])
        input_shape.append(shape)
        is_dyn_shape = (-1 in shape or -2 in shape) or is_dyn_shape
    for output in desc_d.get("output_desc"):
        shape = output.get("shape", [])
        is_dyn_shape = (-1 in shape or -2 in shape) or is_dyn_shape
    return input_shape, is_dyn_shape


def _get_input_dtype(desc):
    desc_d = json.loads(desc)
    input_desc = desc_d.get("input_desc", [])
    if input_desc is None:
        return []
    input_dtype = []
    for inputs in input_desc:
        shape = inputs[0].get("data_type")
        input_dtype.append(shape)
    return input_dtype


def _run_single_file(file_path, compile_args, run_res=None, run_idx=None):
    with open(file_path, "r") as f:
        desc = f.read()
        kernel_name = _get_kernel_name(desc)
        input_shape, is_dyn_shape = _get_input_shape(desc)
        static_desc = None
        if is_dyn_shape:
            static_shape_path = file_path.replace(".info", "_static.info")
            if not os.path.exists(static_shape_path):
                raise ValueError(
                    "Dynamic shape info must come with static shape info.")
            with open(static_shape_path, "r") as s_f:
                static_desc = s_f.read()
        if compile_args.profiling:
            print("profiling ", kernel_name)
            compute = _get_compute(desc)
            logging.info("input_shape = %s compute %s", input_shape, compute)
            logging.info("input_dtype = %s", _get_input_dtype(desc))

        try:
            print("Start running " + kernel_name)
            run_a_kernel(desc,
                         file_path,
                         backend=compile_args.backend,
                         profiling_trails=compile_args.prof_trails,
                         static_desc=static_desc,
                         clear_tmp=compile_args.clear_tmp,
                         dump_ir=bool(compile_args.dump_ir),
                         replace_dso=compile_args.replace_dso,
                         repo_path=compile_args.repo_path,
                         enable_akg_loop_fusion=compile_args.akg_fusion)
        except ValueError:
            print(kernel_name + " precision error = 9999999997ms")
            if run_res is not None and run_idx is not None:
                run_res[run_idx] = False
            return False

        if run_res is not None and run_idx is not None:
            run_res[run_idx] = True
        return True

class TestUtils:
    """Class for getting cycle and core num."""

    @staticmethod
    def record_cycle(cycle):
        if os.environ.get(PERFORMANCE_TEST_FILE):
            result_file = os.environ.get(PERFORMANCE_TEST_FILE)
            with os.fdopen(os.open(result_file, os.O_WRONLY | os.O_CREAT), "a+") as f:
                f.write("{0}\n".format(cycle))

    @staticmethod
    def record_core(stmt):
        """Function for getting performance data from cores."""

        def get_core_num():
            core_num = 1
            if hasattr(stmt, 'attr_key') and stmt.attr_key == 'thread_extent':
                core_num = stmt.value
            return core_num

        if os.environ.get(PERFORMANCE_TEST_FILE):
            result_file = os.environ.get(PERFORMANCE_TEST_FILE)
            with os.fdopen(os.open(result_file, os.O_WRONLY | os.O_CREAT), "a+") as f:
                f.write("{0}; ".format(get_core_num()))

def main(args=None):
    # usage: python py_benchmark.py -e gpu --file ./info_cases/Fused_BiasAdd_1551558231201032373.info --prof_trails 100
    parser = argparse.ArgumentParser(description='Run cases')
    parser.add_argument("-e",
                        "--backend",
                        choices=['cpu', 'gpu', 'ascend'],
                        type=str,
                        required=False,
                        default="",
                        help="Hardware environment: cpu, gpu or ascend.")
    parser.add_argument("-f", "--file", type=str, default="")
    parser.add_argument("-d", "--dir", type=str, default="")
    parser.add_argument("-tr",
                        "--prof_trails",
                        type=int,
                        required=False,
                        default=0)
    parser.add_argument("-p", "--profiling", type=bool, default=False)
    parser.add_argument("-c", "--clear_tmp", type=bool, default=False)
    parser.add_argument("-ci", "--ci_test", type=bool, default=False)
    parser.add_argument("-t", "--threads", type=int, default=1)
    parser.add_argument("-dump", "--dump_ir", type=int, default=0)
    parser.add_argument("-r", "--replace_dso", type=bool, default=False)
    parser.add_argument("-repo", "--repo_path", type=str, default="")
    parser.add_argument("-af", "--akg_fusion", type=bool, default=False)

    args = parser.parse_args()
    _create_dirs()
    if args.dir:
        files = [
            f for f in os.listdir(args.dir)
            if f.endswith(".info") and not f.endswith("_static.info")
        ]
        process_state = multiprocessing.Manager().list(
            [None for _ in range(len(files))]) if args.ci_test else None
        process_tasks = []

        def _has_fail():
            if process_state is None:
                return False
            for idx, succ in enumerate(process_state):
                not_start = idx >= len(process_tasks)
                if not_start:
                    continue
                not_finish = succ is None and process_tasks[idx].is_alive()
                if not_finish:
                    continue
                if not succ and args.ci_test:
                    logging.warning("dir test fail : %s", files[idx])
                    return True
            return False

        def _alive_task():
            return sum(1 for p in process_tasks if p.is_alive())

        for i, file in enumerate(files):
            path = args.dir + "/" + file

            while _alive_task() >= args.threads:
                time.sleep(1)
            if _has_fail():
                break
            p = multiprocessing.Process(target=_run_single_file,
                                        args=(path, args, process_state, i))
            process_tasks.append(p)
            p.start()

        while _alive_task() > 0:
            time.sleep(1)
        print("Finish profiling, total file {}".format(len(files)))
        if args.ci_test and not _has_fail():
            print("dir test success")
    else:
        if args.ci_test and args.file.endswith("_static.info"):
            print("Skip static info")
            print("precision correct")
        else:
            _run_single_file(args.file, args)

if __name__ == "__main__":
    main()