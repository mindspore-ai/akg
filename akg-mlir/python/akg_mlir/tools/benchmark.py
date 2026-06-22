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
"""Module for kernel test and profiling"""
import os
import sys
import argparse
import json
import logging
import multiprocessing
import pathlib
import shutil
import time
import distutils
import numpy as np

from akg import MlirDriver
from akg.utils.composite_op_helper import compare_tensor, gen_json_data
from akg.utils.result_analysis import get_compare_tolerance
from akg.utils.torch_mlir_utils import (find_first_func_name, run_torch_mlir_to_json,
                                        run_torch_mlir_to_linalg_on_tensors, get_named_op_str)

logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)s] %(asctime)s [%(filename)s:%(lineno)d] %(message)s')


def _get_json_dict(desc):
    return json.loads(desc) if isinstance(desc, str) else desc


def _get_kernel_name(desc):
    json_obj = _get_json_dict(desc)
    return json_obj["op"]


def _get_arch_name(desc):
    json_obj = _get_json_dict(desc)
    target_info = json_obj.get("target_info", {})
    return target_info.get("arch", "")


def _compare_func(output, expect, compare_tolerance=None):
    return compare_tensor(output, expect, rtol=compare_tolerance, atol=compare_tolerance)


def compare_results(kernel_name, desc, input_for_mod, output_indexes, expect):
    """Helper function to compare result"""
    output = list(input_for_mod[i] for i in output_indexes)
    if isinstance(output, tuple) and len(output) > 0 and isinstance(output[-1], dict):
        output = output[0]
    output = output if isinstance(output, (list, tuple)) else [output]
    expect = expect if isinstance(expect, (list, tuple)) else [expect]
    output = list(output)
    expect = list(expect)
    compare_tolerance = get_compare_tolerance(desc, output_indexes)
    compare_res = list(map(_compare_func, output, expect, compare_tolerance))
    if not all(compare_res):
        logging.error("%s precision error", kernel_name)
    else:
        logging.info("%s precision correct", kernel_name)


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
    """clear tmp dirs"""
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
    process_map = {'cpu':'cpu', 'cuda':'gpu', 'aicore':'ascend'}
    desc_d = json.loads(desc)
    process = desc_d.get("process", None)
    if process is None:
        raise RuntimeError("Can't get process in the json desc")
    target = process_map.get(process)
    if target is None:
        raise RuntimeError(f"Can't get target for process({process}) in desc")
    return target


def _is_matmul_op(desc):
    json_obj = _get_json_dict(desc)
    for op in json_obj.get("op_desc", []):
        if op.get("name", "") in ("MatMul", "BatchMatMul"):
            return True
    return False


def run_a_kernel(desc, file_path, compile_args, static_desc=None):
    """function to run a single kernel"""
    is_dyn_shape = static_desc is not None
    backend = compile_args.backend
    if not backend:
        backend = _auto_get_target(desc)

    if backend == "ascend" and compile_args.akg_fusion and _is_matmul_op(desc):
        raise RuntimeError("MatMul is not supported on ascend backend with akg fusion")

    kernel_name = _get_kernel_name(desc)
    arch = _get_arch_name(desc)
    # Generate data
    input_for_mod, expect, output_indexes = gen_json_data(
        static_desc if is_dyn_shape else desc, with_compute=True)
    # Init MlirDriver
    mlir_driver = MlirDriver(kernel_name=kernel_name,
                             input_file=file_path,
                             output_dir=_get_kernel_meta_dir(),
                             backend=backend,
                             llvm_tools_dir=os.getenv("LLVM_HOME", ""),
                             dynamic_shape=is_dyn_shape,
                             dump_ir=compile_args.dump_ir,
                             mlir_timing=compile_args.mlir_timing,
                             repo_path=compile_args.repo_path,
                             profiling_trails=compile_args.prof_trails,
                             runtime_provider="MLIR",
                             enable_loop_fusion=compile_args.akg_fusion,
                             arch=arch)

    mlir_driver.compile()
    mlir_driver.run(input_for_mod, output_indexes)

    for idx, d in enumerate(expect):
        expect[idx] = d.astype(np.float32) if d.dtype.name == "bfloat16" else d
    compare_results(kernel_name, desc, input_for_mod, output_indexes, expect)

    if compile_args.clear_tmp:
        _clear_tmp_dirs(kernel_name)


def _get_compute(desc):
    desc_d = json.loads(desc)
    compute = []
    for op in desc_d.get("op_desc", []):
        op_name = op.get("name", "")
        compute.append(op_name)
    return compute


def _get_input_shape(desc):
    """get input shape."""
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
    """run single info."""
    dump_dir = pathlib.Path(_get_kernel_meta_dir())
    info_file = pathlib.Path(file_path)
    input_file = file_path

    input_file = pathlib.Path(file_path)
    if info_file.suffix == ".mlir":
        kernel_name = find_first_func_name(info_file)
        if not kernel_name:
            raise RuntimeError(f"Cannot find `func.func @NAME(` in: {file_path}")
        info_file = dump_dir / f"{kernel_name}_.info"
        run_torch_mlir_to_json(input_file, info_file)
        if compile_args.akg_fusion:
            input_file = dump_dir / f"{kernel_name}_linalg.mlir"
            run_torch_mlir_to_linalg_on_tensors(file_path, input_file)
        else:
            input_file = dump_dir / f"{kernel_name}_hfusion.mlir"
            get_named_op_str(file_path, input_file, f"{kernel_name}", False, str(dump_dir))

    with open(info_file, "r", encoding='utf-8') as f:
        desc = f.read()
        kernel_name = _get_kernel_name(desc)
        input_shape, is_dyn_shape = _get_input_shape(desc)
        static_desc = None
        if is_dyn_shape:
            static_info_path = info_file.with_name(info_file.stem + "_static" + info_file.suffix)
            if static_info_path.exists():
                raise ValueError("Dynamic shape info must come with static shape info.")
            static_desc = static_info_path.read_text('utf-8')
        if compile_args.profiling:
            logging.info("profiling %s", kernel_name)
            logging.info("input_shape = %s compute %s", input_shape, _get_compute(desc))
            logging.info("input_dtype = %s", _get_input_dtype(desc))

        try:
            logging.info("Start running %s", kernel_name)
            run_a_kernel(desc, str(input_file), compile_args, static_desc=static_desc)
        except ValueError as e:
            logging.error("run %s get an error, error message: %s", kernel_name, e)
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
        """Record cycle data to the profiling result file."""
        if os.environ.get(PERFORMANCE_TEST_FILE):
            result_file = os.environ.get(PERFORMANCE_TEST_FILE)
            with os.fdopen(os.open(result_file, os.O_WRONLY | os.O_CREAT, 0o644), "a") as f:
                f.write(f"{format(cycle)}\n")

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
            with os.fdopen(os.open(result_file, os.O_WRONLY | os.O_CREAT, 0o644), "a") as f:
                f.write(f"{format(get_core_num())}; ")


def main(args=None):
    """Main entry point for running kernel benchmark tests."""
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
    parser.add_argument("-dump", "--dump_ir", type=bool, default=False)
    parser.add_argument("-time", "--mlir_timing", type=bool, default=False)
    parser.add_argument("-r", "--replace_dso", type=bool, default=False)
    parser.add_argument("-repo", "--repo_path", type=str, default="")
    parser.add_argument("-af", "--akg_fusion", type=distutils.util.strtobool, default=True)
    args = parser.parse_args()
    _create_dirs()
    if args.dir:
        files = [
            f for f in os.listdir(args.dir)
            if not f.endswith("_static.json") and not f.endswith("_static.info")
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
        logging.info("Finish profiling, total file %s", len(files))
        if args.ci_test and not _has_fail():
            logging.info("dir test success")
    else:
        if args.ci_test and args.file.endswith("_static.info"):
            logging.info("Skip static info")
            logging.info("precision correct")
        else:
            _run_single_file(args.file, args)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError) as e:
        logging.error("Unexpected error: %s", e)
        sys.exit(1)
