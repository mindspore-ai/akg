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
import time
import distutils
import numpy as np

from akg import MlirDriver
from akg.mlir_driver import get_kernel_meta_path
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


def run_pypto_kernel(input_file, output_dir, kernel_name, arch, buffers, output_indexes, options):
    """Compile and run once, copying only result buffers back for AKG comparison.

    The compiled parameter contract supplies physical shapes and dtypes. In
    particular BF16 must never be inferred from an FP32 NumPy reference.
    Inputs and the NumPy golden stay in memory throughout the benchmark.
    """
    # Load optional PTO dependencies only when this backend is selected.
    # pylint: disable=import-outside-toplevel
    import torch
    import torch_npu  # pylint: disable=unused-import
    from akg.backends.ascend import ascend_compile
    from pypto.ir.compiled_artifact import CompiledArtifact
    from pypto.ir.param_info import _to_torch_dtype
    # pylint: enable=import-outside-toplevel

    compiled = ascend_compile(
        str(input_file), str(output_dir), kernel_name=kernel_name, arch=arch or None,
        backend="pypto", opt_level=options.opt_level, enable_loop_fusion=True,
        dump_ir=options.dump_ir, mlir_timing=options.mlir_timing,
        dump_ir_path=str(pathlib.Path(output_dir) / f"{kernel_name}_opt.log"),
        num_outputs=len(output_indexes))
    artifact = CompiledArtifact.from_dir(compiled.path)
    parameters = artifact.metadata.parameters
    if len(parameters) < len(buffers):
        raise ValueError(f"{kernel_name}: compiled parameters={len(parameters)}, "
                         f"benchmark buffers={len(buffers)}")
    device_id = getattr(options, "device_id", None)
    if device_id is None:
        device_id = int(os.environ.get("DEVICE_ID", "0"))
    torch.npu.set_device(device_id)
    arguments = []
    for index, parameter in enumerate(parameters):
        if index >= len(buffers):
            # Lowering appends private reduction workspaces to the ABI. These
            # are not graph inputs or results and need their physical size.
            dtype = _to_torch_dtype(parameter.dtype)
            if parameter.shape is None or dtype is None:
                raise ValueError(f"invalid workspace parameter: {parameter}")
            arguments.append(torch.empty(tuple(parameter.shape), dtype=dtype,
                                         device=f"npu:{device_id}"))
            continue
        buffer = buffers[index]
        if parameter.shape is None:
            arguments.append(np.asarray(buffer).item())
            continue
        array = np.asarray(buffer)
        if array.dtype.name == "bfloat16":
            array = array.astype(np.float32)
        dtype = _to_torch_dtype(parameter.dtype)
        if dtype is None:
            raise ValueError(f"unsupported benchmark dtype: {parameter.dtype}")
        if array.dtype.name.startswith("float8_"):
            if dtype != getattr(torch, array.dtype.name):
                raise ValueError(f"FP8 buffer dtype {array.dtype} does not match compiled dtype {dtype}")
            # PyTorch cannot import NumPy FP8 directly; preserve its storage bits.
            tensor = torch.from_numpy(np.array(array, copy=True).view(np.uint8)).view(dtype)
        else:
            tensor = torch.from_numpy(np.array(array, copy=True)).to(dtype=dtype)
        # AKG represents scalar tensors as [1]; use the compiled ABI shape.
        tensor = tensor.reshape(tuple(parameter.shape)).to(f"npu:{device_id}")
        arguments.append(tensor)
    stream = torch.npu.current_stream(device_id)
    compiled.launcher(*arguments, stream=stream.npu_stream)
    stream.synchronize()
    for index in output_indexes:
        # Output indices refer to graph buffers, before appended workspaces.
        argument_index = index if index >= 0 else len(buffers) + index
        result = arguments[argument_index].detach().cpu()
        if result.dtype == torch.bfloat16 or str(result.dtype).startswith("torch.float8_"):
            result = result.float()
        buffers[index] = result.numpy().reshape(np.asarray(buffers[index]).shape)


def run_a_kernel(desc, file_path, compile_args, static_desc=None):
    """function to run a single kernel"""
    is_dyn_shape = static_desc is not None
    backend = compile_args.backend
    if not backend:
        backend = _auto_get_target(desc)

    if backend == "ascend" and compile_args.akg_fusion and _is_matmul_op(desc):
        raise RuntimeError("MatMul is not supported on ascend backend with akg fusion")

    kernel_name = _get_kernel_name(desc)
    dump_dir = pathlib.Path(get_kernel_meta_path()) / kernel_name
    dump_dir.mkdir(parents=True, exist_ok=True)
    arch = getattr(compile_args, "arch", "") or _get_arch_name(desc)
    # Generate data
    input_for_mod, expect, output_indexes = gen_json_data(static_desc if is_dyn_shape else desc, with_compute=True)
    if getattr(compile_args, "ascend_backend", "bisheng") == "pypto":
        if backend != "ascend" or is_dyn_shape or not compile_args.akg_fusion:
            raise ValueError("PyPTO requires static shapes, --backend ascend and --akg_fusion 1")
        run_pypto_kernel(file_path, dump_dir, kernel_name, arch, input_for_mod,
                         output_indexes, compile_args)
    else:
        # Init MlirDriver
        mlir_driver = MlirDriver(kernel_name=kernel_name,
                                 input_file=file_path,
                                 output_dir=dump_dir,
                                 backend=backend,
                                 llvm_tools_dir=os.getenv("LLVM_HOME", ""),
                                 dynamic_shape=is_dyn_shape,
                                 dump_ir=compile_args.dump_ir,
                                 mlir_timing=compile_args.mlir_timing,
                                 repo_path=compile_args.repo_path,
                                 profiling_trails=compile_args.prof_trails,
                                 runtime_provider="MLIR",
                                 enable_loop_fusion=compile_args.akg_fusion,
                                 opt_level=compile_args.opt_level,
                                 arch=arch)

        mlir_driver.compile()
        mlir_driver.run(input_for_mod, output_indexes)

    for idx, d in enumerate(expect):
        expect[idx] = d.astype(np.float32) if d.dtype.name == "bfloat16" else d
    compare_results(kernel_name, desc, input_for_mod, output_indexes, expect)


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
    info_file = pathlib.Path(file_path)
    input_file = file_path

    input_file = pathlib.Path(file_path)
    if info_file.suffix == ".mlir":
        kernel_name = find_first_func_name(info_file)
        dump_dir = pathlib.Path(get_kernel_meta_path()) / kernel_name
        dump_dir.mkdir(parents=True, exist_ok=True)
        if not kernel_name:
            raise RuntimeError(f"Cannot find `func.func @NAME(` in: {file_path}")
        info_file = dump_dir / f"{kernel_name}_.info"
        run_torch_mlir_to_json(input_file, info_file, backend=getattr(compile_args, "ascend_backend", None))
        if compile_args.akg_fusion:
            input_file = dump_dir / f"{kernel_name}_linalg.mlir"
            run_torch_mlir_to_linalg_on_tensors(file_path, input_file)
        else:
            input_file = dump_dir / f"{kernel_name}_hfusion.mlir"
            get_named_op_str(file_path, input_file, f"{kernel_name}", False, str(dump_dir))

    with open(os.path.realpath(info_file), "r", encoding='utf-8') as f:
        desc = f.read()
        kernel_name = _get_kernel_name(desc)
        input_shape, is_dyn_shape = _get_input_shape(desc)
        static_desc = None
        if is_dyn_shape:
            static_info_path = info_file.with_name(info_file.stem + "_static" + info_file.suffix)
            if not static_info_path.exists():
                raise ValueError("Dynamic shape info must come with static shape info.")
            static_desc = static_info_path.read_text('utf-8')
        if compile_args.profiling:
            logging.info("profiling %s", kernel_name)
            logging.info("input_shape = %s compute %s", input_shape, _get_compute(desc))
            logging.info("input_dtype = %s", _get_input_dtype(desc))

        try:
            logging.info("Start running %s", kernel_name)
            run_a_kernel(desc, str(input_file), compile_args, static_desc=static_desc)
        except (ValueError, RuntimeError) as e:
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
    parser.add_argument("--ascend-backend", choices=["bisheng", "pypto"], default="bisheng",
                        help="Ascend compiler used for compilation and precision checking")
    parser.add_argument("--arch", default="", help="Override target architecture in the descriptor")
    parser.add_argument("--device-id", type=int, default=None, help="NPU device (defaults to DEVICE_ID or 0)")
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
    parser.add_argument("-O", "-ol", "--opt-level", dest="opt_level", type=int, default=2,
                        help="Ascend optimization level passed to --ascend-opt (default: 2). "
                             "Supports -O1/-O2 and -ol 1/-ol 2.")
    args = parser.parse_args()
    if args.device_id is not None:
        os.environ["DEVICE_ID"] = str(args.device_id)
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
            if not _run_single_file(args.file, args):
                return 1
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError) as e:
        logging.error("Unexpected error: %s", e)
        sys.exit(1)
