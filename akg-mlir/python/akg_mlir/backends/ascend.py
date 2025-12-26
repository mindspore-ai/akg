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
""" akg launch and compile utils """
import os
import sys
import ctypes
import logging
import subprocess
import numpy as np

flags = sys.getdlopenflags()
sys.setdlopenflags(flags | os.RTLD_GLOBAL)
# pylint: disable=wrong-import-position
# pylint: disable=wrong-import-order
from akg import akgAscendLaunch
sys.setdlopenflags(flags)
from akg.utils.dynamic_utils import get_device_shape

def get_akg_opt_path(akg_tools_dir=None):
    """Get the path of akg-opt executable."""
    if akg_tools_dir is None:
        # Default to the directory containing this file
        akg_tools_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(akg_tools_dir, "bin", "akg-opt")

def run_akg_opt(
    input_file,
    output_file,
    akg_tools_dir=None,
    dyn_shape=False,
    enable_akg_loop_fusion=False,
    arch=None,
    dump_ir=False,
    dump_log_path=None
):
    """
    Run akg-opt to optimize MLIR for Ascend backend.

    Args:
        input_file: Input MLIR file path
        output_file: Output MLIR file path
        akg_tools_dir: Directory containing akg tools (default: auto-detect)
        dyn_shape: Whether to enable dynamic shape optimization
        enable_akg_loop_fusion: Whether to enable akg loop fusion
        arch: Architecture specification (optional)
        dump_ir: Whether to dump IR after all passes
        dump_log_path: Path to dump log file (optional)

    Returns:
        subprocess.CompletedProcess result

    Raises:
        RuntimeError: If akg-opt execution fails
    """
    akg_opt_path = get_akg_opt_path(akg_tools_dir)

    # Build ascend-opt option
    ascend_opt_option = "--ascend-opt"
    options = []

    if dyn_shape:
        options.append("dynamic-shape=true")
    if enable_akg_loop_fusion:
        options.append("enable-akg-loop-fusion=1")
    if arch:
        options.append(f"arch={arch}")

    if options:
        ascend_opt_option += "=" + " ".join(options)

    cmd = [akg_opt_path, input_file, ascend_opt_option, "-o", output_file]

    if dump_ir:
        cmd.append("--mlir-print-ir-after-all")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if dump_ir and dump_log_path:
            with os.fdopen(os.open(dump_log_path, os.O_WRONLY | os.O_CREAT, 0o755), "w") as f:
                f.write(result.stderr)
        logging.info("akg-opt pipeline success")
        return result
    except subprocess.CalledProcessError as e:
        logging.error("run akg-opt failed! cmd:\n %s \nerror message:\n %s", e.cmd, e.stderr)
        raise RuntimeError("mlir pipeline failed in case: " + os.path.basename(input_file) + "!\n") from e


def run_mlir_ascend_pipeline(
    input_file,
    output_file,
    akg_tools_dir=None,
    dyn_shape=False,
    enable_akg_loop_fusion=False,
    arch=None,
    dump_ir=False,
    dump_log_path=None,
):
    """
    Run complete MLIR pipeline for Ascend: akg-opt.

    Args:
        input_file: Input MLIR file path
        output_file: Final output MLIR file path
        akg_tools_dir: Directory containing akg tools (default: auto-detect)
        dyn_shape: Whether to enable dynamic shape optimization
        enable_akg_loop_fusion: Whether to enable akg loop fusion
        arch: Architecture specification (optional)
        dump_ir: Whether to dump IR after all passes
        dump_log_path: Path to dump log file (optional)
    Returns:
        Path to final output file
    """
    run_akg_opt(
        input_file=input_file,
        output_file=output_file,
        akg_tools_dir=akg_tools_dir,
        dyn_shape=dyn_shape,
        enable_akg_loop_fusion=enable_akg_loop_fusion,
        arch=arch,
        dump_ir=dump_ir,
        dump_log_path=dump_log_path
    )
    return output_file

def ascend_compile(input_file, output_so_path):
    """ using bisheng-compile """
    compile_cmd = [
        "bishengir-compile",
        input_file,
        "-enable-hfusion-compile=false",
        "-enable-hivm-compile=true",
        "-enable-bin-relocation=false",
        "-block-dim=40",
        "-enable-auto-multi-buffer=true",
        "-o",
        output_so_path
    ]
    logging.info("exec command: %s", compile_cmd)
    subprocess.run(
        compile_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True
    )


def transform_data_to_ascend(
    data,
    kernel_name,
    output_indexes,
    is_dyn_shape=False,
    backend="ascend",
    is_profile_params=False
):
    """ transform tensor input data to ctypes for ascend """
    data_ctypes = []
    if len(data) == 0:
        # dynamic shape info cannot generate inputs while compilation
        return data_ctypes

    device_shape, _, _ = get_device_shape(
        data, kernel_name, is_dyn_shape and not is_profile_params
    )

    output_idx_set = []
    for output_idx in output_indexes:
        if output_idx >= 0:
            output_idx_set.append(output_idx)
        else:
            output_idx_set.append(output_idx + len(data))
    output_idx_set = set(output_idx_set)
    for data_idx, d in enumerate(data):
        data_shape = np.array(device_shape[data_idx])
        data_bytes = d.nbytes
        is_numpy_bf16 = False
        is_numpy_output = False
        if isinstance(d, int):
            data_ctypes.append(ctypes.c_int(d))
        elif isinstance(d, np.ndarray):
            if data_idx in output_idx_set:
                is_numpy_output = True
            if d.dtype.name == "bfloat16":
                d = d.astype(np.float32)
                data[data_idx] = d
                is_numpy_bf16 = True

        ascend_tensor_obj = akgAscendLaunch.AscendTensorObjStructPyTorch()
        ascend_tensor_obj.tensor_info = d
        ascend_tensor_obj.shape_info = data_shape
        ascend_tensor_obj.nbytes = data_bytes
        ascend_tensor_obj.is_output = is_numpy_output
        ascend_tensor_obj.is_bf16 = is_numpy_bf16
        data_ctypes.append(ascend_tensor_obj)

    return data_ctypes


def launch(
    output_so_dir,
    kernel_name,
    device_id,
    is_dyn_shape,
    *input_for_mod_ctypes
):
    """ launch .so file by akg_ascend_backend """
    akgAscendLaunch.akg_ascend_run(
        output_so_dir,
        kernel_name,
        device_id,
        is_dyn_shape,
        *input_for_mod_ctypes
    )
