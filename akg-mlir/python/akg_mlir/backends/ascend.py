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
import hashlib
import json
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


def write_code(js_dict, fname):
    """
    Export kernel config files.

    Args:
        js_dict: dict of kernel information.
        fname: the name of json file to be generated.
    """
    if os.path.exists(fname):
        os.remove(fname)
    with os.fdopen(os.open(fname, os.O_WRONLY | os.O_CREAT, 0o400), "w") as f:
        json.dump(js_dict, f, sort_keys=True, indent=4, separators=(",", ":"))


def get_block_dim_from_arch(arch):
    """ Get block_dim from architecture name.
    When arch is None or empty, returns 40 (default for 910B4, same as original hardcoded value).
    """
    if not arch:
        return 40
    arch_str = str(arch).upper()
    if "910B4" in arch_str:
        return 40
    if "910B2" in arch_str:
        return 48
    error_msg = (f"Unsupported architecture: {arch}. "
                 f"Supported architectures: 910B4 (block_dim=40), 910B2 (block_dim=48)")
    raise ValueError(error_msg)

def set_ascend_info(core_type, title_dict):
    """Set ascend binary metadata (magic, coreType, etc.) in title_dict by core type.

    Args:
        core_type: Core type ("MIX", "AiCore", or "VectorCore")
        title_dict: Dict to update in-place with magic, coreType, etc.
    """
    if len(core_type) == 0:
        return
    if core_type == "MIX":
        title_dict["magic"] = "RT_DEV_BINARY_MAGIC_ELF"
        title_dict["coreType"] = "MIX"
        title_dict["intercoreSync"] = 1
        title_dict["taskRation"] = "1:2"
    elif core_type == "AiCore":
        title_dict["coreType"] = "AiCore"
        title_dict["magic"] = "RT_DEV_BINARY_MAGIC_ELF_AICUBE"
    elif core_type == "VectorCore":
        title_dict["coreType"] = "VectorCore"
        title_dict["magic"] = "RT_DEV_BINARY_MAGIC_ELF_AIVEC"


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
    enable_loop_fusion=False,
    arch=None,
    dump_ir=False,
    mlir_timing=False,
    dump_log_path=None,
):
    """
    Run akg-opt to optimize MLIR for Ascend backend.

    Args:
        input_file: Input MLIR file path
        output_file: Output MLIR file path
        akg_tools_dir: Directory containing akg tools (default: auto-detect)
        dyn_shape: Whether to enable dynamic shape optimization
        enable_loop_fusion: Whether to enable loop fusion
        arch: Architecture specification (optional)
        dump_ir: Whether to dump IR after all passes
        mlir_timing: Whether to print every pass time
        dump_log_path: Path to dump log file (optional)

    Returns:
        subprocess.CompletedProcess result

    Raises:
        RuntimeError: If akg-opt execution fails
    """
    dump_ir = dump_ir or (os.environ.get("AKG_DUMP_IR", "0") == "1")

    akg_opt_path = get_akg_opt_path(akg_tools_dir)

    # Build ascend-opt option
    ascend_opt_option = "--ascend-opt"
    options = []

    if dyn_shape:
        options.append("dynamic-shape=true")
    if not enable_loop_fusion:
        options.append("enable-loop-fusion=0")
    if arch:
        options.append(f"arch={arch}")

    if options:
        ascend_opt_option += "=" + " ".join(options)

    cmd = [akg_opt_path, input_file, ascend_opt_option, "-o", output_file]

    if dump_ir:
        cmd.append("--mlir-print-ir-after-all")
    if mlir_timing:
        cmd.append("--mlir-timing")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if dump_ir:
            if not dump_log_path:
                output_dir = os.path.dirname(output_file)
                base_name = os.path.basename(output_file)
                kernel_name, _ = os.path.splitext(base_name)
                if kernel_name.endswith("_out"):
                    kernel_name = kernel_name[: -len("_out")]
                dump_log_path = os.path.join(output_dir, kernel_name + "_dump_ascend_state1.log")
            with os.fdopen(os.open(dump_log_path, os.O_WRONLY | os.O_CREAT, 0o755), "w") as f:
                f.write(result.stderr)
        logging.info("akg-opt pipeline success")
        return result
    except subprocess.CalledProcessError as e:
        logging.error("run akg-opt failed! cmd:\n %s \nerror message:\n %s", e.cmd, e.stderr)
        raise RuntimeError("mlir pipeline failed in case: " + os.path.basename(input_file) + "!\n") from e

def ascend_compile(input_file, output_so_path, block_dim, enable_loop_fusion=True, dump_ir=False, dump_log_path=None):
    """Using bishengir-compile to generate Ascend binary.

    Args:
        input_file: Input MLIR file path
        output_so_path: Output .so file path
        block_dim: Block dimension
        enable_loop_fusion: Whether loop fusion is enabled in pipeline
        dump_ir: Whether to dump bishengir-compile stderr log
        dump_log_path: Optional path to dump bishengir-compile stderr log.
    """
    dump_ir = dump_ir or (os.environ.get("AKG_DUMP_IR", "0") == "1")

    compile_cmd = [
        "bishengir-compile",
        input_file,
        "-enable-hivm-compile=true",
        "-enable-bin-relocation=false",
        f"-block-dim={block_dim}",
        "-enable-auto-multi-buffer=true",
        "-o",
        output_so_path
    ]

    if enable_loop_fusion:
        compile_cmd.append("-enable-hfusion-compile=false")
    else:
        compile_cmd.append("-enable-hfusion-compile=true")

    logging.info("exec command: %s", compile_cmd)
    try:
        result = subprocess.run(
            compile_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        if dump_ir and dump_log_path:
            with os.fdopen(os.open(dump_log_path, os.O_WRONLY | os.O_CREAT, 0o755), "w") as f:
                f.write(result.stderr)
    except subprocess.CalledProcessError as e:
        logging.error("run bishengir-compile failed! cmd:\n %s \nerror message:\n %s", e.cmd, e.stderr)
        if dump_ir and dump_log_path:
            with os.fdopen(os.open(dump_log_path, os.O_WRONLY | os.O_CREAT, 0o755), "w") as f:
                f.write(str(e))
        raise RuntimeError("generate ascend binary: " + input_file + "!\n") from e
    logging.info("generate ascend binary success")

    output_dir = os.path.dirname(output_so_path)
    kernel_name = os.path.splitext(os.path.basename(output_so_path))[0]
    if kernel_name.startswith("lib"):
        kernel_name = kernel_name[3:]
    dump_ascend_meta_data(output_dir, kernel_name, block_dim=block_dim)


def transform_data_to_ascend(
    data,
    kernel_name,
    output_indexes,
    is_dyn_shape=False,
    backend="ascend",
    is_profile_params=False
):
    """ Transform tensor input data to ctypes for ascend.

    Args:
        data: List of input tensors or scalars
        kernel_name: Name of the kernel
        output_indexes: Indices of output tensors
        is_dyn_shape: Whether dynamic shape is used
        backend: Backend name
        is_profile_params: Whether profile params mode
    """
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
        if isinstance(d, (int, float, bool, complex)):
            data_ctypes.append(d)
            continue
        data_shape = np.array(device_shape[data_idx])
        data_bytes = d.nbytes
        is_numpy_bf16 = False
        is_numpy_output = False
        if isinstance(d, np.ndarray):
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
    *input_for_mod_ctypes,
    use_mem_pool=False,
    stream=None
):
    """ launch .so file by akg_ascend_backend

    Args:
        output_so_dir: Directory containing the .so and .json files
        kernel_name: Name of the kernel
        device_id: Device index
        is_dyn_shape: Whether dynamic shape is used
        *input_for_mod_ctypes: Tensor/ctypes inputs for the kernel
        use_mem_pool: Whether to use memory pool
        stream: Current stream from PTA; None for py_benchmark (AKG uses own stream)
    """
    akgAscendLaunch.akg_ascend_run(
        output_so_dir,
        kernel_name,
        device_id,
        is_dyn_shape,
        use_mem_pool,
        *input_for_mod_ctypes,
        stream=stream
    )


def dump_ascend_meta_data(output_dir, kernel_name, block_dim):
    """
    Dump ascend meta data to JSON file.
    
    Args:
        output_dir: Directory where the binary file and JSON file will be saved
        kernel_name: Name of the kernel
        block_dim: Block dimension
    """
    logging.info("dump ascend meta data:")
    title_dict = {}
    # ascend info
    set_ascend_info("VectorCore", title_dict)
    title_dict["kernelName"] = kernel_name
    # thread info
    title_dict["blockDim"] = block_dim
    # bin file info
    bin_file_suffix = ".so"
    title_dict["binFileSuffix"] = bin_file_suffix
    bin_file_name = "lib" + kernel_name
    title_dict["binFileName"] = bin_file_name
    # sha256
    buf_size = 64 * 1024  # once read 64kb
    sha256 = hashlib.sha256()
    kernel_file_name = os.path.join(output_dir, bin_file_name + bin_file_suffix)
    with open(kernel_file_name, "rb") as kf:
        while True:
            data = kf.read(buf_size)
            if not data:
                break
            sha256.update(data)
    title_dict["sha256"] = sha256.hexdigest()

    json_file = os.path.join(output_dir, kernel_name + ".json")
    write_code(title_dict, json_file)
