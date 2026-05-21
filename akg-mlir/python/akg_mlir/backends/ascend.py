# Copyright 2025-2026 Huawei Technologies Co., Ltd
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
import re
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
from akg.akgAscendLaunch import akg_ascend_run
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


# Matches device func attrs like: hacc.block_dim = 40 : i64
_BLOCK_DIM_MLIR_RE = re.compile(r"hacc\.block_dim\s*=\s*(\d+)\s*:\s*i64")


def get_block_dim_from_mlir(mlir_path):
    """Read ``hacc.block_dim`` from an MLIR file on disk.

    Args: mlir_path (str): Path to the MLIR file.

    Returns: int: Parsed block dimension.

    Raises:
        FileNotFoundError: If ``mlir_path`` is not a file.
        ValueError: If no ``hacc.block_dim = <n> : i64`` line is found.
    """
    if not os.path.isfile(mlir_path):
        raise FileNotFoundError(f"MLIR file not found: {mlir_path}")
    with open(mlir_path, "r", encoding="utf-8") as f:
        text = f.read()
    match = _BLOCK_DIM_MLIR_RE.search(text)
    if not match:
        raise ValueError(
            f"No hacc.block_dim = <n> : i64 attribute found in {mlir_path}"
        )
    return int(match.group(1))

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

def akg_opt(
    input_file,
    output_file,
    dyn_shape=False,
    enable_loop_fusion=True,
    enable_hivm_compile=False,
    arch=None,
    dump_ir=False,
    mlir_timing=False,
    dump_ir_path=None,
):
    """
    Run akg-opt to optimize MLIR for Ascend backend.

    Args:
        input_file: Input MLIR file path
        output_file: Output MLIR file path
        dyn_shape: Whether to enable dynamic shape optimization
        enable_loop_fusion: Whether to enable loop fusion
        arch: Architecture specification (optional)
        dump_ir: Whether to dump IR after all passes
        mlir_timing: Whether to print every pass time
        dump_ir_path: Path to dump log file (optional)

    Returns:
        subprocess.CompletedProcess result

    Raises:
        RuntimeError: If akg-opt execution fails
    """
    dump_ir = dump_ir or (os.environ.get("AKG_DUMP_IR", "0") == "1")

    akg_opt_path = get_akg_opt_path()

    # Build ascend-opt option
    ascend_opt_option = "--ascend-opt"
    options = []

    if dyn_shape:
        options.append("dynamic-shape=true")
    if not enable_loop_fusion:
        options.append("enable-loop-fusion=false")
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
            if not dump_ir_path:
                output_dir = os.path.dirname(output_file)
                log_file_name = os.path.basename(output_file) + ".log"
                dump_ir_path = os.path.join(output_dir, log_file_name)
            with os.fdopen(os.open(dump_ir_path, os.O_WRONLY | os.O_CREAT, 0o755), "w") as f:
                f.write(result.stderr)
        logging.debug("akg-opt pipeline success")
        return result
    except subprocess.CalledProcessError as e:
        logging.error("run akg-opt failed! cmd:\n %s \nerror message:\n %s", e.cmd, e.stderr)
        raise RuntimeError("mlir pipeline failed in case: " + os.path.basename(input_file) + "!\n") from e


def check_ascend_compile(output_file):
    """Assert bishengir output exists: non-empty ``.o`` or ``.so`` for the ``-o`` path.

    Checks ``-o`` target first, then same-stem ``.so``/``.o`` alternate.

    Args:
        output_file: Path passed to bishengir ``-o`` (same stem as ``.o``/``.so``).

    Raises:
        RuntimeError: If neither artifact exists or both are empty.
    """
    root, ext = os.path.splitext(output_file)
    ext = ext.lower()
    cands = [output_file]
    if ext == ".so":
        cands.append(root + ".o")
    elif ext == ".o":
        cands.append(root + ".so")
    for p in cands:
        if os.path.isfile(p) and os.path.getsize(p) > 0:
            return
    logging.error("bishengir-compile: .o/.so not found at %s", output_file)
    raise RuntimeError(
        "generate ascend binary: .o or .so not found at " + output_file
    ) from None


def bisheng_compile(input_file,
                    output_file,
                    enable_hfusion_compile=False,
                    enable_hivm_compile=True,
                    enable_auto_multi_buffer=True,
                    enable_bin_relocation=False,
                    block_dim=40,
                    dump_ir=False,
                    dump_ir_path=None):
    """Using bishengir-compile to generate Ascend binary.

    Args:
        input_file: Input MLIR file path
        output_file: Output file
        enable_hfusion_compile: Whether hfusion compile pipeline
        enable_hivm_compile: Whether hivm compile pipeline
        enable_auto_multi_buffer: Whether enabled auto mulit buffer in hivm compile pipeline
        enable_bin_relocation: Whether enabled relocation,
        dump_ir: Whether to dump bishengir-compile stderr log
        dump_ir_path: Optional path to dump bishengir-compile log.

    """
    dump_ir = dump_ir or (os.environ.get("AKG_DUMP_IR", "0") == "1")
    output_dir = os.path.dirname(output_file)


    compile_cmd = [
        "bishengir-compile",
        input_file,
        "-o",
        output_file,
        f"-block-dim={block_dim}",
        "-disable-auto-cv-work-space-manage"
    ]

    if enable_hfusion_compile:
        compile_cmd.append("-enable-hfusion-compile")
    if enable_hivm_compile:
        compile_cmd.append("-enable-hivm-compile")
    if enable_auto_multi_buffer:
        compile_cmd.append("-enable-auto-multi-buffer")
    if enable_bin_relocation:
        compile_cmd.append("-enable-bin-relocation")
    if dump_ir:
        compile_cmd.append("-mlir-print-ir-after-all")


    logging.debug("exec command: %s", compile_cmd)
    try:
        result = subprocess.run(
            compile_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        if dump_ir and dump_ir_path:
            if not dump_ir_path:
                log_file_name = os.path.basename(output_file) + ".log"
                dump_ir_path = os.path.join(output_dir, log_file_name)
            with os.fdopen(os.open(dump_ir_path, os.O_WRONLY | os.O_CREAT, 0o755), "w") as f:
                f.write(result.stderr)
        check_ascend_compile(output_file)
    except subprocess.CalledProcessError as e:
        logging.error("run bishengir-compile failed! cmd:\n %s \nerror message:\n %s", e.cmd, e.stderr)
        if dump_ir and dump_ir_path:
            with os.fdopen(os.open(dump_ir_path, os.O_WRONLY | os.O_CREAT, 0o755), "w") as f:
                f.write(str(e))
        raise RuntimeError("generate ascend binary: " + input_file + "!\n") from e
    logging.debug("generate ascend binary success")

    kernel_name = os.path.splitext(os.path.basename(output_file))[0]
    if kernel_name.startswith("lib"):
        kernel_name = kernel_name[3:]
    dump_ascend_meta_data(output_dir, kernel_name, block_dim=block_dim)

def ascend_compile(
    input_file,
    output_file,
    dyn_shape=False,
    enable_loop_fusion=True,
    arch=None,
    dump_ir=False,
    mlir_timing=False,
    dump_ir_path=None):
    """
    Run MLIR compile for Ascend backend.

    Args:
        input_file: Input MLIR file path
        output_file: Output ascend binary file
        dyn_shape: Whether to enable dynamic shape optimization
        arch: Architecture specification (optional)
        dump_ir: Whether to dump IR after all passes
        mlir_timing: Whether to print every pass time
        dump_ir_path: Path to dump log file (optional)

    Returns:
        subprocess.CompletedProcess result

    Raises:
        RuntimeError: If compile execution fails
    """

    output_dir = os.path.dirname(output_file)
    opt_file_name = os.path.basename(output_file) + "_opt"
    opt_file = os.path.join(output_dir, f"{opt_file_name}.mlir")
    akg_opt(
        input_file=input_file,
        output_file=opt_file,
        dyn_shape=dyn_shape,
        enable_loop_fusion=enable_loop_fusion,
        arch=arch,
        dump_ir=dump_ir,
        mlir_timing=mlir_timing,
        dump_ir_path=dump_ir_path,
    )
    block_dim = get_block_dim_from_mlir(opt_file)
    bisheng_compile(
        opt_file,
        output_file,
        enable_hfusion_compile=not enable_loop_fusion,
        block_dim=block_dim,
        dump_ir=dump_ir,
        dump_ir_path=dump_ir_path,
    )


def _get_device_shape_for_dyn(data, kernel_name, work_dir):
    """Get device shape for dynamic shape. Returns None if not dynamic."""
    cur_dir = os.path.dirname(work_dir) if work_dir else ""
    try:
        device_shape, _, _ = get_device_shape(data, kernel_name, True, cur_dir=cur_dir)
        return device_shape
    except (ValueError, RuntimeError):
        return [
            d.shape if hasattr(d, 'shape') and not isinstance(d, (int, float, bool)) else ()
            for d in data
        ]


def _build_output_idx_set(output_indexes, data_len):
    """Build set of output indices from output_indexes list."""
    output_idx_set = set()
    for idx in output_indexes or []:
        output_idx_set.add(idx if idx >= 0 else idx + data_len)
    return output_idx_set


def _process_numpy_arg(d, data_idx, data, is_output, device_shape):
    """Process numpy array: bf16, float16, contiguous, shape.
    Returns (processed_d, is_bf16, shape). May modify data[data_idx] in place.
    """
    is_bf16 = d.dtype.name == "bfloat16"
    if is_bf16 and not is_output:
        d = d.astype(np.float32)
        data[data_idx] = d
    if d.dtype == np.float16 or (hasattr(d.dtype, 'char') and d.dtype.char in ('e', 'E')):
        d = d.view(np.uint16)
    if not is_output and not d.flags.c_contiguous:
        d = np.ascontiguousarray(d)
        data[data_idx] = d
    if device_shape is not None and data_idx < len(device_shape):
        s = device_shape[data_idx]
        shape = list(s) if isinstance(s, (tuple, list)) else []
    else:
        shape = list(d.shape)
    return (d, is_bf16, shape)


def _process_tensor_arg(d):
    """Process tensor (non-numpy): is_bf16, shape. Returns (is_bf16, shape)."""
    is_bf16 = hasattr(d, 'dtype') and str(d.dtype) == 'torch.bfloat16'
    shape = list(d.shape) if hasattr(d, 'shape') else []
    return (is_bf16, shape)


def _prepare_ascend_args(data, kernel_name, output_indexes, is_dyn_shape, work_dir=""):
    """Preprocess in Python: bf16, output_indexes, device_shape.
    Returns list of (numpy_or_tensor, is_output, is_bf16, shape) - still passing numpy/tensor.
    """
    if len(data) == 0:
        return []

    device_shape = _get_device_shape_for_dyn(data, kernel_name, work_dir) if is_dyn_shape else None
    output_idx_set = _build_output_idx_set(output_indexes, len(data))

    result = []
    for data_idx, d in enumerate(data):
        if isinstance(d, (int, float, bool, complex)):
            result.append((d, False, False, None))
            continue

        is_output = data_idx in output_idx_set
        if isinstance(d, np.ndarray):
            d, is_bf16, shape = _process_numpy_arg(d, data_idx, data, is_output, device_shape)
        else:
            is_bf16, shape = _process_tensor_arg(d)
        result.append((d, is_output, is_bf16, shape))
    return result


def benchmark_launch(
    work_dir,
    kernel_name,
    device_id,
    is_dyn_shape,
    *input_for_mod,
    use_mem_pool=False,
    stream=None,
    output_indexes=None
):
    """Launch .so file by akg_ascend_backend.

    All preprocessing (bf16, output_indexes, device_shape) done in Python.
    Interface: *input_for_mod (numpy/tensor), stream. No kwargs.
    """
    data_list = list(input_for_mod)
    processed = _prepare_ascend_args(
        data_list, kernel_name, output_indexes or [], is_dyn_shape, work_dir
    )

    akg_ascend_run(
        work_dir,
        kernel_name,
        device_id,
        is_dyn_shape,
        use_mem_pool,
        processed_args=processed,
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
    logging.debug("dump ascend meta data:")
    title_dict = {}
    # ascend info
    set_ascend_info("VectorCore", title_dict)
    title_dict["kernelName"] = kernel_name
    # thread info
    title_dict["blockDim"] = block_dim
    # bin file info
    bin_file_suffix = ".o"
    title_dict["binFileSuffix"] = bin_file_suffix
    bin_file_name = kernel_name
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
