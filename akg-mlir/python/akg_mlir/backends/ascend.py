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

__all__ = [
    "REG_BASED_ARCH_PREFIXES",
    "write_code",
    "get_block_dim_from_mlir",
    "set_ascend_info",
    "get_akg_opt_path",
    "akg_opt",
    "check_ascend_compile",
    "bisheng_compile",
    "ascend_compile",
    "benchmark_launch",
    "dump_ascend_meta_data",
]

import os
import re
import hashlib
import pathlib
import json
import logging
import subprocess
import shlex
import runpy
from dataclasses import dataclass
from typing import Optional
import numpy as np

from ..ascend_launch import akg_ascend_run, get_device_function, get_host_functions, torch_launch
from ..utils.dynamic_utils import get_device_shape

REG_BASED_ARCH_PREFIXES = ("Ascend910_95", "Ascend950DT", "Ascend950PR")
# Matches device func attrs like: hacc.block_dim = 40 : i64
_BLOCK_DIM_MLIR_RE = re.compile(r"hacc\.block_dim\s*=\s*(\d+)\s*:\s*i64")


@dataclass
class KernelLaunch:
    """Launcher and cached paths returned to Kernel.compile."""
    path: str
    launcher: object
    binary_path: Optional[str] = None
    metadata_path: Optional[str] = None
    supports_autotune: bool = True


def _load_bisheng_launch(kernel_name, output_dir, dynamic=False, torch_path=None, **_options):
    """Reuse the existing binary launcher, loading functions on first execution."""
    binary_path = os.path.join(output_dir, f"lib{kernel_name}.so" if dynamic else f"{kernel_name}.o")
    metadata_path = os.path.join(output_dir, kernel_name + ".json")
    check_ascend_compile(binary_path)
    block_dim = int(json.loads(pathlib.Path(metadata_path).read_text(encoding="utf-8"))["blockDim"])
    functions = None

    def launch(*args, stream=None):
        nonlocal functions
        if functions is None:
            if dynamic:
                kernel, tiling, _, size = get_host_functions(kernel_name, binary_path)
                functions = (kernel, tiling, size)
            else:
                functions = (get_device_function(kernel_name, binary_path), 0, 0)
        if torch_path is None or stream is None:
            import torch_npu  # pylint: disable=import-outside-toplevel
            library_path = torch_path or str(pathlib.Path(torch_npu.__file__).parent)
            if stream is None:
                stream = torch_npu.npu.current_stream().npu_stream
        else:
            library_path = torch_path
        torch_launch(kernel_name, library_path, *functions, block_dim, stream, dynamic, *args)

    return {"path": binary_path, "launcher": launch, "binary_path": binary_path, "metadata_path": metadata_path}


def _load_pypto_launch(kernel_name, output_dir, num_outputs=None, **_options):
    """Load a prebuilt kernel and execute borrowed buffers on the caller's stream."""
    # Keep optional PyPTO dependencies out of the binary backend's import path.
    # pylint: disable=import-outside-toplevel
    from pypto._artifact_contract import ArtifactExecutionMode
    from pypto.ir.compiled_artifact import CompiledArtifact
    from pypto.ir.param_info import _DATATYPE_TO_CTYPE
    from pypto.jit._artifact_manifest import BuildKind
    from pypto.runtime._prebuilt import load_prebuilt
    from pypto.torch.interop import CallSignature
    from simpler.buffer import AddressSpace, TensorTransfer
    from simpler.task_interface import (CallConfig, ChipStorageTaskArgs, ChipTensor, _ChipWorker,
                                       _initialize_chip_runtime, scalar_to_uint64)
    from simpler_setup.runtime_builder import RuntimeBuilder
    from simpler_setup.torch_interop import torch_dtype_to_datatype

    build_dir = os.path.join(output_dir, kernel_name + "_build")
    artifact = CompiledArtifact.from_dir(build_dir)
    artifact.execution_capabilities.require(ArtifactExecutionMode.KERNEL)
    platform = artifact.platform
    actual_outputs = len(artifact.metadata.return_aliases)
    if num_outputs is not None and actual_outputs != num_outputs:
        raise ValueError(f"PyPTO kernel {kernel_name}: expected {num_outputs} outputs, got {actual_outputs}")
    chip_callable, runtime_name, runtime_config = load_prebuilt(
        artifact.output_dir, platform, BuildKind.SINGLE_CHIP)["."]
    signature = CallSignature(artifact.metadata.parameters, return_aliases=artifact.metadata.return_aliases,
                              caller_name=kernel_name, platform=platform)
    binaries = RuntimeBuilder(platform).get_binaries(runtime_name)
    config = CallConfig()
    config.aicpu_thread_num = runtime_config.get("aicpu_thread_num", 0)

    def launch(*args, stream):
        caller_stream = int(stream)
        frame = signature.describe_call(args)
        worker = _ChipWorker()
        try:
            _initialize_chip_runtime(worker, frame.device_index, binaries, borrowed_context_generation=1)
            handle = worker.register_callable(chip_callable)
            packed = ChipStorageTaskArgs()
            for argument in frame.tensors:
                tensor = argument.tensor
                packed.add_tensor(ChipTensor.make(tensor.data_ptr(), tuple(tensor.shape),
                                                 torch_dtype_to_datatype(tensor.dtype),
                                                 address_space=AddressSpace.DEVICE), transfer=TensorTransfer.NONE)
            for scalar in frame.scalars:
                packed.add_scalar(scalar_to_uint64(_DATATYPE_TO_CTYPE[str(scalar.dtype)](scalar.value)))
            # Wait while args still own the borrowed NPU storage.
            run = worker._submit_chip_run_direct(  # pylint: disable=protected-access
                handle, packed, config, caller_stream=caller_stream)
            run.wait()
        finally:
            worker.finalize()

    return {"path": build_dir, "launcher": launch, "supports_autotune": False}


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
        logging.error("get block number failed, No hacc.block_dim = <n> : i64 attribute found in %s, "
                      "using 48 as default", mlir_path)
        return 48
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


def get_akg_translate_path():
    """Use the translator shipped alongside akg-opt."""
    return str(pathlib.Path(get_akg_opt_path()).with_name("akg-translate"))


def _resolve_arch(arch):
    """Resolve an omitted architecture from the current Ascend device."""
    if not arch:
        try:
            import acl  # pylint: disable=import-outside-toplevel
            arch = acl.get_soc_name()
        except ImportError as err:
            raise ImportError("Please run 'source set_env.sh' in the CANN installation path.") from err
    return arch


def _build_ascend_opt_option(dyn_shape, enable_loop_fusion, arch, opt_level=2, backend="bisheng"):
    """Build the --ascend-opt option string, resolving arch if needed."""
    if backend not in ("pypto", "bisheng"):
        raise ValueError(f"Unknown Ascend backend: {backend}")
    options = []
    if backend == "pypto":
        options.append("enable-emitpypto=true")
    if dyn_shape:
        options.append("dynamic-shape=true")
    if not enable_loop_fusion:
        options.append("enable-loop-fusion=false")
    options.append(f"opt-level={opt_level}")
    options.append(f"arch={_resolve_arch(arch)}")
    ascend_opt_option = "--ascend-opt"
    if options:
        ascend_opt_option += "=" + " ".join(options)
    return ascend_opt_option


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
    opt_level=2,
    backend="bisheng",
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
        opt_level: Ascend optimization level passed to the C++ pipeline
        backend: Compiler backend, either "bisheng" or "pypto"

    Raises:
        RuntimeError: If akg-opt execution fails
    """
    dump_ir = dump_ir or (os.environ.get("AKG_DUMP_IR", "0") == "1")

    ascend_opt_option = _build_ascend_opt_option(
        dyn_shape, enable_loop_fusion, arch, opt_level=opt_level, backend=backend)

    cmd = [get_akg_opt_path(), input_file, ascend_opt_option, "-o", output_file]

    if dump_ir:
        cmd.append("--mlir-print-ir-after-all")
    if mlir_timing:
        cmd.append("--mlir-timing")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if dump_ir:
            if not dump_ir_path:
                dump_ir_path = pathlib.Path(output_file).with_suffix(".log")
            with os.fdopen(os.open(dump_ir_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644), "a") as f:
                f.write("COMMAND: " + shlex.join(map(str, cmd)) + "\n\n" + result.stderr)
        logging.debug("akg-opt pipeline success")
    except subprocess.CalledProcessError as e:
        if dump_ir:
            pathlib.Path(dump_ir_path or pathlib.Path(output_file).with_suffix(".log")).write_text(
                "COMMAND: " + shlex.join(map(str, cmd)) + "\n\n" + e.stderr, encoding="utf-8")
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


def _is_reg_based_arch(arch):
    return isinstance(arch, str) and any(arch.startswith(p) for p in REG_BASED_ARCH_PREFIXES)


def _build_compile_cmd(input_file, output_file, block_dim,
                       enable_hfusion_compile, enable_hivm_compile,
                       enable_auto_multi_buffer, enable_bin_relocation,
                       dump_ir, arch):
    """Build bishengir-compile command list."""
    if not arch:
        try:
            import acl  # pylint: disable=import-outside-toplevel
            arch = acl.get_soc_name()
        except ImportError:
            pass
    compile_cmd = [
        "bishengir-compile",
        input_file,
        "-o",
        output_file,
        f"-block-dim={block_dim}"
    ]

    if enable_hfusion_compile:
        compile_cmd.append("-enable-hfusion-compile")
    else:
        compile_cmd.append("-disable-hivm-tensor-compile")
    if enable_hivm_compile:
        compile_cmd.append("-enable-hivm-compile")
    if enable_auto_multi_buffer:
        compile_cmd.append("-enable-auto-multi-buffer")
    if not enable_bin_relocation:
        compile_cmd.append("-enable-bin-relocation=false")
    if dump_ir:
        compile_cmd.append("-mlir-print-ir-after-all")

    if _is_reg_based_arch(arch):
        compile_cmd.append(f"-target={arch}")
    else:
        compile_cmd.append("-disable-auto-cv-work-space-manage")

    return compile_cmd


def _write_dump_ir(dump_ir_path, content):
    """Write content to dump_ir log file."""
    with os.fdopen(os.open(dump_ir_path, os.O_WRONLY | os.O_CREAT, 0o644), "a") as f:
        f.write(content)


def _handle_dump_ir(result, dump_ir, dump_ir_path, output_file):
    """Handle dump_ir output after successful compile."""
    if not dump_ir:
        return
    if not dump_ir_path:
        dump_ir_path = pathlib.Path(output_file).with_suffix(".log")
    _write_dump_ir(dump_ir_path, result.stderr)


def bisheng_compile(input_file,
                    output_dir,
                    kernel_name,
                    enable_hfusion_compile=False,
                    enable_hivm_compile=True,
                    enable_auto_multi_buffer=True,
                    enable_bin_relocation=False,
                    dump_ir=False,
                    dump_ir_path=None,
                    arch=None,
                    dyn_shape=False,
                    ):
    """Using bishengir-compile to generate Ascend binary.

    Args:
        input_file: Input MLIR file path
        output_dir: Cache directory for compiled binaries and metadata
        kernel_name: Name of the kernel
        enable_hfusion_compile: Whether hfusion compile pipeline
        enable_hivm_compile: Whether hivm compile pipeline
        enable_auto_multi_buffer: Whether enabled auto mulit buffer in hivm compile pipeline
        enable_bin_relocation: Whether enabled relocation,
        dump_ir: Whether to dump bishengir-compile stderr log
        dump_ir_path: Optional path to dump bishengir-compile log.
        dyn_shape: Whether to generate a shared library in the cache directory.

    """
    output_file = os.path.join(output_dir, f"lib{kernel_name}.so" if dyn_shape else f"{kernel_name}.o")
    block_dim = get_block_dim_from_mlir(input_file)
    dump_ir = dump_ir or (os.environ.get("AKG_DUMP_IR", "0") == "1")

    compile_cmd = _build_compile_cmd(
        input_file, output_file, block_dim,
        enable_hfusion_compile, enable_hivm_compile,
        enable_auto_multi_buffer, enable_bin_relocation,
        dump_ir, arch
    )

    logging.debug("exec command: %s", compile_cmd)
    try:
        result = subprocess.run(
            compile_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        _handle_dump_ir(result, dump_ir, dump_ir_path, output_file)
        check_ascend_compile(output_file)
    except subprocess.CalledProcessError as e:
        logging.error("run bishengir-compile failed! cmd:\n %s \nerror message:\n %s", e.cmd, e.stderr)
        raise RuntimeError("generate ascend binary: " + input_file + "!\n") from e
    logging.debug("generate ascend binary success")

    dump_ascend_meta_data(output_dir, kernel_name, block_dim=block_dim)


def pypto_compile(input_file, output_dir, kernel_name, arch, dump_ir=False, **_options):
    """Translate optimized AKG IR and save the compiled PyPTO kernel."""
    # pylint: disable=import-outside-toplevel
    from pypto._artifact_contract import ArtifactExecutionMode
    from pypto.ir.compile import _compile_artifact
    from pypto.jit._artifact_manifest import BuildKind
    from pypto.runtime._prebuilt import BINARY_MANIFEST, prepare_prebuilt

    output_file = os.path.join(output_dir, kernel_name + ".py")
    command = [get_akg_translate_path(), input_file, "--mlir-to-pypto", "-o", output_file]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"akg-translate --mlir-to-pypto failed: {error.stderr}") from error
    except OSError as error:
        raise RuntimeError(f"Cannot execute {command[0]}; ensure akg-translate is installed and executable") from error
    function = runpy.run_path(output_file)[kernel_name]
    platform = "a5" if _is_reg_based_arch(arch) else "a2a3"
    build_dir = os.path.join(output_dir, kernel_name + "_build")
    # A failed rebuild must not pair fresh metadata with stale binaries.
    (pathlib.Path(build_dir) / BINARY_MANIFEST).unlink(missing_ok=True)
    artifact = _compile_artifact(
        function.specialize(), contract=ArtifactExecutionMode.KERNEL, platform=platform,
        output_dir=build_dir, dump_passes=dump_ir)
    prepare_prebuilt(pathlib.Path(artifact.output_dir), platform, BuildKind.SINGLE_CHIP)


def ascend_compile(
    input_file,
    output_dir,
    dyn_shape=False,
    enable_loop_fusion=True,
    arch=None,
    dump_ir=False,
    mlir_timing=False,
    dump_ir_path=None,
    opt_level=2,
    backend="bisheng",
    need_compile=True,
    kernel_name=None,
    torch_path=None,
    num_outputs=None):
    """
    Run MLIR compile for Ascend backend.

    Args:
        input_file: Input MLIR file path
        output_dir: Cache directory; compiled files are resolved by the backend.
        dyn_shape: Whether to enable dynamic shape optimization
        arch: Architecture specification (optional)
        dump_ir: Whether to dump IR after all passes
        mlir_timing: Whether to print every pass time
        dump_ir_path: Path to dump log file (optional)
        opt_level: Ascend optimization level passed to the C++ pipeline

    Returns:
        KernelLaunch with a callable and optional binary paths.

    Raises:
        RuntimeError: If compile execution fails
    """

    if backend == "pypto":
        compile_backend, load_backend = pypto_compile, _load_pypto_launch
    elif backend == "bisheng":
        compile_backend, load_backend = bisheng_compile, _load_bisheng_launch
    else:
        raise ValueError(f"Unknown Ascend backend: {backend}")
    name = kernel_name or pathlib.Path(input_file).stem
    opt_file = os.path.join(output_dir, name + "_opt.mlir")
    if need_compile:
        arch = _resolve_arch(arch)
        akg_opt(input_file, opt_file, dyn_shape=dyn_shape, enable_loop_fusion=enable_loop_fusion,
                arch=arch, dump_ir=dump_ir, mlir_timing=mlir_timing, dump_ir_path=dump_ir_path,
                opt_level=opt_level, backend=backend)
        compile_backend(opt_file, output_dir, kernel_name=name,
                        arch=arch, dump_ir=dump_ir, dyn_shape=dyn_shape,
                        enable_hfusion_compile=not enable_loop_fusion, dump_ir_path=dump_ir_path)
    return KernelLaunch(**load_backend(name, output_dir, dynamic=dyn_shape, torch_path=torch_path,
                                      num_outputs=num_outputs))


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
    Returns (processed_d, shape). May modify data[data_idx] in place.
    """
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
    return (d, shape)


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
            d, shape = _process_numpy_arg(d, data_idx, data, is_output, device_shape)
        else:
            shape = list(d.shape) if hasattr(d, 'shape') else []
        result.append((d, is_output, shape))
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
