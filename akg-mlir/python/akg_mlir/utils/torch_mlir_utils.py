# Copyright 2026 Huawei Technologies Co., Ltd
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

"""torch op acc check pipline utils"""
import os
import re
import math
import pathlib
import subprocess
import logging
from typing import Optional


def _import_torch_mlir():
    """Import torch-mlir lazily.

    torch-mlir is only required by torch-mlir pipeline functions. Keep this
    import lazy so that users of pure Python helpers in this module do not need
    torch-mlir installed.
    """
    # pylint: disable=import-outside-toplevel
    try:
        from torch_mlir import ir as torch_mlir_ir
        from torch_mlir.passmanager import PassManager as TorchMlirPassManager
        from torch_mlir.dialects import torch as torch_dialect
    except ImportError as exc:
        raise RuntimeError(
            "torch_mlir is required for torch-mlir pipeline functions, "
            "but it is not installed or cannot be imported. Please install "
            "torch-mlir in the current Python environment, or avoid calling "
            "run_torch_mlir_to_json() and "
            "run_torch_mlir_to_linalg_on_tensors()."
        ) from exc

    return torch_mlir_ir, TorchMlirPassManager, torch_dialect


def find_first_func_name(input_file: str) -> Optional[str]:
    """Get kernel name."""
    input_file = pathlib.Path(input_file)
    mlir_text = input_file.read_text(encoding='utf-8')
    pat = re.compile(
        r"""\bfunc\.func\b(?:\s+\w+)*\s+@([^\s(]+)\s*\(""",
        re.MULTILINE,
    )
    m = pat.search(mlir_text)
    return m.group(1) if m else None


def torch_normalize_dtype(dtype):
    """Normalize a torch dtype string to a standard form."""
    if dtype in ("int1", "i1"):
        return "bool"
    return dtype


def get_named_op_str(
    input_file_path: str,
    output_file_path: str,
    kernel_name: str,
    dynamic: bool = False,
    output_dir: Optional[str] = None,
) -> str:
    """Run complete MLIR pipeline for Ascend: bishengir-opt."""
    if output_dir is None:
        output_dir_obj = tempfile.TemporaryDirectory()
        output_dir = output_dir_obj.name
    else:
        os.makedirs(output_dir, exist_ok=True)
        output_dir_obj = None

    cmd = [
        "bishengir-opt",
        "--torch-backend-to-named-op-backend-pipeline=ensure-no-implicit-broadcast=true",
        input_file_path,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
        logging.debug("bishengir-opt exec %s.mlir", kernel_name)

        if result.returncode != 0:
            logging.error("bishengir-opt failed, message: %s", result.stderr)
            raise RuntimeError(f"MLIR fail: {result.stderr[:500]}")

        processed_lines = []
        for line in result.stdout.splitlines():
            if "ml_program.global" not in line:
                processed_lines.append(line)

        func_attr = ("attributes {hacc.entry, "
                    "hacc.function_kind = #hacc.function_kind<HOST>}"
                    if dynamic else
                    "attributes {hacc.entry, "
                    "hacc.function_kind = #hacc.function_kind<DEVICE>}")

        processed_mlir = "\n".join(processed_lines)

        def _inject_func_attrs(mlir_text: str, func_attr: str) -> str:

            func_line_re = re.compile(
                r'^(\s*func\.func\b.*?)(\s*\{\s*)$',
                re.MULTILINE,
            )

            def repl(m):
                line_before_brace = m.group(1)
                brace = m.group(2)
                if (
                    " attributes " in line_before_brace
                    or line_before_brace.rstrip().endswith("attributes")
                ):
                    return m.group(0)
                return f"{line_before_brace} {func_attr}{brace}"

            new_text, n = func_line_re.subn(repl, mlir_text, count=1)
            if n == 0:
                raise ValueError("not find `func.func ... {`")
            return new_text

        processed_mlir = _inject_func_attrs(processed_mlir, func_attr)

        with open(output_file_path, "w", encoding="utf=8") as f:
            f.write(processed_mlir)

        logging.debug("wrote named-op mlir to %s", output_file_path)
        return output_file_path

    except Exception as e:
        logging.error("bishengir-opt failed!")
        raise RuntimeError("bishengir-opt failed!") from e


def run_torch_mlir_to_json(file_path, output_file):
    """
    Run torch-to-json pass by using torch-mlir Python wheel package.

    Args:
        file_path: input .mlir path.
        output_file: output info/json file path, for example
                     "/tmp/kernel_meta/model.info" or "model.info".

    Returns:
        Path to the expected output file.
    """
    torch_mlir_ir, torch_mlir_pass_manager, torch_dialect = _import_torch_mlir()
    file_path = pathlib.Path(file_path)
    output_file = pathlib.Path(output_file)
    if not str(output_file):
        raise ValueError("output_file cannot be empty")
    output_parent = output_file.parent
    if str(output_parent) and str(output_parent) != ".":
        output_parent.mkdir(parents=True, exist_ok=True)
    output_file_str = str(output_file)
    if any(ch.isspace() for ch in output_file_str):
        raise ValueError(
            f"output_file cannot contain whitespace for MLIR pass pipeline: "
            f"{output_file_str}"
        )
    pass_options = [f"output-name={output_file_str}"]
    mlir_text = file_path.read_text(encoding="utf-8")
    try:
        with torch_mlir_ir.Context() as ctx:
            torch_dialect.register_dialect(ctx)
            module = torch_mlir_ir.Module.parse(mlir_text)
            pipeline = (
                "builtin.module("
                "func.func("
                f"torch-to-json{{{' '.join(pass_options)}}}"
                ")"
                ")"
            )
            logging.debug("torch-to-json pipeline: %s", pipeline)
            pm = torch_mlir_pass_manager.parse(pipeline)
            pm.run(module.operation)
        logging.debug("torch-mlir to Json success")
    except Exception as exc:
        logging.error("torch-to-json failed!")
        raise RuntimeError("torch-to-json failed!") from exc
    return output_file


def run_torch_mlir_to_linalg_on_tensors(file_path, output_path):
    """
    Run torch backend to linalg-on-tensors backend pipeline by using
    torch-mlir Python wheel package.

    Equivalent to:
      torch-mlir-opt <file_path> \
        --torch-backend-to-linalg-on-tensors-backend-pipeline \
        -o <output_path>

    Args:
        file_path: input torch mlir file
        output_path: output mlir path, e.g. dump_dir / "{kernel_name}_linalg.mlir"

    Returns:
        Path: output_path as Path

    Raises:
        RuntimeError: if the pipeline fails
    """
    torch_mlir_ir, torch_mlir_pass_manager, torch_dialect = _import_torch_mlir()
    file_path = pathlib.Path(file_path)
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mlir_text = file_path.read_text(encoding="utf-8")
    try:
        with torch_mlir_ir.Context() as ctx:
            torch_dialect.register_dialect(ctx)
            module = torch_mlir_ir.Module.parse(mlir_text)
            pipeline = (
                "builtin.module("
                "torch-backend-to-linalg-on-tensors-backend-pipeline"
                ")"
            )
            logging.debug("torch-mlir to linalg-on-tensors pipeline: %s", pipeline)
            pm = torch_mlir_pass_manager.parse(pipeline)
            pm.run(module.operation)
            output_path.write_text(str(module), encoding="utf-8")

        logging.debug("torch-mlir to linalg-on-tensors success: %s", output_path)
    except Exception as exc:
        logging.error("torch-mlir to linalg-on-tensors failed!")
        raise RuntimeError("torch-mlir to linalg-on-tensors failed!") from exc
    return output_path


_VAR_REF_RE = re.compile(r"^(output|input)_\d+$")


def format_py_value(v):
    """Format a Python value as source code for generated NumPy reference code."""
    if isinstance(v, str):
        lv = v.strip().lower()
        if lv in ("inf", "+inf"):
            return 'float("inf")'
        if lv in ("-inf",):
            return 'float("-inf")'
        if lv == "nan":
            return 'float("nan")'
        if _VAR_REF_RE.match(v.strip()):
            return v.strip()
        return repr(v)

    if isinstance(v, float):
        if math.isnan(v):
            return 'float("nan")'
        if math.isinf(v):
            return 'float("inf")' if v > 0 else 'float("-inf")'
        return repr(v)

    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(format_py_value(x) for x in v) + "]"

    return repr(v)


def gen_slice_tensor(dst_name, src, dim, start, end, step):
    """Generate NumPy code that approximates torch.aten.slice.Tensor semantics.

    The generated code:
    - builds a slice on dimension `dim` with (`start`, `end`, `step`),
    - normalizes the large int64 end sentinel to the dimension size,
    - applies the slice to `src`,
    - stores the result in `dst_name`.

    Notes:
    - This is intended for generated NumPy reference code in the bisheng
      pipeline.
    - Using Python/NumPy slice syntax preserves negative indices and other
      standard slicing behavior better than np.take(range(...)).
    """
    idx_name = f"_{dst_name}_slices"
    dim_name = f"_{dst_name}_dim"
    start_name = f"_{dst_name}_start"
    end_name = f"_{dst_name}_end"
    step_name = f"_{dst_name}_step"
    dim_size_name = f"_{dst_name}_dim_size"

    start_expr = "None" if start is None else f"int({start})"
    end_expr = "None" if end is None else f"int({end})"

    return "\n".join([
        f"{dim_name} = int({dim})",
        f"{start_name} = {start_expr}",
        f"{end_name} = {end_expr}",
        f"{step_name} = int({step})",
        f"{dim_size_name} = {src}.shape[{dim_name}]",
        f"if {start_name} is None:",
        f"    {start_name} = 0 if {step_name} > 0 else {dim_size_name} - 1",
        f"if {end_name} is None or {end_name} >= 2**63 - 1:",
        f"    {end_name} = {dim_size_name}",
        f"{idx_name} = [slice(None)] * {src}.ndim",
        f"{idx_name}[{dim_name}] = slice({start_name}, {end_name}, {step_name})",
        f"{dst_name} = {src}[tuple({idx_name})]",
    ])


def gen_slice_scatter(dst_name, base, src, dim, start, end, step):
    """Generate NumPy code that approximates torch.aten.slice_scatter semantics.

    The generated code:
    - copies `base` into `dst_name`,
    - builds a slice on dimension `dim` with (`start`, `end`, `step`),
    - checks that the target slice shape matches `src.shape`,
    - assigns `src` into that slice.

    Notes:
    - This is intended for generated NumPy reference code in the bisheng
      pipeline.
    - It matches the common lowering pattern used in our tests, but is not a
      full reimplementation of every PyTorch edge case.
    """
    idx_name = f"_{dst_name}_slices"
    start_name = f"_{dst_name}_start"
    end_name = f"_{dst_name}_end"
    step_name = f"_{dst_name}_step"
    dim_size_name = f"_{dst_name}_dim_size"
    target_view_name = f"_{dst_name}_target_view"

    start_expr = "None" if start is None else f"int({start})"
    end_expr = "None" if end is None else f"int({end})"

    return "\n".join([
        f"{dst_name} = np.array({base}, copy=True)",
        f"_{dst_name}_dim = int({dim})",
        f"{start_name} = {start_expr}",
        f"{end_name} = {end_expr}",
        f"{step_name} = int({step})",
        f"{dim_size_name} = {dst_name}.shape[_{dst_name}_dim]",
        f"if {start_name} is None:",
        f"    {start_name} = 0 if {step_name} > 0 else {dim_size_name} - 1",
        f"if {end_name} is None or {end_name} >= 2**63 - 1:",
        f"    {end_name} = {dim_size_name}",
        f"{idx_name} = [slice(None)] * {dst_name}.ndim",
        f"{idx_name}[_{dst_name}_dim] = slice({start_name}, {end_name}, {step_name})",
        f"{target_view_name} = {dst_name}[tuple({idx_name})]",
        f"if {target_view_name}.shape != {src}.shape:",
        f"    raise ValueError('slice_scatter shape mismatch: %s vs %s' % ({target_view_name}.shape, {src}.shape))",
        f"{dst_name}[tuple({idx_name})] = {src}",
    ])


def gen_constant_pad_nd(dst_name, x, pad, value):
    """Generate reference code for torch.aten.constant_pad_nd.

    Semantics:
    - `pad` is interpreted from the last dimension outward, pairwise:
      [left_last, right_last, left_second_last, right_second_last, ...]
    - positive pad means constant padding with `value`
    - negative pad means cropping on that side

    Notes:
    - This generates NumPy-based fallback code in the same style as
      `gen_slice_scatter`.
    - It handles the common cases used by our Torch-IR fallback path.
    """
    pad_name = f"_{dst_name}_pad"
    ndim_name = f"_{dst_name}_ndim"
    num_pad_dims_name = f"_{dst_name}_num_pad_dims"
    pad_width_name = f"_{dst_name}_pad_width"
    i_name = f"_{dst_name}_i"
    left_name = f"_{dst_name}_left"
    right_name = f"_{dst_name}_right"
    start_name = f"_{dst_name}_start"
    end_name = f"_{dst_name}_end"
    cropped_name = f"_{dst_name}_cropped"

    return "\n".join([
        f"{pad_name} = list({pad})",
        f"{ndim_name} = {x}.ndim",
        f"{num_pad_dims_name} = len({pad_name}) // 2",
        f"_{dst_name}_slices = [slice(None)] * {ndim_name}",
        f"{pad_width_name} = [(0, 0)] * {ndim_name}",
        f"for {i_name} in range({num_pad_dims_name}):",
        f"    _{dst_name}_dim = {ndim_name} - 1 - {i_name}",
        f"    {left_name} = int({pad_name}[2 * {i_name}])",
        f"    {right_name} = int({pad_name}[2 * {i_name} + 1])",
        f"    {start_name} = max(-{left_name}, 0)",
        f"    {end_name} = {x}.shape[_{dst_name}_dim] - max(-{right_name}, 0)",
        f"    _{dst_name}_slices[_{dst_name}_dim] = slice({start_name}, {end_name})",
        f"    {pad_width_name}[_{dst_name}_dim] = (max({left_name}, 0), max({right_name}, 0))",
        f"{cropped_name} = {x}[tuple(_{dst_name}_slices)]",
        f"{dst_name} = np.pad({cropped_name}, {pad_width_name}, mode='constant', constant_values={value})",
    ])


def gen_broadcast_to(dst_name, x, shape):
    """Generate reference code for torch.aten.broadcast_to.

    Semantics:
    - `shape` is the target shape template.
    - A value of `-1` means keeping the corresponding input dimension size.
    - Dimension mapping follows broadcasting right-alignment semantics.

    Notes:
    - NumPy's np.broadcast_to does not accept `-1` in the shape, so we
      resolve it first.
    - `-1` is resolved against the input shape using right alignment.
    - This generates NumPy-based fallback code.
    """
    shape_name = f"_{dst_name}_shape"
    input_shape_name = f"_{dst_name}_input_shape"
    input_rank_name = f"_{dst_name}_input_rank"
    target_rank_name = f"_{dst_name}_target_rank"
    resolved_shape_name = f"_{dst_name}_resolved_shape"
    i_name = f"_{dst_name}_i"
    dim_name = f"_{dst_name}_dim"
    input_idx_name = f"_{dst_name}_input_idx"

    return "\n".join([
        f"{shape_name} = list({shape})",
        f"{input_shape_name} = list({x}.shape)",
        f"{input_rank_name} = len({input_shape_name})",
        f"{target_rank_name} = len({shape_name})",
        f"{resolved_shape_name} = []",
        f"for {i_name}, {dim_name} in enumerate({shape_name}):",
        f"    if {dim_name} == -1:",
        f"        {input_idx_name} = {i_name} - ({target_rank_name} - {input_rank_name})",
        f"        if {input_idx_name} < 0 or {input_idx_name} >= {input_rank_name}:",
        f"            raise ValueError("
        f"f'Cannot resolve -1 in broadcast shape {{{shape_name}}} from input shape {{{input_shape_name}}}'"
        f")",
        f"        {resolved_shape_name}.append({input_shape_name}[{input_idx_name}])",
        "    else:",
        f"        {resolved_shape_name}.append(int({dim_name}))",
        f"{dst_name} = np.broadcast_to({x}, {resolved_shape_name})",
    ])


TORCH_DTYPE_TO_NUMPY = {
    # unsigned
    0:  "np.uint8",
    27: "np.uint16",
    28: "np.uint32",
    29: "np.uint64",

    # signed
    1: "np.int8",
    2: "np.int16",
    3: "np.int32",
    4: "np.int64",

    # float
    5: "np.float16",
    6: "np.float32",
    7: "np.float64",

    # complex
    9:  "np.complex64",
    10: "np.complex128",

    # bool
    11: "bool",

    # bfloat16 fallback
    15: "np.float32",

    # float8 fallback
    23: "np.float32",
    24: "np.float32",
    25: "np.float32",
    26: "np.float32",
    44: "np.float32",
    45: "np.float32",
}
