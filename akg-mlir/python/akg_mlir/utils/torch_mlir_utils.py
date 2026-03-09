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
import subprocess
import logging
import re
from typing import Optional
from pathlib import Path
from akg.backends.ascend import run_akg_opt, dump_ascend_meta_data


def find_first_func_name(mlir_text: str) -> Optional[str]:
    pat = re.compile(
        r"""\bfunc\.func\b(?:\s+\w+)*\s+@([^\s(]+)\s*\(""",
        re.MULTILINE,
    )
    m = pat.search(mlir_text)
    return m.group(1) if m else None

def torch_normalize_dtype(dtype):
    if dtype in ("int1", "i1"):
        return "bool"
    return dtype

def get_named_op_str(
    input_file_path: str,
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

    cmd = (
        f"bishengir-opt "
        "--torch-backend-to-named-op-backend-pipeline="
        "\"ensure-no-implicit-broadcast=true\" "
        f"{input_file_path}"
    )

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            check=False,
        )
        print(f"[INFO] bishengir-opt exec {kernel_name}.mlir")

        if result.returncode != 0:
            print("[ERROR] bishengir-opt failed")
            raise RuntimeError(f"MLIR fail: {result.stderr[:500]}")

        processed_lines = []
        for line in result.stdout.splitlines():
            if "ml_program.global" not in line:
                processed_lines.append(line)

        func_attr = (
            "attributes {hacc.entry, "
            "hacc.function_kind = #hacc.function_kind<HOST>}"
        )

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

        output_file_path = os.path.join(
            output_dir, f"{kernel_name}_named_op.mlir"
        )
        with open(output_file_path, "w", encoding="utf=8") as f:
            f.write(processed_mlir)

        print(f"[INFO] wrote named-op mlir to {output_file_path}")
        return output_file_path

    except Exception as e:
        print(f"[ERROR] exception: {e}")
        raise


def ascend_compile_with_hfusion(input_file, output_so_path):
    "bisheng-compile for bisheng pipeline."
    compile_cmd = [
        "bishengir-compile",
        input_file,
        "-enable-hfusion-compile=true",
        "-enable-hivm-compile=true",
        "-enable-bin-relocation=false",
        "-block-dim=40",
        "-enable-auto-multi-buffer=true",
        "-o",
        output_so_path,
    ]
    try:
        subprocess.run(
            compile_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        print(f"[INFO] bishengir-compile {input_file} success")
    except subprocess.CalledProcessError as e:
        print("compile_error!!!")
        logging.error(
            "run bishengir-compile failed! cmd:\n %s \nerror message:\n %s",
            e.cmd,
            e.stderr,
        )
        raise RuntimeError(
            "bishengir-compile failed in case: "
            + os.path.basename(input_file)
            + "!\n"
        ) from e
    output_dir = os.path.dirname(output_so_path)
    kernel_name = os.path.splitext(os.path.basename(output_so_path))[0]
    if kernel_name.startswith("lib"):
        kernel_name = kernel_name[3:]
    dump_ascend_meta_data(output_dir, kernel_name, block_dim=40)

def run_mlir_ascend_pipeline(
    input_file,
    output_file,
    akg_tools_dir=None,
    dyn_shape=False,
    enable_loop_fusion=True,
    arch=None,
    dump_ir=False,
    mlir_timing=False,
    dump_log_path=None,
):
    """
    Run complete MLIR pipeline for Ascend: akg-opt.

    Args:
        input_file: Input MLIR file path
        output_file: Final output MLIR file path
        akg_tools_dir: Directory containing akg tools (default: auto-detect)
        dyn_shape: Whether to enable dynamic shape optimization
        enable_loop_fusion: Whether to enable loop fusion
        arch: Architecture specification (optional)
        dump_ir: Whether to dump IR after all passes
        mlir_timing: Whether to print every pass time
        dump_log_path: Path to dump log file (optional)
    Returns:
        Path to final output file
    """
    run_akg_opt(
        input_file=input_file,
        output_file=output_file,
        akg_tools_dir=akg_tools_dir,
        dyn_shape=dyn_shape,
        enable_loop_fusion=enable_loop_fusion,
        arch=arch,
        dump_ir=dump_ir,
        mlir_timing=mlir_timing,
        dump_log_path=dump_log_path
    )
    return output_file

def run_torch_mlir_to_json(torch_mlir_opt: str, file_path: str | Path) -> None:
    """
    Run: torch-mlir-opt <file_path> --torch-to-json

    Args:
        torch_mlir_opt: path to torch-mlir-opt executable
        file_path: input mlir file

    Raises:
        RuntimeError: if the command fails
    """
    file_path = Path(file_path)
    try:
        subprocess.run(
            [torch_mlir_opt, str(file_path), "--torch-to-json"],
            check=True,
        )
        print("[INFO] torch-mlir to Json success")
    except subprocess.CalledProcessError as e:
        print("[ERROR] torch-to-json failed")
        raise RuntimeError("torch-to-json failed") from e

def run_torch_mlir_to_linalg_on_tensors(
    torch_mlir_opt: str,
    file_path: str | Path,
    output_path: str | Path,
) -> Path:
    """
    Run:
      torch-mlir-opt <file_path> --torch-backend-to-linalg-on-tensors-backend-pipeline -o <output_path>

    Args:
        torch_mlir_opt: path to torch-mlir-opt executable
        file_path: input torch mlir file
        output_path: output mlir path (e.g. dump_dir/'out_linalg.mlir')

    Returns:
        Path: output_path as Path

    Raises:
        RuntimeError: if the command fails
    """
    file_path = Path(file_path)
    output_path = Path(output_path)

    try:
        subprocess.run(
            [
                torch_mlir_opt,
                str(file_path),
                "--torch-backend-to-linalg-on-tensors-backend-pipeline",
                "-o",
                str(output_path),
            ],
            check=True,
        )
        print("[INFO] torch-mlir to linalg-on-tensors success")
    except subprocess.CalledProcessError as e:
        print("[ERROR] torch-mlir to linalg-on-tensors failed")
        raise RuntimeError("torch-mlir to linalg-on-tensors failed") from e

    return output_path

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
