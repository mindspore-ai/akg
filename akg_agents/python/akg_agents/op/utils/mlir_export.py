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

"""Optional torch-mlir export helpers for MathIR prompt context."""

from __future__ import annotations

import argparse
import hashlib
import multiprocessing as mp
import pathlib
import re
import shutil
import subprocess
import sys
import types
from typing import Any, List, Union

import torch


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _move_to_device(obj: Any, device: torch.device) -> Any:
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, (list, tuple)):
        return type(obj)(_move_to_device(v, device) for v in obj)
    if isinstance(obj, dict):
        return {k: _move_to_device(v, device) for k, v in obj.items()}
    return obj


def _import_module_from_context(py_context: str, *, mod_name: str, filename: str):
    module = types.ModuleType(mod_name)
    module.__file__ = filename
    exec(compile(py_context, filename, "exec"), module.__dict__)
    return module


def _lower_linalg_to_scf_text(mlir_text: str, *, mlir_opt: str) -> str:
    if mlir_opt == "torch-mlir-opt":
        cmd = [
            mlir_opt,
            "-",
            "--tm-tensor-bufferize",
            "--tm-tensor-to-loops",
            "--canonicalize",
            "--cse",
        ]
    else:
        cmd = [
            mlir_opt,
            "-",
            "--one-shot-bufferize=bufferize-function-boundaries",
            "--convert-linalg-to-loops",
            "--canonicalize",
            "--cse",
        ]
    proc = subprocess.run(
        cmd,
        input=mlir_text,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "mlir-opt lowering failed\n"
            + "cmd: "
            + " ".join(cmd)
            + "\n\nstderr:\n"
            + (proc.stderr or "")
        )
    return proc.stdout or ""


def _resolve_mlir_opt(name: str | None = None) -> str:
    candidates = [name] if name else ["torch-mlir-opt", "mlir-opt"]
    for candidate in candidates:
        if candidate and shutil.which(candidate):
            return candidate
    raise RuntimeError(
        "No mlir optimizer found on PATH. Install torch-mlir and ensure "
        "torch-mlir-opt or mlir-opt is available."
    )


def _redact_dialect_resource_values(
    mlir_text: str,
    *,
    placeholder: str = "<stripped>",
) -> str:
    def _redact_block(match: re.Match) -> str:
        block = match.group(0)
        return re.sub(
            r'(:\s*)"0x[0-9A-Fa-f]+"',
            lambda m: f'{m.group(1)}"{placeholder}"',
            block,
        )

    return re.sub(r"\{\-#.*?\#-\}", _redact_block, mlir_text, flags=re.DOTALL)


def _export_one_impl(
    py_context: str,
    output_type: str,
    device: Union[str, torch.device],
    *,
    op_name: str,
    mlir_opt: str = "torch-mlir-opt",
    redact_resource_values: bool = True,
    resource_value_placeholder: str = "<stripped>",
) -> str:
    from torch_mlir import fx

    device = torch.device(device) if isinstance(device, str) else device
    content_hash = hashlib.sha1(py_context.encode("utf-8")).hexdigest()[:8]
    module = _import_module_from_context(
        py_context,
        mod_name=f"mathir_{op_name}_{content_hash}",
        filename=f"<{op_name}>",
    )

    model_cls = getattr(module, "Model", None)
    if model_cls is None:
        raise RuntimeError(f"{op_name}: no Model class")

    init_args = _as_list(getattr(module, "get_init_inputs", lambda: [])())
    inputs = _as_list(getattr(module, "get_inputs", lambda: [])())
    if not inputs:
        raise RuntimeError(f"{op_name}: no get_inputs() inputs")

    model = model_cls(*init_args).eval().to(device)
    inputs = _move_to_device(inputs, device)

    torch_mlir_output_type = "linalg-on-tensors" if output_type == "scf" else output_type
    mlir_module = fx.export_and_import(
        model,
        *inputs,
        output_type=torch_mlir_output_type,
        func_name=op_name,
    )
    mlir_text = str(mlir_module)

    if output_type == "scf":
        mlir_text = _lower_linalg_to_scf_text(mlir_text, mlir_opt=mlir_opt)

    if redact_resource_values:
        mlir_text = _redact_dialect_resource_values(
            mlir_text,
            placeholder=resource_value_placeholder,
        )

    return mlir_text


def _export_one_worker(conn, kwargs: dict):
    try:
        conn.send(("ok", _export_one_impl(**kwargs)))
    except BaseException as exc:
        conn.send(("err", repr(exc)))
    finally:
        conn.close()


def export_one(
    py_context: str,
    output_type: str = "scf",
    device: Union[str, torch.device] = "cpu",
    *,
    op_name: str,
    mlir_opt: str = "mlir-opt",
    timeout_s: int = 120,
    isolate: bool = True,
    start_method: str = "spawn",
) -> str:
    """Export a KernelBench-style ``Model`` to MLIR text.

    The function intentionally imports ``torch_mlir`` only inside the worker path
    so MathIR can remain usable when MLIR dependencies are absent.
    """
    kwargs = {
        "py_context": py_context,
        "output_type": output_type,
        "device": device,
        "op_name": op_name,
        "mlir_opt": mlir_opt,
    }

    if not isolate:
        return _export_one_impl(**kwargs)

    ctx = mp.get_context(start_method)
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_export_one_worker, args=(child_conn, kwargs))
    proc.start()
    child_conn.close()
    proc.join(timeout_s)
    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        raise TimeoutError(f"MLIR export timed out after {timeout_s}s")

    if not parent_conn.poll():
        raise RuntimeError("MLIR export worker exited without result")

    status, payload = parent_conn.recv()
    if status == "ok":
        return payload or ""
    raise RuntimeError(payload)


def _default_kernel_path() -> pathlib.Path:
    return (
        pathlib.Path(__file__).resolve().parents[4]
        / "thirdparty/KernelBench/KernelBench/level1"
        / "50_conv_standard_2D__square_input__square_kernel.py"
    )


def _check_torch_mlir_available() -> tuple[bool, str]:
    try:
        from torch_mlir import fx  # noqa: F401

        return True, "torch_mlir import ok"
    except ImportError as exc:
        return False, f"torch_mlir import failed: {exc}"


def _run_smoke_test(
    *,
    kernel_file: pathlib.Path,
    output_type: str,
    device: str,
    mlir_opt: str,
    isolate: bool,
    preview_chars: int,
) -> int:
    ok, message = _check_torch_mlir_available()
    print(f"[check] {message}")
    if not ok:
        return 1

    try:
        resolved_mlir_opt = _resolve_mlir_opt(mlir_opt)
    except RuntimeError as exc:
        print(f"[check] {exc}")
        return 1
    print(f"[check] mlir optimizer: {resolved_mlir_opt}")

    if not kernel_file.is_file():
        print(f"[check] kernel file not found: {kernel_file}")
        return 1

    op_name = kernel_file.stem
    py_context = kernel_file.read_text(encoding="utf-8")
    print(f"[test] exporting {op_name} (output_type={output_type}, device={device})")

    mlir = export_one(
        py_context,
        output_type=output_type,
        device=device,
        op_name=op_name,
        mlir_opt=resolved_mlir_opt,
        isolate=isolate,
        timeout_s=120,
    )
    if not mlir.strip():
        print("[test] FAIL: export returned empty MLIR")
        return 1

    print(f"[test] OK: exported {len(mlir)} chars")
    if preview_chars > 0:
        print("-" * 60)
        print(mlir[:preview_chars])
        if len(mlir) > preview_chars:
            print("...")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smoke test for torch-mlir export.")
    parser.add_argument(
        "--kernel-file",
        type=pathlib.Path,
        default=_default_kernel_path(),
        help="KernelBench-style python file containing Model/get_inputs.",
    )
    parser.add_argument(
        "--output-type",
        choices=["linalg-on-tensors", "scf", "torch"],
        default="linalg-on-tensors",
        help="MLIR output dialect. Use scf to also test mlir-opt lowering.",
    )
    parser.add_argument("--device", default="cpu", help="Torch device for export.")
    parser.add_argument(
        "--mlir-opt",
        default=None,
        help="mlir optimizer binary. Default: auto-detect torch-mlir-opt, then mlir-opt.",
    )
    parser.add_argument(
        "--no-isolate",
        action="store_true",
        help="Run export in-process instead of a child process.",
    )
    parser.add_argument(
        "--preview-chars",
        type=int,
        default=2000,
        help="Print the first N chars of exported MLIR. Set 0 to skip preview.",
    )
    args = parser.parse_args()
    sys.exit(
        _run_smoke_test(
            kernel_file=args.kernel_file,
            output_type=args.output_type,
            device=args.device,
            mlir_opt=args.mlir_opt,
            isolate=not args.no_isolate,
            preview_chars=args.preview_chars,
        )
    )
