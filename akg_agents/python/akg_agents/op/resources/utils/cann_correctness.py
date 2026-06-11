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

"""
CANN-Bench tensor accuracy comparison engine.

Transplanted from CANN-Bench native kernel_eval/utils/compare.py and
kernel_eval/utils/thresholds.py. Only keeps the comparison logic needed
for pass/fail + error detail; scoring/report fields are omitted.
"""

import math
import traceback
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

# ---------------------------------------------------------------------------
# Threshold tables (from kernel_eval/utils/thresholds.py)
# ---------------------------------------------------------------------------

PRECISION_THRESHOLDS: Dict[str, float] = {
    "float16": 2**-10,
    "bfloat16": 2**-7,
    "float32": 2**-13,
    "float64": 2**-13,
    "hifloat32": 2**-11,
    "float8_e4m3fn": 2**-3,
    "float8_e5m2": 2**-2,
    "int8": 0, "int16": 0, "int32": 0, "int64": 0,
    "uint8": 0, "uint16": 0, "uint32": 0, "uint64": 0,
}

SMALL_VALUE_THRESHOLDS: Dict[str, float] = {
    "float16": 2**-11,
    "bfloat16": 2**-8,
    "float32": 2**-14,
    "float64": 2**-14,
    "hifloat32": 2**-12,
    "float8_e4m3fn": 2**-4,
    "float8_e5m2": 2**-3,
}

SMALL_VALUE_ERROR_THRESHOLDS: Dict[str, float] = {
    "float16": 2**-16,
    "bfloat16": 2**-16,
    "float32": 2**-30,
    "float64": 2**-30,
    "hifloat32": 2**-28,
    "float8_e4m3fn": 2**-6,
    "float8_e5m2": 2**-5,
}

CANCEL_BOUNDARY_THRESHOLDS: Dict[str, float] = {
    "float32": 2**-8,
    "float64": 2**-8,
    "float16": 2**-5,
    "bfloat16": 2**-3,
    "hifloat32": 2**-8,
    "float8_e4m3fn": 2**-1,
    "float8_e5m2": 2**-0,
}

CANCEL_ZERO_THRESHOLDS: Dict[str, float] = {
    "float32": 2**-8,
    "float64": 2**-8,
    "float16": 2**-5,
    "bfloat16": 2**-3,
    "hifloat32": 2**-8,
    "float8_e4m3fn": 2**-1,
    "float8_e5m2": 2**-0,
}

_DEFAULT_DTYPE = "float32"


def _get_threshold(dtype_str: str) -> float:
    return PRECISION_THRESHOLDS.get(dtype_str.lower(), PRECISION_THRESHOLDS[_DEFAULT_DTYPE])

def _get_small_value_threshold(dtype_str: str) -> float:
    return SMALL_VALUE_THRESHOLDS.get(dtype_str.lower(), SMALL_VALUE_THRESHOLDS[_DEFAULT_DTYPE])

def _get_small_value_error(dtype_str: str) -> float:
    return SMALL_VALUE_ERROR_THRESHOLDS.get(dtype_str.lower(), SMALL_VALUE_ERROR_THRESHOLDS[_DEFAULT_DTYPE])

def _get_cancel_boundary(dtype_str: str) -> float:
    return CANCEL_BOUNDARY_THRESHOLDS.get(dtype_str.lower(), CANCEL_BOUNDARY_THRESHOLDS[_DEFAULT_DTYPE])

def _get_cancel_zero_threshold(dtype_str: str) -> float:
    return CANCEL_ZERO_THRESHOLDS.get(dtype_str.lower(), CANCEL_ZERO_THRESHOLDS[_DEFAULT_DTYPE])


# ---------------------------------------------------------------------------
# Integer dtype set
# ---------------------------------------------------------------------------

_INTEGER_DTYPES = tuple(
    dt for dt in (
        torch.int8, torch.int16, torch.int32, torch.int64,
        torch.uint8,
        getattr(torch, "uint16", None),
        getattr(torch, "uint32", None),
        getattr(torch, "uint64", None),
    )
    if dt is not None
)


# ---------------------------------------------------------------------------
# Core comparison logic
# ---------------------------------------------------------------------------


def _normalize_outputs(output: Any) -> List[Any]:
    """Normalize output to a list of tensors (with None placeholders)."""
    if isinstance(output, torch.Tensor):
        return [output]
    elif isinstance(output, (tuple, list)):
        result: List[Any] = []
        for item in output:
            if isinstance(item, torch.Tensor):
                result.append(item)
            elif isinstance(item, (tuple, list)):
                for sub_item in item:
                    if isinstance(sub_item, torch.Tensor):
                        result.append(sub_item)
                    else:
                        result.append(None)
            else:
                result.append(None)
        return result
    else:
        return []


def _compare_single_tensor(
    output: torch.Tensor,
    golden: torch.Tensor,
    threshold: float,
    dtype: str,
    native_output: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """Compare a single tensor pair (MERE/MARE + small-value + cancellation).

    Returns: {"passed", "mere", "mare", "threshold", "error_msg"}
    """
    # Move to CPU for comparison
    if output.is_cuda or output.device.type == "npu":
        output = output.cpu()
    if golden.is_cuda or golden.device.type == "npu":
        golden = golden.cpu()

    # Shape mismatch
    if output.shape != golden.shape:
        return {"passed": False, "mere": 0.0, "mare": 0.0, "threshold": threshold,
                "error_msg": f"Shape mismatch: output={output.shape}, golden={golden.shape}"}

    # --- Bit-exact path for threshold=0 floating-point ---
    if threshold == 0 and output.is_floating_point():
        _BIT_VIEW = {
            torch.float16: torch.int16,
            torch.bfloat16: torch.int16,
            torch.float32: torch.int32,
            torch.float64: torch.int64,
        }
        int_dtype = _BIT_VIEW.get(output.dtype)
        if int_dtype is not None:
            golden_cast = golden.to(output.dtype).contiguous()
            output_c = output.contiguous()
            out_bits = output_c.view(int_dtype)
            gold_bits = golden_cast.view(int_dtype)
            if torch.equal(out_bits, gold_bits):
                return {"passed": True, "mere": 0.0, "mare": 0.0, "threshold": threshold, "error_msg": None}
            mismatch_count = int((out_bits != gold_bits).sum())
            return {"passed": False, "mere": 0.0, "mare": 0.0, "threshold": threshold,
                    "error_msg": f"bit-exact: {mismatch_count}/{output.numel()} elements differ"}

    # --- Integer path ---
    if output.dtype in _INTEGER_DTYPES:
        diff = torch.abs(output.long() - golden.long())
        mismatch_mask = diff > max(threshold, 0)
        mismatch_count = int(mismatch_mask.sum())
        if mismatch_count == 0:
            return {"passed": True, "mere": 0.0, "mare": 0.0, "threshold": threshold, "error_msg": None}

        return {"passed": False, "mere": 0.0, "mare": 0.0, "threshold": threshold,
                "error_msg": f"integer mismatch: {mismatch_count}/{output.numel()} elements exceed tolerance {threshold}"}

    # --- Floating-point MERE/MARE path ---
    target_dtype = output.dtype
    golden_truncated = golden.to(target_dtype).double()
    output_fp64 = output.double()

    # NaN handling
    if torch.any(torch.isnan(output_fp64)) or torch.any(torch.isnan(golden_truncated)):
        nan_out = torch.isnan(output_fp64)
        nan_gold = torch.isnan(golden_truncated)
        if not torch.all(nan_out == nan_gold):
            return {"passed": False, "mere": 0.0, "mare": 0.0, "threshold": threshold,
                    "error_msg": "NaN position mismatch"}

    # Inf handling (saturation)
    inf_match_mask = torch.zeros_like(output_fp64, dtype=torch.bool)
    if torch.any(torch.isinf(output_fp64)) or torch.any(torch.isinf(golden_truncated)):
        inf_out = torch.isinf(output_fp64)
        inf_gold = torch.isinf(golden_truncated)
        inf_mismatch = inf_out != inf_gold

        if torch.any(inf_mismatch):
            max_finite = float(torch.finfo(target_dtype).max)
            if torch.any(inf_out & ~inf_gold):
                output_fp64[inf_out & ~inf_gold] = torch.sign(output_fp64[inf_out & ~inf_gold]) * max_finite
            if torch.any(inf_gold & ~inf_out):
                golden_truncated[inf_gold & ~inf_out] = torch.sign(golden_truncated[inf_gold & ~inf_out]) * max_finite

        both_inf = inf_out & inf_gold
        if torch.any(both_inf):
            if not torch.all(torch.sign(output_fp64[both_inf]) == torch.sign(golden_truncated[both_inf])):
                return {"passed": False, "mere": 0.0, "mare": 0.0, "threshold": threshold,
                        "error_msg": "Inf sign mismatch"}
            inf_match_mask[both_inf] = True

    # Relative error
    diff = torch.abs(output_fp64 - golden_truncated)
    golden_abs = torch.abs(golden_truncated)
    denominator = golden_abs + 1e-7
    relative_error = diff / denominator

    valid_mask = ~(torch.isnan(relative_error) | torch.isinf(relative_error) | inf_match_mask)
    valid_relative_error = relative_error[valid_mask]

    if len(valid_relative_error) == 0:
        return {"passed": True, "mere": 0.0, "mare": 0.0, "threshold": threshold, "error_msg": None}

    mere = float(valid_relative_error.mean())
    mare = float(valid_relative_error.max())

    # Phase 1: overall check
    mare_threshold = 10 * threshold
    if mere < threshold and mare < mare_threshold:
        return {"passed": True, "mere": mere, "mare": mare, "threshold": threshold, "error_msg": None}

    # Phase 2: analyze failure domains
    mismatch_mask = relative_error > mare_threshold
    mismatch_mask[~valid_mask] = False
    mismatch_count = int(mismatch_mask.sum())

    # --- Small-value domain ---
    small_value_threshold = _get_small_value_threshold(dtype)
    small_value_error = _get_small_value_error(dtype)

    small_value_mask = golden_abs < small_value_threshold
    small_value_mask[~valid_mask] = False
    small_value_total_count = int(small_value_mask.sum())
    small_value_error_count = int((small_value_mask & (diff > small_value_error)).sum())

    # CPU reference for small-value comparison
    if native_output is not None:
        if native_output.device.type != "cpu":
            native_output = native_output.cpu()
        cpu_diff = torch.abs(native_output.double() - golden_truncated)
    else:
        cpu_diff = torch.abs(golden.to(target_dtype).double() - golden_truncated)

    small_value_cpu_error_count = int((small_value_mask & (cpu_diff > small_value_error)).sum())
    small_value_passed = (small_value_error_count / max(small_value_cpu_error_count, 1)) <= 2 if small_value_total_count > 0 else True

    # --- Cancellation domain ---
    cancel_boundary = _get_cancel_boundary(dtype)
    cancel_zero_threshold = _get_cancel_zero_threshold(dtype)

    output_abs = torch.abs(output_fp64)
    output_near_zero = output_abs < cancel_zero_threshold
    golden_in_cancel_range = (golden_abs < cancel_boundary) & (golden_abs >= small_value_threshold)
    cancel_mask = output_near_zero & golden_in_cancel_range & valid_mask
    cancel_total_count = int(cancel_mask.sum())

    cancel_error_count = int((cancel_mask & (relative_error > mare_threshold)).sum())
    cpu_relative_error = cpu_diff / (golden_abs + 1e-7)
    cancel_cpu_error_count = int((cancel_mask & (cpu_relative_error > mare_threshold)).sum())
    cancel_passed = (cancel_error_count / max(cancel_cpu_error_count, 1)) <= 2 if cancel_total_count > 0 else True

    # Final judgment
    normal_mismatch_count = int((mismatch_mask & ~small_value_mask & ~cancel_mask).sum())

    if normal_mismatch_count > 0:
        passed = False
        normal_mask = ~small_value_mask & ~cancel_mask & valid_mask
        normal_re = relative_error[normal_mask]
        mere = float(normal_re.mean()) if len(normal_re) > 0 else 0.0
        mare = float(normal_re.max()) if len(normal_re) > 0 else 0.0
    else:
        passed = small_value_passed and cancel_passed

    return {"passed": passed, "mere": mere, "mare": mare, "threshold": threshold, "error_msg": None}


def compare_tensors(
    output: Union[torch.Tensor, Tuple, List],
    golden: Union[torch.Tensor, Tuple, List],
    dtype: str = "float32",
    threshold: Optional[float] = None,
    native_output: Optional[Union[torch.Tensor, Tuple, List]] = None,
    ignore_output_indices: Optional[List[int]] = None,
    custom_thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Compare output tensors against golden reference (MERE/MARE standard).

    Returns: {"passed", "mere", "mare", "threshold", "error_msg", "output_results"}
    """
    if custom_thresholds is None:
        custom_thresholds = {}

    def _get_output_threshold(dtype_str: str) -> float:
        dtype_lower = dtype_str.lower()
        if dtype_lower in custom_thresholds:
            return custom_thresholds[dtype_lower]
        return _get_threshold(dtype_str)

    if threshold is None:
        threshold = _get_output_threshold(dtype)

    try:
        outputs = _normalize_outputs(output)
        goldens = _normalize_outputs(golden)
        native_outputs = _normalize_outputs(native_output) if native_output is not None else None

        if len(outputs) != len(goldens):
            return {"passed": False, "mere": 0.0, "mare": 0.0, "threshold": threshold,
                    "error_msg": f"Output count mismatch: output={len(outputs)}, golden={len(goldens)}",
                    "output_results": []}

        all_passed = True
        mere_sum = 0.0
        mare_max = 0.0
        total_count = 0
        output_results: List[Dict[str, Any]] = []

        for i, (out_tensor, gold_tensor) in enumerate(zip(outputs, goldens)):
            if ignore_output_indices and i in ignore_output_indices:
                output_results.append({"index": i, "passed": True, "mere": 0.0, "mare": 0.0,
                                       "threshold": threshold, "error_msg": "(skipped)"})
                continue

            if out_tensor is None or gold_tensor is None:
                output_results.append({"index": i, "passed": True, "mere": 0.0, "mare": 0.0,
                                       "threshold": threshold, "error_msg": "(None placeholder)"})
                continue

            out_dtype_str = str(out_tensor.dtype).replace("torch.", "")
            out_threshold = _get_output_threshold(out_dtype_str)
            native_tensor = native_outputs[i] if native_outputs is not None else None

            result = _compare_single_tensor(out_tensor, gold_tensor, out_threshold, out_dtype_str, native_tensor)

            sr = {"index": i, "dtype": out_dtype_str, "passed": result["passed"],
                  "mere": result["mere"], "mare": result["mare"],
                  "threshold": out_threshold, "error_msg": result.get("error_msg", "")}
            output_results.append(sr)

            all_passed = all_passed and result["passed"]
            mere_sum += result["mere"] * out_tensor.numel()
            mare_max = max(mare_max, result["mare"])
            total_count += out_tensor.numel()

        mere = mere_sum / total_count if total_count > 0 else 0.0

        # Find first failure for error_msg
        error_msg = None
        for sr in output_results:
            if not sr["passed"] and not (sr.get("error_msg") or "").startswith("(skip"):
                error_msg = sr["error_msg"]
                break

        return {"passed": all_passed, "mere": mere, "mare": mare_max,
                "threshold": threshold, "error_msg": error_msg,
                "output_results": output_results}

    except Exception as e:
        return {"passed": False, "mere": 0.0, "mare": 0.0, "threshold": threshold,
                "error_msg": f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
                "output_results": []}


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import torch_npu
        if torch.npu.is_available():
            torch.npu.manual_seed_all(seed)
    except ImportError:
        pass
