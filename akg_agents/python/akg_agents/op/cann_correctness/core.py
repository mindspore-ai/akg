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

        first_idx = int(mismatch_mask.reshape(-1).nonzero()[0].item())
        out_value = output.reshape(-1)[first_idx].item()
        golden_value = golden.reshape(-1)[first_idx].item()
        diff_value = diff.reshape(-1)[first_idx].item()
        return {"passed": False, "mere": 0.0, "mare": 0.0, "threshold": threshold,
                "error_msg": (
                    f"integer mismatch: {mismatch_count}/{output.numel()} elements exceed tolerance {threshold}; "
                    f"first_mismatch_flat_index={first_idx}, output={out_value}, "
                    f"golden={golden_value}, diff={diff_value}"
                )}

    # --- Floating-point MERE/MARE path ---
    target_dtype = output.dtype
    golden_truncated = golden.to(target_dtype).double()
    output_fp64 = output.double()

    # NaN handling
    if torch.any(torch.isnan(output_fp64)) or torch.any(torch.isnan(golden_truncated)):
        nan_out = torch.isnan(output_fp64)
        nan_gold = torch.isnan(golden_truncated)
        if not torch.all(nan_out == nan_gold):
            nan_diff = nan_out != nan_gold
            first_idx = int(nan_diff.reshape(-1).nonzero()[0].item())
            out_nan_count = int(nan_out.sum().item())
            golden_nan_count = int(nan_gold.sum().item())
            return {"passed": False, "mere": 0.0, "mare": 0.0, "threshold": threshold,
                    "error_msg": (
                        "NaN position mismatch: "
                        f"output_nan={out_nan_count}, golden_nan={golden_nan_count}, "
                        f"first_mismatch_flat_index={first_idx}, "
                        f"output_is_nan={bool(nan_out.reshape(-1)[first_idx].item())}, "
                        f"golden_is_nan={bool(nan_gold.reshape(-1)[first_idx].item())}"
                    )}

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

    error_msg = None
    if not passed:
        debug_re = torch.where(
            valid_mask,
            relative_error,
            torch.full_like(relative_error, -1.0),
        )
        worst_idx = int(torch.argmax(debug_re.reshape(-1)).item())
        error_msg = (
            f"floating mismatch: normal={normal_mismatch_count}, "
            f"small_value={small_value_error_count}/{small_value_total_count}, "
            f"cancel={cancel_error_count}/{cancel_total_count}; "
            f"worst_flat_index={worst_idx}, output={output_fp64.reshape(-1)[worst_idx].item()}, "
            f"golden={golden_truncated.reshape(-1)[worst_idx].item()}, "
            f"diff={diff.reshape(-1)[worst_idx].item()}, "
            f"relative_error={relative_error.reshape(-1)[worst_idx].item()}"
        )

    return {"passed": passed, "mere": mere, "mare": mare, "threshold": threshold, "error_msg": error_msg}


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


def dual_reference(model, inputs):
    """Compute the two references CANN-Bench's precision gate needs from a single
    reference callable, so the (already-identical) compare engine gets identical
    inputs to CANN-Bench:

    - golden: the reference at **FP64 on CPU** (CANN-Bench's fp64-CPU golden).
      NPU has no fp64, so the golden must run on CPU; floating inputs are moved
      to CPU and upcast to double.
    - native: the reference at the **target precision on CPU** (the same-precision
      reference the small-value / cancellation excusal compares against). It must
      be an INDEPENDENT reference, NOT the candidate's own path — if it were run
      on NPU it could coincide with a candidate that delegates to the same NPU
      builtins and vacuously pass the excusal, defeating the gate.

    Returns ``(golden_list, native_list)`` (each normalized to a list). Non-float
    inputs (indices / scalars / shapes) are passed through unchanged. Runs the
    reference on CPU (the golden formula is device-agnostic torch), matching
    CANN-Bench's fp64-CPU golden + CPU same-precision native.
    """
    def _to_cpu(x, fp64):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu()
            return x.double() if (fp64 and x.is_floating_point()) else x
        return x

    cpu_model = getattr(model, "cpu", lambda: model)()

    # native: target-precision reference on CPU (compute before upcasting model).
    native_inputs = [_to_cpu(x, fp64=False) for x in inputs]
    native = cpu_model(*native_inputs)

    # golden: FP64 reference on CPU.
    fp64_model = getattr(cpu_model, "double", lambda: cpu_model)()
    fp64_inputs = [_to_cpu(x, fp64=True) for x in inputs]
    golden = fp64_model(*fp64_inputs)

    return _normalize_outputs(golden), _normalize_outputs(native)


def assert_outputs(impl_out, golden_out, *, index, dtype=None, native_out=None,
                   op_name=None, impl_inputs=None, all_impl_outputs=None) -> None:
    """One-output CANN-Bench accuracy gate for a generated verify script: print
    one ``[cannbench_precision]`` line and raise ``AssertionError`` (a numerical
    kernel_miss) on failure, so the template stays a one-line call.

    ``golden_out`` should be the FP64 golden and ``native_out`` the same-precision
    reference (see :func:`dual_reference`); passing ``native_out`` makes the
    small-value / cancellation excusal use the same NPU/CPU-error ratio
    CANN-Bench's scoring uses. ``dtype`` names the target output precision (taken
    from ``native_out`` when available, else ``golden_out``).

    If ``op_name``'s output ``index`` is a registered index output (see
    :data:`_INDEX_GATHER_REGISTRY`), the element-wise MERE/MARE compare is
    replaced by CANN-Bench's tie-order-independent index_gather check —
    ``x.gather(dim, idx) == value_output`` — using ``impl_inputs`` /
    ``all_impl_outputs`` for x, dim and the paired value output.

    If ``op_name`` carries a per-op ``precision_thresholds`` block in its
    CANN-Bench proto (see :data:`_PRECISION_THRESHOLD_REGISTRY`), those dtype
    thresholds override the strict defaults for the MERE/MARE compare (e.g.
    int8 ±1 for quantized outputs), matching CANN-Bench's own scoring band.

    If ``op_name``'s output ``index`` is registered as compare-skipped (see
    :data:`_IGNORE_OUTPUT_REGISTRY` — proto ``compare: false`` WITHOUT an
    ``index_gather`` block, e.g. pure position indices), it is not checked at
    all, matching CANN-Bench's ``ignore_output_indices``."""
    if _is_ignored_output(op_name, index):
        print(f"[cannbench_precision] output={index} skipped=True "
              f"(proto compare:false)")
        return
    spec = _index_gather_spec(op_name, index)
    if spec is not None and impl_inputs is not None and all_impl_outputs is not None:
        inp = list(impl_inputs) if isinstance(impl_inputs, (list, tuple)) else [impl_inputs]
        outs = list(all_impl_outputs) if isinstance(all_impl_outputs, (list, tuple)) \
            else [all_impl_outputs]
        x = inp[spec["input"]]
        dim = int(inp[spec["dim_arg"]])
        vals = outs[spec["value_output"]]
        ok, msg = validate_index_output(x, dim, impl_out, vals)
        print(f"[cannbench_index_gather] output={index} "
              f"value_output={spec['value_output']} passed={ok}"
              + ("" if ok else f" msg={msg}"))
        if not ok:
            raise AssertionError(f"index_gather (output {index}): {msg}")
        return

    if dtype is None:
        ref = native_out if native_out is not None else golden_out
        d = getattr(ref, "dtype", None)
        dtype = str(d).replace("torch.", "") if d is not None else "float32"
    custom_thresholds = _precision_threshold_spec(op_name)
    r = compare_tensors(impl_out, golden_out, dtype=dtype, native_output=native_out,
                        custom_thresholds=custom_thresholds)
    print(f"[cannbench_precision] output={index} dtype={dtype} "
          f"mere={r['mere']:.6e} mare={r['mare']:.6e} "
          f"threshold={r['threshold']:.6e} passed={r['passed']}")
    if not r["passed"]:
        raise AssertionError(
            r.get("error_msg")
            or f"CANN-Bench MERE/MARE miss (output {index}, dtype {dtype})")


# ---------------------------------------------------------------------------
# Index-output "pointed-value" check (from CANN-Bench eval/index_check.py,
# issue #40). TopK / ArgSort / Cummin etc. return an index output whose golden
# (torch.topk) tie order is non-deterministic, so an element-wise index compare
# fails on ties even for a correct kernel. CANN-Bench marks the index output
# ``compare: false`` and instead checks, tie-order-independently, that the
# candidate's indices point to its own value output:
#     x.gather(dim, idx_candidate) == values_candidate
# Combined with the value output's own MERE/MARE check against golden this fully
# validates the index (wrong values -> value check fails; garbage indices ->
# this check fails; only-tie-order-differs -> both pass).
# ---------------------------------------------------------------------------


def validate_index_output(
    x: torch.Tensor,
    dim: int,
    idx_candidate: torch.Tensor,
    values_candidate: torch.Tensor,
) -> Tuple[bool, str]:
    """Check that ``x.gather(dim, idx)`` equals ``values`` element-wise (NaN-aware),
    independent of tie order. Returns ``(ok, msg)``. Transplanted verbatim from
    CANN-Bench ``eval/index_check.py::validate_index_output``."""
    if not isinstance(x, torch.Tensor) or not isinstance(idx_candidate, torch.Tensor) \
            or not isinstance(values_candidate, torch.Tensor):
        return False, "index_gather: x / idx / values must all be Tensors"

    x_c = x.detach().cpu()
    idx_c = idx_candidate.detach().cpu().to(torch.int64)
    val_c = values_candidate.detach().cpu()

    if idx_c.dim() != x_c.dim():
        return False, f"index_gather: idx ndim {idx_c.dim()} != x ndim {x_c.dim()}"
    dim_n = dim if dim >= 0 else x_c.dim() + dim
    if not (0 <= dim_n < x_c.dim()):
        return False, f"index_gather: dim={dim} out of range (x is {x_c.dim()}D)"

    if idx_c.shape != val_c.shape:
        return False, (f"index_gather: idx shape {tuple(idx_c.shape)} != value shape "
                       f"{tuple(val_c.shape)}")

    for d in range(x_c.dim()):
        if d != dim_n and idx_c.size(d) > x_c.size(d):
            return False, (f"index_gather: dim {d} idx size {idx_c.size(d)} > x "
                           f"{x_c.size(d)}")

    dim_size = x_c.size(dim_n)
    if idx_c.numel() > 0:
        lo = int(idx_c.min().item())
        hi = int(idx_c.max().item())
        if lo < 0 or hi >= dim_size:
            return False, f"index_gather: idx out of range [{lo},{hi}], valid [0,{dim_size})"

    gathered = torch.gather(x_c, dim_n, idx_c).to(val_c.dtype)
    # NaN-aware (an all-NaN case with correct indices must pass; torch.equal would
    # reject it since NaN != NaN).
    if gathered.is_floating_point() or val_c.is_floating_point():
        both_nan = torch.isnan(gathered) & torch.isnan(val_c)
        eq = (gathered == val_c) | both_nan
    else:
        eq = gathered == val_c
    if not bool(eq.all()):
        mism = int((~eq).sum().item())
        return False, (f"index_gather: candidate indices point to elements that "
                       f"differ from its value output ({mism} mismatches)")
    return True, ""


# op_name -> {index_output_pos: {"input", "dim_arg", "value_output"}}, all
# positional against the op's own call args / outputs. This mirrors the
# ``compare: false`` + ``index_gather`` blocks in CANN-Bench's proto.yaml for
# ops whose index output is tie-non-deterministic. Add an entry per such op.
_INDEX_GATHER_REGISTRY: Dict[str, Dict[int, Dict[str, int]]] = {
    # top_k(x, k, dim, largest) -> (values, idx): idx is output 1; validate it
    # points into x (arg 0) along dim (arg 2) reproducing the value output (0).
    "top_k": {1: {"input": 0, "dim_arg": 2, "value_output": 0}},
    "TopK": {1: {"input": 0, "dim_arg": 2, "value_output": 0}},
}


def _index_gather_spec(op_name: Optional[str], index: int) -> Optional[Dict[str, int]]:
    if not op_name:
        return None
    return _INDEX_GATHER_REGISTRY.get(op_name, {}).get(index)


# op_name (normalized: lowercased, underscores removed) -> set of output indices
# that CANN-Bench's proto marks ``compare: false`` WITHOUT an ``index_gather``
# block, i.e. skipped entirely (no candidate-vs-golden check). Distinct from
# _INDEX_GATHER_REGISTRY: those index outputs still get tie-order-independent
# gather validation; these are pure position indices with no gather relation to
# any value output, so CANN-Bench ignores them outright.
_IGNORE_OUTPUT_REGISTRY: Dict[str, set] = {
    # moe_gating_top_k_softmax(x, finished, k) -> (y, expert_idx, row_idx):
    # y is softmax(x) top-k values, so x.gather(dim, expert_idx) != y (no gather
    # check applies); row_idx is a bare row position. Proto marks both
    # compare:false with no index_gather -> skip outputs 1 and 2.
    "moegatingtopksoftmax": {1, 2},
}


def _is_ignored_output(op_name: Optional[str], index: int) -> bool:
    if not op_name:
        return False
    return index in _IGNORE_OUTPUT_REGISTRY.get(op_name.lower().replace("_", ""), set())


# ---------------------------------------------------------------------------
# Per-op precision-threshold overrides (from CANN-Bench proto.yaml
# ``operator.precision_thresholds``). The autoresearch scaffold generates the
# verify script from reference.py/task.yaml, NOT proto.yaml, so these per-op
# relaxations are otherwise dropped and the gate uses the strict dtype defaults
# in PRECISION_THRESHOLDS — spuriously failing e.g. quantized (int8 ±1) or
# division/normalization ops that CANN-Bench itself scores with a wider band.
# Keys are normalized (lowercased, underscores removed) so one entry matches
# both the snake_case op name and proto's CamelCase operator.name. Values are
# {lowercase-dtype: threshold}, forwarded verbatim to compare_tensors'
# custom_thresholds (a threshold applies to every output of that dtype).
# Mirror proto.yaml exactly; add an entry per op that carries the block.
# ---------------------------------------------------------------------------
_PRECISION_THRESHOLD_REGISTRY: Dict[str, Dict[str, float]] = {
    "foreachaddcdivscalar": {"float32": 0.005, "float16": 0.01, "bfloat16": 0.01},
    "applyadamw":           {"float32": 0.005, "float16": 0.01, "bfloat16": 0.01},
    "applyrotaryposemb":    {"float32": 0.005, "float16": 0.01, "bfloat16": 0.01},
    "dynamicquant":         {"int8": 1.0},
    "groupnorm":            {"float16": 0.005, "float32": 0.005, "bfloat16": 0.01},
    "unsortedsegmentsum":   {"float32": 0.001},
    "addrmsnormdynamicquant": {"int8": 1.0},
    "dequantswigluquant":   {"int8": 1.0},
    "roialign":             {"float16": 0.1, "float32": 0.01},
    "unique":               {"bfloat16": 0.0, "float16": 0.0, "float32": 0.0, "int8": 0.0},
    "groupedmatmulswigluquant": {"int8": 1.0, "float32": 0.001},
    "gru":                  {"float32": 0.05, "float16": 0.05, "bfloat16": 0.05},
    "lstm":                 {"float32": 0.05, "float16": 0.05, "bfloat16": 0.05},
}


def _precision_threshold_spec(op_name: Optional[str]) -> Optional[Dict[str, float]]:
    """Per-op ``custom_thresholds`` for :func:`compare_tensors`, or None. Matches
    ``op_name`` case- and underscore-insensitively (snake_case == CamelCase)."""
    if not op_name:
        return None
    return _PRECISION_THRESHOLD_REGISTRY.get(op_name.lower().replace("_", ""))


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
