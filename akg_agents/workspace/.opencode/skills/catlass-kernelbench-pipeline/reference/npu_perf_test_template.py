"""NPU 精度 + device 侧性能测试（与 akg KernelVerifier compare 标准对齐）。

用法：复制到交付目录（与 reference.py / kernel.py 同级），勿放进 catlass_op/。

  cp reference/npu_perf_test_template.py output/<task>/test_<task>.py
  cd output/<task> && python test_<task>.py --dtype auto
  python test_<task>.py --dtype float16 --warmup 25 --active 50 --skip-perf
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import torch
import torch_npu

# ---------------------------------------------------------------------------
# Compare（精简版，容差表与 akg FrameworkAdapterTorch 一致）
# ---------------------------------------------------------------------------

_DTYPE_ALIASES = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


def _get_tolerance(data_type: torch.dtype):
    """与 akg_agents/op/verifier/adapters/framework/torch.py 相同。"""
    if data_type == torch.float32:
        return (1.22e-4, 1e-5, 1.22e-3, 1e-4, 0.001)
    if data_type == torch.float16:
        return (9.77e-4, 1e-3, 9.77e-3, 1e-2, 0.005)
    if data_type == torch.bfloat16:
        return (7.81e-3, 1e-2, 7.81e-2, 1e-1, 0.01)
    return (1.22e-4, 1e-5, 1.22e-3, 1e-4, 0.001)


def compare(ref_out: torch.Tensor, impl_out: torch.Tensor, data_type: torch.dtype) -> None:
    """分层容差：strict + outlier 比例上限（对齐 NPUKernelBench / CANN）。"""
    ref = ref_out.detach().cpu()
    impl = (
        impl_out.detach().cpu()
        if isinstance(impl_out, torch.Tensor)
        else torch.tensor(impl_out, dtype=ref.dtype)
    )

    if ref.shape != impl.shape:
        raise AssertionError(f"shape mismatch: ref={ref.shape}, impl={impl.shape}")

    for name, a, b in (("NaN", torch.isnan(ref), torch.isnan(impl)), ("Inf", torch.isinf(ref), torch.isinf(impl))):
        if not torch.equal(a, b):
            raise AssertionError(f"{name} mask mismatch")

    finite = torch.isfinite(ref) & torch.isfinite(impl)
    if not finite.any():
        print("[precision] all non-finite, skip numeric check")
        return

    ref_f = ref[finite]
    impl_f = impl[finite]
    if ref_f.dtype == torch.bool:
        if not torch.equal(ref_f, impl_f):
            raise AssertionError("bool mismatch")
        return
    if impl_f.dtype != ref_f.dtype:
        impl_f = impl_f.to(ref_f.dtype)

    rtol, atol, out_rtol, out_atol, out_ratio = _get_tolerance(data_type)
    diff = torch.abs(ref_f.float() - impl_f.float())
    abs_ref = torch.abs(ref_f.float())
    strict = diff <= (atol + rtol * abs_ref)
    relaxed = diff <= (out_atol + out_rtol * abs_ref)

    hard = int((~relaxed).sum().item())
    outlier = int(((~strict) & relaxed).sum().item())
    total = ref_f.numel()
    cap = int(total * out_ratio)
    mere = float((diff / (abs_ref + atol)).mean().item())
    mare = float((diff / (abs_ref + atol)).max().item())
    print(
        f"[precision] dtype={data_type} total={total} strict={int(strict.sum())} "
        f"outlier={outlier}/{cap} hard={hard} mere={mere:.6e} mare={mare:.6e}"
    )

    if hard > 0:
        idx = torch.where((~relaxed))[0][:3]
        samples = "; ".join(
            f"i={i.item()} ref={ref_f[i]:.4e} impl={impl_f[i]:.4e} diff={diff[i]:.4e}"
            for i in idx
        )
        raise AssertionError(f"hard_fail={hard} rtol={rtol} atol={atol} samples: {samples}")

    if outlier > cap:
        raise AssertionError(f"outlier {outlier} > cap {cap} (ratio={out_ratio})")


def _resolve_dtype(name: str, ref_tensor: torch.Tensor) -> torch.dtype:
    if name == "auto":
        return ref_tensor.dtype
    key = name.lower().replace("-", "")
    if key not in _DTYPE_ALIASES:
        raise ValueError(f"unknown --dtype {name!r}, use auto|float16|float32|bfloat16")
    return _DTYPE_ALIASES[key]


def _as_list(x):
    return x if isinstance(x, (list, tuple)) else [x]


# ---------------------------------------------------------------------------
# Profiler
# ---------------------------------------------------------------------------


def profile_op(forward_fn, label: str, inputs, warmup: int, active: int) -> float:
    for _ in range(warmup):
        forward_fn(*inputs)
    torch.npu.synchronize()

    profile_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"profile_{label}")
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.NPU],
        schedule=torch_npu.profiler.schedule(
            wait=0, warmup=0, active=active, repeat=1, skip_first=1
        ),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(profile_path),
        experimental_config=torch_npu.profiler._ExperimentalConfig(
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            l2_cache=False,
            data_simplification=False,
        ),
    ) as prof:
        for _ in range(1 + active):
            forward_fn(*inputs)
            prof.step()
            torch.npu.synchronize()

    import pandas as pd

    avg_time_us = float("inf")
    for root, _, files in os.walk(profile_path):
        for f in files:
            if f != "op_statistic.csv":
                continue
            df = pd.read_csv(os.path.join(root, f))
            if not all(c in df.columns for c in ("Count", "Total Time(us)")):
                print(f"[WARN] Missing columns in {label} op_statistic.csv")
                continue
            measured = df[df["Count"] % active == 0]
            if measured.empty:
                print(f"[WARN] No valid ops in {label} op_statistic.csv")
                continue
            total_time = measured["Total Time(us)"].sum()
            if pd.isna(total_time) or total_time <= 0:
                print(f"[WARN] Invalid timing in {label}")
                continue
            avg_time_us = total_time / active
            break

    if os.path.exists(profile_path):
        shutil.rmtree(profile_path)
    return avg_time_us


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="Catlass KernelBench NPU verify + perf")
    p.add_argument(
        "--dtype",
        default="auto",
        help="容差 dtype：auto 取 ref 输出 dtype；或 float16/float32/bfloat16",
    )
    p.add_argument("--warmup", type=int, default=25, help="profiler 前预热次数")
    p.add_argument("--active", type=int, default=50, help="profiler active 步数")
    p.add_argument("--skip-perf", action="store_true", help="只做精度，不测性能")
    p.add_argument(
        "--result",
        default="bench_result.json",
        help="结果 JSON 路径（相对本脚本目录）",
    )
    return p.parse_args()


def main():
    args = parse_args()
    task_dir = os.path.dirname(os.path.abspath(__file__))
    catlass_op_dir = os.path.join(task_dir, "catlass_op")
    lib_path = os.path.join(catlass_op_dir, "build", "libcatlass_torch.so")
    if not os.path.isfile(lib_path):
        print(f"[ERROR] missing {lib_path}, build catlass_op first", file=sys.stderr)
        sys.exit(1)
    torch.ops.load_library(lib_path)

    torch.manual_seed(0)
    torch_npu.npu.manual_seed(0)

    from reference import Model as RefModel, get_inputs, get_init_inputs
    from kernel import ModelNew

    device = torch.device("npu")

    init = get_init_inputs()
    ref_model = RefModel(*init).to(device)
    catlass_model = ModelNew(*init)
    inputs = [x.to(device) for x in get_inputs()]

    # Precision: compute on NPU, compare on CPU (compare() handles .cpu() internally)
    ref_out = _as_list(ref_model(*inputs))
    impl_out = _as_list(catlass_model(*inputs))

    if len(ref_out) != len(impl_out):
        raise AssertionError(f"output count mismatch: {len(ref_out)} vs {len(impl_out)}")

    check_dtype = _resolve_dtype(args.dtype, ref_out[0])
    max_diff = 0.0
    for i, (r, c) in enumerate(zip(ref_out, impl_out)):
        compare(r, c, check_dtype)
        d = (r.float() - c.float()).abs().max().item()
        max_diff = max(max_diff, d)
        print(f"[Accuracy] output[{i}] ok, max_diff={d:.6e}")

    perf = {}
    if not args.skip_perf:
        catlass_us = profile_op(catlass_model, "catlass", inputs, args.warmup, args.active)
        ref_us = profile_op(ref_model, "ref", inputs, args.warmup, args.active)
        speedup = ref_us / catlass_us if catlass_us > 0 and catlass_us != float("inf") else float("inf")
        print(f"[Catlass] {catlass_us:.1f} us | [Ref] {ref_us:.1f} us | speedup {speedup:.2f}x")
        perf = {
            "catlass_avg_time_us": catlass_us,
            "ref_avg_time_us": ref_us,
            "speedup": speedup,
            "warmup": args.warmup,
            "active": args.active,
        }

    result = {
        "accuracy": {"passed": True, "dtype": str(check_dtype), "max_diff": max_diff},
        "performance": perf,
    }
    out_path = os.path.join(task_dir, args.result)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"[OK] wrote {out_path}")


if __name__ == "__main__":
    main()
