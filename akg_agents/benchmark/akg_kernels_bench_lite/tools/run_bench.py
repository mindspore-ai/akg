#!/usr/bin/env python3
"""批量运行 bench，对一个目录下的所有参赛选手进行评测。

目录结构约定:
    <submissions_dir>/
    ├── team_a/
    │   ├── meta.json          (可选)
    │   ├── t1/
    │   │   ├── gelu.py
    │   │   └── matmul_basic.py
    │   └── t2/
    │       └── rope.py
    └── team_b/
        └── ...

用法:
    # 跑目录下所有队伍
    python run_bench.py <submissions_dir>

    # 只跑某个队伍
    python run_bench.py <submissions_dir> --team team_a

    # 指定结果输出目录
    python run_bench.py <submissions_dir> --output results/

    # 跳过已有结果 (resume)
    python run_bench.py <submissions_dir> --resume

    # 自定义精度和性能参数
    python run_bench.py <submissions_dir> --rtol 1e-2 --atol 1e-2 --warmup 10 --iterations 100 --num-trials 3
"""

import argparse
import importlib.util
import json
import math
import os
import platform
import signal
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

DEFAULT_BENCH_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = Path(__file__).resolve().parents[3]

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO_ROOT / "examples" / "kernel_related"))
from bench_lite_common import check_torch_outputs
from scoring import compute_case_score, compute_weighted_score, get_tier_weight

# ──────────────────────── 精度与性能参数 ──────────────────────────────────────

DEFAULT_RTOL = 1e-2
DEFAULT_ATOL = 1e-2
DEFAULT_WARMUP_RUNS = 10
DEFAULT_ITERATIONS = 100
DEFAULT_NUM_TRIALS = 3
DEFAULT_TIMEOUT = 300  # 每个 case 最大执行秒数
DEFAULT_CORRECTNESS_TRIALS = 3
EVAL_SEED = 0


# ──────────────────────────── 动态加载模块 ─────────────────────────────────────

def _load_module(py_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, py_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载模块: {py_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _set_eval_seed(seed: int = EVAL_SEED):
    """Reset torch RNGs so ref/solution inputs are value-identical."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    npu_mod = getattr(torch, "npu", None)
    if npu_mod is not None and hasattr(npu_mod, "manual_seed_all"):
        try:
            npu_mod.manual_seed_all(seed)
        except Exception:
            pass


def _move_to_device(obj: Any, device: str) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, list):
        return [_move_to_device(x, device) for x in obj]
    if isinstance(obj, tuple):
        return tuple(_move_to_device(x, device) for x in obj)
    if isinstance(obj, dict):
        return {k: _move_to_device(v, device) for k, v in obj.items()}
    return obj


def _make_seeded_inputs(get_inputs, device: str, seed: int = EVAL_SEED) -> Any:
    _set_eval_seed(seed)
    inputs = get_inputs()
    if device != "cpu":
        inputs = _move_to_device(inputs, device)
    return inputs


# ──────────────────────────── 精度验证 ─────────────────────────────────────────

def check_correctness(
    ref_outputs: Any,
    sol_outputs: Any,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
) -> dict:
    """Use the shared Bench Lite allclose and non-finite checks."""
    return check_torch_outputs(ref_outputs, sol_outputs, rtol=rtol, atol=atol)


def _check_correctness_trials(
    ref_model,
    sol_model,
    get_inputs,
    device: str,
    rtol: float,
    atol: float,
    trials: int = DEFAULT_CORRECTNESS_TRIALS,
) -> dict:
    """Run correctness on multiple random input sets using the same model instances."""
    max_abs = 0.0
    max_rel = 0.0

    for trial_idx in range(trials):
        seed = EVAL_SEED + trial_idx
        ref_inputs = _make_seeded_inputs(get_inputs, device, seed=seed)
        sol_inputs = _make_seeded_inputs(get_inputs, device, seed=seed)

        with torch.no_grad():
            ref_out = ref_model(*ref_inputs)
            sol_out = sol_model(*sol_inputs)

        corr = check_correctness(ref_out, sol_out, rtol=rtol, atol=atol)
        max_abs = max(max_abs, corr["max_abs_diff"])
        max_rel = max(max_rel, corr["max_rel_diff"])
        if not corr["correct"]:
            return {
                "correct": False,
                "max_abs_diff": corr["max_abs_diff"],
                "max_rel_diff": corr["max_rel_diff"],
                "detail": f"trial={trial_idx + 1}/{trials}, seed={seed}: {corr['detail']}",
            }

    return {
        "correct": True,
        "max_abs_diff": max_abs,
        "max_rel_diff": max_rel,
        "detail": f"PASS ({trials} random trials)",
    }


# ──────────────────────────── 性能测量 ──────────────────────────────────────────

def measure_latency(
    fn,
    args: list,
    warmup_runs: int = DEFAULT_WARMUP_RUNS,
    iterations: int = DEFAULT_ITERATIONS,
    num_trials: int = DEFAULT_NUM_TRIALS,
) -> dict:
    """测量 kernel 延迟。多轮试验取中位数，每轮内多次迭代取平均。"""
    device = _get_device()

    trial_times = []
    timing_method = "cuda_synchronized_perf_counter" if device == "cuda" else "cpu_perf_counter"

    if device == "npu":
        from akg_agents.op.verifier.profiler import profiler_npu

        def npu_benchmark_fn():
            return fn(*args)

        for _ in range(num_trials):
            elapsed_us = profiler_npu(
                npu_benchmark_fn,
                warmup=warmup_runs,
                active=iterations,
                prof_dir_name="bench_lite_perf",
                keep_res=False,
                suppress_warnings=True,
                clear_l2_cache=True,
                dsl="triton_ascend",
            )
            elapsed_s = elapsed_us / 1_000_000.0
            if not math.isfinite(elapsed_s) or elapsed_s <= 0:
                raise RuntimeError(f"profiler_npu returned invalid latency: {elapsed_us!r} us")
            trial_times.append(elapsed_s)
        timing_method = "profiler_npu"
    else:
        for _ in range(warmup_runs):
            fn(*args)
        _sync_device(device)

        for _ in range(num_trials):
            _sync_device(device)
            start = time.perf_counter()
            for _ in range(iterations):
                fn(*args)
            _sync_device(device)
            elapsed = time.perf_counter() - start
            trial_times.append(elapsed / iterations)

    trial_times.sort()
    median_s = trial_times[len(trial_times) // 2]
    return {
        "median_ms": median_s * 1000,
        "min_ms": trial_times[0] * 1000,
        "max_ms": trial_times[-1] * 1000,
        "all_trials_ms": [t * 1000 for t in trial_times],
        "timing_method": timing_method,
    }


def _get_device() -> str:
    if torch.npu.is_available() if hasattr(torch, "npu") else False:
        return "npu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _sync_device(device: str):
    if device == "npu" and hasattr(torch, "npu"):
        torch.npu.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


# ──────────────────────────── 单 case 评测 ──────────────────────────────────────

def run_single_case(
    ref_path: Path,
    sol_path: Path,
    tier: str,
    rtol: float,
    atol: float,
    warmup_runs: int,
    iterations: int,
    num_trials: int,
    timeout: int,
) -> dict:
    """评测单个 case，返回结构化结果。"""
    case_name = f"{tier}/{sol_path.stem}"
    result: dict[str, Any] = {"case": case_name, "tier": tier}

    try:
        ref_mod = _load_module(ref_path, f"ref_{ref_path.stem}")
        sol_mod = _load_module(sol_path, f"sol_{sol_path.stem}")

        RefModel = ref_mod.Model
        SolModel = sol_mod.ModelNew
        get_inputs = ref_mod.get_inputs
        get_init_inputs = ref_mod.get_init_inputs

        device = _get_device()
        init_args = get_init_inputs()

        ref_model = RefModel(*init_args)
        sol_model = SolModel(*init_args)

        if device != "cpu":
            ref_model = ref_model.to(device)
            sol_model = sol_model.to(device)

        ref_model.eval()
        sol_model.eval()

        # ── 正确性验证 ──
        corr = _check_correctness_trials(
            ref_model,
            sol_model,
            get_inputs,
            device,
            rtol=rtol,
            atol=atol,
        )
        result["correctness"] = corr["correct"]
        result["max_abs_diff"] = corr["max_abs_diff"]
        result["max_rel_diff"] = corr["max_rel_diff"]
        result["correctness_detail"] = corr["detail"]

        if not corr["correct"]:
            result["status"] = "fail"
            result["score"] = 0.0
            result["weighted_score"] = 0.0
            result["error"] = corr["detail"]
            return result

        # ── 性能测量 ──
        def ref_fn(*a):
            return ref_model(*a)

        def sol_fn(*a):
            return sol_model(*a)

        ref_perf_inputs = _make_seeded_inputs(get_inputs, device)
        sol_perf_inputs = _make_seeded_inputs(get_inputs, device)

        with torch.no_grad():
            baseline = measure_latency(
                ref_fn, ref_perf_inputs,
                warmup_runs=warmup_runs,
                iterations=iterations,
                num_trials=num_trials,
            )
            solution = measure_latency(
                sol_fn, sol_perf_inputs,
                warmup_runs=warmup_runs,
                iterations=iterations,
                num_trials=num_trials,
            )

        speedup = baseline["median_ms"] / solution["median_ms"] if solution["median_ms"] > 0 else 0.0

        result["status"] = "pass"
        result["baseline_ms"] = round(baseline["median_ms"], 4)
        result["solution_ms"] = round(solution["median_ms"], 4)
        result["speedup"] = round(speedup, 4)
        result["score"] = round(compute_case_score(speedup), 2)
        result["weighted_score"] = round(compute_weighted_score(tier, speedup), 2)
        result["timing_method"] = solution["timing_method"]
        result["baseline_detail"] = baseline
        result["solution_detail"] = solution
        result["error"] = None

    except Exception as e:
        result["status"] = "error"
        result["correctness"] = False
        result["score"] = 0.0
        result["weighted_score"] = 0.0
        result["error"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc()

    return result


# ──────────────────────────── 扫描与运行 ──────────────────────────────────────

def discover_teams(submissions_dir: Path) -> list[str]:
    """发现目录下所有队伍。"""
    teams = []
    for d in sorted(submissions_dir.iterdir()):
        if d.is_dir() and not d.name.startswith("."):
            has_tier = any(
                (d / t).is_dir()
                for t in ("t1", "t2", "t3", "t4", "t5")
            )
            if has_tier:
                teams.append(d.name)
    return teams


def discover_tiers(bench_root: Path) -> list[str]:
    return sorted(
        d.name for d in bench_root.iterdir()
        if d.is_dir() and d.name.startswith("t") and d.name[1:].isdigit()
    )


def run_team(
    submissions_dir: Path,
    team: str,
    output_dir: Path,
    bench_root: Path,
    rtol: float,
    atol: float,
    warmup_runs: int,
    iterations: int,
    num_trials: int,
    timeout: int,
    resume: bool,
) -> dict:
    """运行一个队伍的全部 case。"""
    team_dir = submissions_dir / team
    tiers = discover_tiers(bench_root)

    meta = {}
    meta_path = team_dir / "meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

    # resume: 如果结果已存在，加载已有结果
    result_path = output_dir / f"{team}.json"
    existing_cases: dict[str, dict] = {}
    if resume and result_path.exists():
        with open(result_path) as f:
            existing = json.load(f)
            for c in existing.get("cases", []):
                existing_cases[c["case"]] = c

    cases: list[dict] = []
    device = _get_device()

    for tier in tiers:
        ref_tier_dir = bench_root / tier
        sol_tier_dir = team_dir / tier

        if not sol_tier_dir.exists():
            continue

        for sol_file in sorted(sol_tier_dir.glob("*.py")):
            if sol_file.name == "__init__.py":
                continue

            case_key = f"{tier}/{sol_file.stem}"

            if resume and case_key in existing_cases:
                print(f"  [SKIP] {case_key} (已有结果)")
                cases.append(existing_cases[case_key])
                continue

            ref_file = ref_tier_dir / sol_file.name
            if not ref_file.exists():
                cases.append({
                    "case": case_key,
                    "tier": tier,
                    "status": "error",
                    "correctness": False,
                    "score": 0.0,
                    "weighted_score": 0.0,
                    "error": f"题库中不存在对应的 reference: {sol_file.name}",
                })
                continue

            print(f"  [RUN]  {case_key} ...", end=" ", flush=True)
            result = run_single_case(
                ref_path=ref_file,
                sol_path=sol_file,
                tier=tier,
                rtol=rtol,
                atol=atol,
                warmup_runs=warmup_runs,
                iterations=iterations,
                num_trials=num_trials,
                timeout=timeout,
            )
            status_icon = "✓" if result["status"] == "pass" else "✗"
            extra = ""
            if result["status"] == "pass":
                extra = f" speedup={result['speedup']}x score={result['weighted_score']}"
            elif result.get("error"):
                extra = f" {result['error'][:80]}"
            print(f"{status_icon}{extra}")
            cases.append(result)

    # 汇总
    passed = [c for c in cases if c["status"] == "pass"]
    total_weighted = sum(c["weighted_score"] for c in cases)
    avg_speedup = (
        sum(c["speedup"] for c in passed) / len(passed)
        if passed else 0.0
    )

    report = {
        "team_name": meta.get("team_name", team),
        "meta": meta,
        "device": device,
        "timestamp": datetime.now().isoformat(),
        "bench_config": {
            "rtol": rtol,
            "atol": atol,
            "warmup_runs": warmup_runs,
            "iterations": iterations,
            "num_trials": num_trials,
        },
        "cases": cases,
        "summary": {
            "total": len(cases),
            "passed": len(passed),
            "failed": len(cases) - len(passed),
            "total_weighted_score": round(total_weighted, 2),
            "avg_speedup": round(avg_speedup, 4),
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  [DONE] 结果已写入 {result_path}")

    return report


def main():
    parser = argparse.ArgumentParser(description="运行 AKG Bench Lite 评测")
    parser.add_argument(
        "submissions_dir",
        help="包含参赛队伍目录的根目录",
    )
    parser.add_argument("--team", default=None, help="只跑指定队伍")
    parser.add_argument(
        "--bench-dir",
        default=None,
        help="题库根目录 (默认: 脚本所在目录的上级目录)",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="结果输出目录 (默认: bench_dir/results/)",
    )
    parser.add_argument("--resume", action="store_true", help="跳过已有结果的 case")
    parser.add_argument("--rtol", type=float, default=DEFAULT_RTOL, help=f"相对误差容差 (默认: {DEFAULT_RTOL})")
    parser.add_argument("--atol", type=float, default=DEFAULT_ATOL, help=f"绝对误差容差 (默认: {DEFAULT_ATOL})")
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP_RUNS, help=f"预热轮数 (默认: {DEFAULT_WARMUP_RUNS})")
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS, help=f"每轮迭代次数 (默认: {DEFAULT_ITERATIONS})")
    parser.add_argument("--num-trials", type=int, default=DEFAULT_NUM_TRIALS, help=f"试验轮数 (默认: {DEFAULT_NUM_TRIALS})")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help=f"每个 case 超时秒数 (默认: {DEFAULT_TIMEOUT})")
    args = parser.parse_args()

    bench_root = Path(args.bench_dir).resolve() if args.bench_dir else DEFAULT_BENCH_ROOT
    submissions_dir = Path(args.submissions_dir).resolve()
    output_dir = Path(args.output).resolve() if args.output else bench_root / "results"

    if not submissions_dir.exists():
        print(f"[ERROR] 目录不存在: {submissions_dir}")
        sys.exit(1)

    if args.team:
        teams = [args.team]
        if not (submissions_dir / args.team).is_dir():
            print(f"[ERROR] 队伍目录不存在: {submissions_dir / args.team}")
            sys.exit(1)
    else:
        teams = discover_teams(submissions_dir)
        if not teams:
            print(f"[ERROR] 未发现任何队伍目录: {submissions_dir}")
            sys.exit(1)

    print(f"=== AKG Bench Lite ===")
    print(f"题库: {bench_root}")
    print(f"队伍: {', '.join(teams)}")
    print(f"精度: torch.allclose, rtol={args.rtol}, atol={args.atol}, NaN/Inf fail-closed")
    print(f"性能: warmup={args.warmup}, iterations={args.iterations}, trials={args.num_trials}")
    print(f"输出: {output_dir}")
    print()

    for team in teams:
        print(f"── 队伍: {team} ──")
        run_team(
            submissions_dir=submissions_dir,
            team=team,
            output_dir=output_dir,
            bench_root=bench_root,
            rtol=args.rtol,
            atol=args.atol,
            warmup_runs=args.warmup,
            iterations=args.iterations,
            num_trials=args.num_trials,
            timeout=args.timeout,
            resume=args.resume,
        )
        print()

    print("=== 全部完成 ===")


if __name__ == "__main__":
    main()
