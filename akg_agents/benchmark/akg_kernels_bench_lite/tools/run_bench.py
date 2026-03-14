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
import os
import platform
import signal
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

DEFAULT_BENCH_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scoring import compute_case_score, compute_weighted_score, get_tier_weight

# ──────────────────────── 精度与性能参数 (参考业界标准) ────────────────────────

DEFAULT_RTOL = 1e-2
DEFAULT_ATOL = 1e-2
DEFAULT_WARMUP_RUNS = 10
DEFAULT_ITERATIONS = 100
DEFAULT_NUM_TRIALS = 3
DEFAULT_TIMEOUT = 300  # 每个 case 最大执行秒数


# ──────────────────────────── 动态加载模块 ─────────────────────────────────────

def _load_module(py_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, py_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载模块: {py_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ──────────────────────────── 精度验证 ─────────────────────────────────────────

def check_correctness(
    ref_outputs: Any,
    sol_outputs: Any,
    rtol: float,
    atol: float,
) -> dict:
    """验证 solution 输出与 reference 输出的一致性。

    支持单 tensor、tuple/list of tensors。
    返回 {"correct": bool, "max_abs_diff": float, "max_rel_diff": float, "detail": str}
    """
    if isinstance(ref_outputs, torch.Tensor):
        ref_list = [ref_outputs]
        sol_list = [sol_outputs] if isinstance(sol_outputs, torch.Tensor) else [sol_outputs]
    elif isinstance(ref_outputs, (tuple, list)):
        ref_list = list(ref_outputs)
        sol_list = list(sol_outputs) if isinstance(sol_outputs, (tuple, list)) else [sol_outputs]
    else:
        return {
            "correct": False,
            "max_abs_diff": float("inf"),
            "max_rel_diff": float("inf"),
            "detail": f"不支持的输出类型: {type(ref_outputs)}",
        }

    if len(ref_list) != len(sol_list):
        return {
            "correct": False,
            "max_abs_diff": float("inf"),
            "max_rel_diff": float("inf"),
            "detail": f"输出数量不匹配: ref={len(ref_list)}, sol={len(sol_list)}",
        }

    max_abs = 0.0
    max_rel = 0.0

    for i, (ref_t, sol_t) in enumerate(zip(ref_list, sol_list)):
        if not isinstance(ref_t, torch.Tensor) or not isinstance(sol_t, torch.Tensor):
            return {
                "correct": False,
                "max_abs_diff": float("inf"),
                "max_rel_diff": float("inf"),
                "detail": f"输出[{i}] 不是 Tensor",
            }
        if ref_t.shape != sol_t.shape:
            return {
                "correct": False,
                "max_abs_diff": float("inf"),
                "max_rel_diff": float("inf"),
                "detail": f"输出[{i}] shape 不匹配: ref={ref_t.shape}, sol={sol_t.shape}",
            }

        ref_f = ref_t.float()
        sol_f = sol_t.float()
        abs_diff = (ref_f - sol_f).abs()
        rel_diff = abs_diff / (ref_f.abs() + 1e-8)

        max_abs = max(max_abs, abs_diff.max().item())
        max_rel = max(max_rel, rel_diff.max().item())

    correct = (max_abs <= atol) and (max_rel <= rtol)
    detail = "PASS" if correct else f"max_abs_diff={max_abs:.6e}, max_rel_diff={max_rel:.6e}"

    return {
        "correct": correct,
        "max_abs_diff": max_abs,
        "max_rel_diff": max_rel,
        "detail": detail,
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

    for _ in range(warmup_runs):
        fn(*args)
    _sync_device(device)

    trial_times = []
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

        inputs = get_inputs()
        if device != "cpu":
            inputs = [
                x.to(device) if isinstance(x, torch.Tensor) else x
                for x in inputs
            ]

        # ── 正确性验证 ──
        with torch.no_grad():
            ref_out = ref_model(*inputs)
            sol_out = sol_model(*inputs)

        corr = check_correctness(ref_out, sol_out, rtol=rtol, atol=atol)
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

        with torch.no_grad():
            baseline = measure_latency(
                ref_fn, inputs,
                warmup_runs=warmup_runs,
                iterations=iterations,
                num_trials=num_trials,
            )
            solution = measure_latency(
                sol_fn, inputs,
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
        "timestamp": datetime.now(timezone.utc).isoformat(),
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
    print(f"精度: rtol={args.rtol}, atol={args.atol}")
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
