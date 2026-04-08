#!/usr/bin/env python3
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

"""
自适应搜索 (Adaptive Search) 复现脚本

直接调用 adaptive_search() API 对算子逐个进行搜索式生成。
每个算子内部并行探索多个实现方案，使用 UCB 策略选择父代并迭代进化。

调用链：
  本脚本 → adaptive_search() → AdaptiveSearchController
    → 内部创建多个 LangGraphTask(workflow=kernelgen_only_workflow) 子任务

配置方式：
  搜索参数通过 YAML 配置文件控制（与 run_single_adaptive_search.py 共用同一套配置结构）。
  默认使用项目内置 adaptive_search_config.yaml；可通过 --config 指定自定义配置。
  CLI 参数（如 --max-total-tasks）会覆盖 YAML 中的对应值。

用法:
  # KernelBench — 使用默认配置
  python reproduce_adaptive_search.py --benchmark kernelbench

  # 使用自定义配置文件
  python reproduce_adaptive_search.py --benchmark kernelbench --config my_search_config.yaml

  # CLI 覆盖部分参数
  python reproduce_adaptive_search.py --benchmark kernelbench --max-total-tasks 100

  # AKGBench Tier 1-2
  python reproduce_adaptive_search.py --benchmark akgbench --tiers 1 2

  # MHC
  python reproduce_adaptive_search.py --benchmark mhc

可调参数：
  --benchmark BM         必选，可选 kernelbench / akgbench / mhc
  --config PATH          搜索参数 YAML 配置文件（默认项目内置 adaptive_search_config.yaml）
  --device ID [ID ...]   NPU 设备 ID，可多个（默认 $DEVICE_ID 或 0）
  --arch ARCH            硬件架构（默认 ascend910b4）
  --output PATH          JSON 报告输出路径

  benchmark 任务选择（按 benchmark 类型使用）：
    --tasks N [N ...]    [kernelbench] Level1 任务序号列表
    --include-conv       [kernelbench] 包含 54-87 conv 算子
    --tiers N [N ...]    [akgbench] Tier 列表（如 --tiers 1 2 3）
    --cases NAME [...]   [akgbench] 指定 case 名称
    --op N [N ...]       [mhc] 算子序号列表

  CLI 覆盖（优先级高于 config.yaml）：
    --max-total-tasks N       每个算子最大总任务数
    --max-concurrent N        每个算子搜索时的最大并发数
    --initial-task-count N    每个算子的初始任务数
    --exploration-coef F      UCB 探索系数
    --random-factor F         随机因子

  多设备说明：
    adaptive_search() 内部通过 register_local_worker 注册的全局 worker 管理设备。
    传入多个 --device 时，搜索子任务会自动分配到不同设备上。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from _common import (
    setup_logging, collect_env_spec, print_env_spec,
    ensure_test_utils_importable, default_output_path,
    PROJECT_ROOT, TESTS_OP_DIR, DEFAULT_LOG_DIR,
)

DEFAULT_AS_CONFIG = str(PROJECT_ROOT / "python" / "akg_agents" / "op" / "config" / "adaptive_search_config.yaml")
CONV_RANGE = set(range(54, 88))


# ============================================================
# Config loading
# ============================================================

def load_search_config(config_path: str, cli_overrides: dict) -> dict:
    """从 YAML 加载搜索参数，CLI 覆盖优先。返回扁平化 dict。"""
    from akg_agents.utils.common_utils import load_yaml

    y = load_yaml(config_path)
    config_dir = os.path.dirname(os.path.abspath(config_path))

    conc = y.get("concurrency", {})
    stop = y.get("stopping", {})
    ucb = y.get("ucb_selection", {})
    insp = y.get("inspiration", {})
    hw = y.get("handwrite", {})

    params = {
        "max_concurrent": conc.get("max_concurrent", 4),
        "initial_task_count": conc.get("initial_task_count", 4),
        "max_total_tasks": stop.get("max_total_tasks", 50),
        "exploration_coef": ucb.get("exploration_coef", 1.414),
        "random_factor": ucb.get("random_factor", 0.1),
        "use_softmax": ucb.get("use_softmax", False),
        "softmax_temperature": ucb.get("softmax_temperature", 1.0),
        "inspiration_sample_num": insp.get("sample_num", 3),
        "use_tiered_sampling": insp.get("use_tiered_sampling", True),
        "handwrite_sample_num": hw.get("sample_num", 2),
        "handwrite_decay_rate": hw.get("decay_rate", 2.0),
    }

    llm_config_path = y.get("config_path", "")
    if llm_config_path and not os.path.isabs(llm_config_path):
        llm_config_path = os.path.normpath(os.path.join(config_dir, llm_config_path))
    params["_llm_config_path"] = llm_config_path

    for k, v in cli_overrides.items():
        if v is not None:
            params[k] = v

    return params


# ============================================================
# Benchmark task discovery
# ============================================================

def _resolve_kernelbench_ops(args) -> tuple[list, str]:
    ensure_test_utils_importable()
    from utils import get_kernelbench_op_name, get_kernelbench_task_desc, add_op_prefix

    if args.tasks:
        indices = args.tasks
    else:
        indices = [i for i in range(1, 101) if args.include_conv or i not in CONV_RANGE]

    names = get_kernelbench_op_name(task_index_list=indices, framework="torch")
    if not names:
        raise RuntimeError(f"未找到 KernelBench 任务（tasks={indices}）")

    ops = []
    for n in names:
        td = get_kernelbench_task_desc(n, framework="torch")
        ops.append((add_op_prefix(n, benchmark="KernelBench"), td))

    all_no_conv = set(range(1, 101)) - CONV_RANGE
    if set(indices) == all_no_conv:
        label = "KernelBench_Level1_no_Conv"
    elif set(indices) == set(range(1, 101)):
        label = "KernelBench_Level1"
    else:
        label = "KernelBench_Level1_custom"
    return ops, label


def _resolve_akgbench_ops(args) -> tuple[list, str]:
    bench_dir = PROJECT_ROOT / "thirdparty" / "AKGBench_Lite"
    if not bench_dir.exists():
        raise RuntimeError(f"AKGBench_Lite 目录不存在: {bench_dir}")

    tiers = args.tiers or [1, 2, 3]
    cases = []
    for tier in sorted(tiers):
        tier_dir = bench_dir / f"Tier{tier}"
        if not tier_dir.is_dir():
            continue
        for py_file in sorted(tier_dir.glob("*.py")):
            if args.cases and py_file.stem not in args.cases:
                continue
            cases.append((py_file.stem, f"Tier{tier}", py_file))

    if not cases:
        raise RuntimeError(f"未找到 AKGBench 算子（tiers={tiers}）")

    ops = []
    for case_name, tier, fp in cases:
        ops.append((f"AKGBench_{tier}_{case_name}", fp.read_text(encoding="utf-8")))
    return ops, "AKGBench_Lite"


def _resolve_mhc_ops(args) -> tuple[list, str]:
    ensure_test_utils_importable()
    from utils import get_evokernel_mhc_op_name, get_evokernel_task_desc, add_op_prefix

    MHC_OP_NAMES = [
        "flash_attention_v2", "flash_attention_triton",
        "context_flashattention_noalibi", "incre_flashattention_noalibi",
        "multi_head_attention_forward", "paged_attention",
    ]

    if args.op is not None:
        names = [MHC_OP_NAMES[i] for i in args.op if i < len(MHC_OP_NAMES)]
    else:
        names = get_evokernel_mhc_op_name()

    if not names:
        raise RuntimeError("未找到 MHC 算子")

    ops = []
    for n in names:
        td = get_evokernel_task_desc(n, category="MHC")
        ops.append((add_op_prefix(f"MHC_{n}", benchmark="EvoKernel"), td))
    return ops, "EvoKernel_MHC"


def resolve_ops(args) -> tuple[list, str]:
    bm = args.benchmark.lower()
    if bm == "kernelbench":
        return _resolve_kernelbench_ops(args)
    elif bm == "akgbench":
        return _resolve_akgbench_ops(args)
    elif bm == "mhc":
        return _resolve_mhc_ops(args)
    else:
        raise ValueError(f"未知 benchmark: {args.benchmark}（可选: kernelbench, akgbench, mhc）")


# ============================================================
# Argparse
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Adaptive Search 复现 — 直接调用 adaptive_search() API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument("--benchmark", required=True,
                   choices=["kernelbench", "akgbench", "mhc"],
                   help="Benchmark 名称")
    p.add_argument("--config", default=None,
                   help=f"搜索参数配置文件（YAML，默认使用项目内置 adaptive_search_config.yaml）")

    bm_group = p.add_argument_group("benchmark 任务选择")
    bm_group.add_argument("--tasks", nargs="+", type=int, default=None,
                          help="[kernelbench] Level1 任务序号列表")
    bm_group.add_argument("--include-conv", action="store_true",
                          help="[kernelbench] 包含 54-87 conv 算子")
    bm_group.add_argument("--tiers", nargs="+", type=int, default=None,
                          help="[akgbench] Tier 列表（如 --tiers 1 2 3）")
    bm_group.add_argument("--cases", nargs="+", default=None,
                          help="[akgbench] 指定 case 名称")
    bm_group.add_argument("--op", nargs="+", type=int, default=None,
                          help="[mhc] 算子序号列表")

    p.add_argument("--device", nargs="+", type=int,
                   default=[int(os.getenv("DEVICE_ID", "0"))],
                   help="NPU 设备 ID（可多个）")
    p.add_argument("--arch", default="ascend910b4")
    p.add_argument("--output", default=None,
                   help="JSON 报告输出路径")

    ov = p.add_argument_group("CLI 覆盖（优先级高于 config.yaml）")
    ov.add_argument("--max-total-tasks", type=int, default=None,
                    help="每个算子最大总任务数")
    ov.add_argument("--max-concurrent", type=int, default=None,
                    help="每个算子搜索时的最大并发数")
    ov.add_argument("--initial-task-count", type=int, default=None,
                    help="每个算子的初始任务数")
    ov.add_argument("--exploration-coef", type=float, default=None)
    ov.add_argument("--random-factor", type=float, default=None)

    return p.parse_args()


# ============================================================
# Runner
# ============================================================

async def run_adaptive_search_benchmark(
    *,
    benchmark: str,
    ops: list,
    framework: str,
    dsl: str,
    backend: str,
    arch: str,
    device_ids: List[int],
    env_spec: dict,
    output_path: str,
    search_params: dict,
):
    from akg_agents.op.adaptive_search import adaptive_search
    from akg_agents.core.worker.manager import register_local_worker
    from akg_agents.op.config.config_validator import load_config
    from akg_agents.utils.environment_check import check_env_for_task
    from akg_agents.utils.task_label import resolve_task_label

    llm_config_path = search_params.pop("_llm_config_path", "")
    if llm_config_path and os.path.exists(llm_config_path):
        config = load_config(config_path=llm_config_path)
    else:
        config = load_config(dsl=dsl, backend=backend)

    if "agent_model_config" not in config or not isinstance(config.get("agent_model_config"), dict):
        config["agent_model_config"] = {}
    mc = config["agent_model_config"]
    default_level = mc.get("default") or "standard"
    mc.setdefault("default", default_level)
    for agent in ["designer", "coder", "conductor", "verifier", "selector", "op_task_builder"]:
        mc.setdefault(agent, mc["default"])

    check_env_for_task(framework, backend, dsl, config)
    await register_local_worker(device_ids, backend=backend, arch=arch)

    os.environ['AKG_AGENTS_STREAM_OUTPUT'] = 'on'

    sp = search_params
    print(f"  benchmark:          {benchmark}")
    print(f"  算子数量:            {len(ops)}")
    print(f"  策略:                adaptive_search (direct API)")
    print(f"  devices:             {device_ids}")
    print(f"  max_total_tasks:     {sp['max_total_tasks']}")
    print(f"  max_concurrent:      {sp['max_concurrent']}")
    print(f"  initial_task_count:  {sp['initial_task_count']}")
    print(f"  exploration_coef:    {sp['exploration_coef']}")
    print(f"  random_factor:       {sp['random_factor']}\n")

    t0 = time.time()
    op_results: Dict[str, Any] = {}
    total_passed = 0

    for i, (op_name, task_desc) in enumerate(ops):
        print(f"\n{'='*70}")
        print(f"[{i+1}/{len(ops)}] adaptive_search: {op_name}")
        print(f"{'='*70}")

        config["task_label"] = resolve_task_label(op_name=op_name, parallel_index=1)

        try:
            result = await adaptive_search(
                op_name=op_name,
                task_desc=task_desc,
                dsl=dsl,
                framework=framework,
                backend=backend,
                arch=arch,
                config=config,
                max_concurrent=sp["max_concurrent"],
                initial_task_count=sp["initial_task_count"],
                max_total_tasks=sp["max_total_tasks"],
                exploration_coef=sp["exploration_coef"],
                random_factor=sp["random_factor"],
                use_softmax=sp["use_softmax"],
                softmax_temperature=sp["softmax_temperature"],
                inspiration_sample_num=sp["inspiration_sample_num"],
                use_tiered_sampling=sp["use_tiered_sampling"],
                handwrite_sample_num=sp["handwrite_sample_num"],
                handwrite_decay_rate=sp["handwrite_decay_rate"],
            )

            success = bool(result and result.get("total_success", 0) > 0)
            if success:
                total_passed += 1

            entry: Dict[str, Any] = {
                "passed": 1 if success else 0,
                "total": 1,
                "search_stats": {
                    "total_submitted": result.get("total_submitted", 0) if result else 0,
                    "total_success": result.get("total_success", 0) if result else 0,
                    "total_failed": result.get("total_failed", 0) if result else 0,
                    "success_rate": result.get("success_rate", 0) if result else 0,
                    "elapsed_time": result.get("elapsed_time", 0) if result else 0,
                    "stop_reason": result.get("stop_reason", "") if result else "",
                },
            }

            best_impls = (result.get("best_implementations") or []) if result else []
            if best_impls:
                best = best_impls[0]
                profile_data = best.get("profile") or {}
                entry["profile"] = {
                    "gen_time": best.get("gen_time", 0),
                    "base_time": profile_data.get("base_time", 0),
                    "speedup": best.get("speedup", 0),
                }

            op_results[op_name] = entry

            sr = result.get("success_rate", 0) if result else 0
            speedup = best_impls[0].get("speedup", 0) if best_impls else 0
            mark = "✅" if success else "❌"
            print(f"  {mark} 成功率: {sr:.1%}, 最佳加速比: {speedup:.2f}x")

        except Exception as e:
            print(f"  ❌ 异常: {e}")
            op_results[op_name] = {"passed": 0, "total": 1, "error": str(e)}

    elapsed = time.time() - t0

    summary = {
        "benchmark": benchmark,
        "script": "adaptive_search",
        "workflow": "adaptive_search",
        "pass_n": 1,
        "ops_count": len(ops),
        "elapsed_s": round(elapsed, 1),
        "device_ids": device_ids,
        "max_concurrency": sp["max_concurrent"],
        "llm_concurrency": sp["max_concurrent"],
        "env_spec": env_spec,
        "adaptive_search_config": sp,
        "stats": {
            "total_ops": len(ops),
            "passed_ops": total_passed,
            "failed_ops": len(ops) - total_passed,
            "pass_rate": round(total_passed / max(1, len(ops)), 4),
            "op_results": op_results,
        },
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*70}")
    print(f"  完成: adaptive_search @ {benchmark}")
    print(f"  算子数: {len(ops)}  |  总耗时: {elapsed:.1f}s ({elapsed/3600:.2f}h)")
    print(f"  通过率: {total_passed}/{len(ops)} ({total_passed/max(1,len(ops)):.1%})")
    print(f"  报告: {output_path}")
    print(f"{'='*70}\n")

    return summary


# ============================================================
# Main
# ============================================================

async def main():
    setup_logging()
    args = parse_args()

    env_spec = collect_env_spec(args.arch)
    print_env_spec(env_spec)

    config_path = args.config or DEFAULT_AS_CONFIG
    print(f"  搜索配置: {config_path}")

    cli_overrides = {
        "max_total_tasks": args.max_total_tasks,
        "max_concurrent": args.max_concurrent,
        "initial_task_count": args.initial_task_count,
        "exploration_coef": args.exploration_coef,
        "random_factor": args.random_factor,
    }
    search_params = load_search_config(config_path, cli_overrides)

    ops, benchmark_label = resolve_ops(args)
    output = args.output or default_output_path(f"adaptive_search_{args.benchmark}")

    await run_adaptive_search_benchmark(
        benchmark=benchmark_label,
        ops=ops,
        framework="torch",
        dsl="triton_ascend",
        backend="ascend",
        arch=args.arch,
        device_ids=args.device,
        env_spec=env_spec,
        output_path=output,
        search_params=search_params,
    )


if __name__ == "__main__":
    asyncio.run(main())
