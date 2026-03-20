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
run_ab_test.py - Skill Evolution A/B 测试批量运行器

用法：
  # 运行 Group 1 的 A/B 测试（先 B 后 A，B 中无 skill 命中的算子自动跳过 A）
  python examples/kernel_related/skill_evolution/run_ab_test.py --group 1 --mode both --device 0

  # 仅收集某次运行的结果（必须指定 --run-dir）
  python examples/kernel_related/skill_evolution/run_ab_test.py --collect-only --run-dir ~/akg_eval_results/run_20260309_155303_a1b2 --group 1

  # 清空 tracking.md 中的所有实验数据
  python examples/kernel_related/skill_evolution/run_ab_test.py --clear

每次运行会在 output-dir 下创建带时间戳的目录，保证不同运行互不干扰：
  ~/akg_eval_results/run_20260309_155303_a1b2/
    ├── run_config.json          # 运行元信息（dsl/backend/arch 等）
    ├── group_1_B/               # B 组结果
    ├── group_1_A/               # A 组结果（仅包含 B 中有 skill 命中的算子）
    └── ab_detail_group1.json    # 汇总的 A/B 详细结果
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List


# ============================================================================
# 项目路径
# ============================================================================

def get_project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "python" / "akg_agents").is_dir():
            return parent
    raise FileNotFoundError("无法定位项目根目录")


sys.path.insert(0, str(get_project_root() / "python"))

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ab_test_utils import (  # noqa: E402
    generate_run_dir,
    save_run_config,
    run_group,
    _find_operators_with_skills,
    _run_single_operator,
    collect_group_results,
    update_tracking_md,
    _clear_tracking_md,
    _fmt,
)


# ============================================================================
# 主入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Skill Evolution A/B 测试运行器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--group", type=int, nargs="*", choices=[1, 2, 3, 4],
        help="要测试的 group 编号（可多选；--clear 模式下可选）",
    )
    parser.add_argument(
        "--mode", choices=["A", "B", "both"], default="both",
        help="A=baseline, B=evolved skill, both=先 B 后 A（B 无 skill 命中则跳过 A）",
    )
    parser.add_argument(
        "--device", type=int, default=0,
        help="NPU/GPU 设备 ID（默认 0）",
    )
    parser.add_argument(
        "--max-rounds", type=int, default=3,
        help="evolve 迭代轮数（默认 3）",
    )
    parser.add_argument(
        "--output-dir", default="~/akg_eval_results",
        help="输出根目录（默认 ~/akg_eval_results）",
    )
    parser.add_argument(
        "--evolved-skill-dir",
        default="python/akg_agents/op/resources/skills/triton-ascend/evolved",
        help="evolved SKILL.md 所在目录（B 组使用）",
    )
    parser.add_argument(
        "--agent-config",
        default="op/config/triton_ascend_evolve_config.yaml",
        help="agent 模型配置文件（相对于 python/akg_agents/）",
    )
    parser.add_argument(
        "--log-dir", default="~/akg_agents_logs",
        help="agent 日志根目录（默认 ~/akg_agents_logs）",
    )
    parser.add_argument(
        "--tracking-md", default=None,
        help="tracking.md 路径（默认自动检测）",
    )
    parser.add_argument(
        "--run-dir", default=None,
        help="指定已有的运行目录（用于 --collect-only）",
    )
    parser.add_argument(
        "--collect-only", action="store_true",
        help="仅收集已有结果（必须配合 --run-dir 使用）",
    )
    parser.add_argument(
        "--clear", action="store_true",
        help="清空 tracking.md 中的所有实验数据，恢复初始状态（清空后不运行测试，无需指定 --group）",
    )
    args = parser.parse_args()

    if args.collect_only and not args.run_dir:
        parser.error("--collect-only 必须配合 --run-dir 使用")
    
    if args.clear and args.collect_only:
        parser.error("--clear 和 --collect-only 不能同时使用")
    
    if not args.clear and not args.group:
        parser.error("--group 是必需的（除非使用 --clear）")

    project_root = get_project_root()

    tracking_path = args.tracking_md or str(
        Path(__file__).resolve().parent / "tracking.md"
    )
    
    # --- Clear 模式：清空 tracking.md 后直接退出 ---
    if args.clear:
        print("=" * 80)
        print("Skill Evolution A/B 测试 - 清空数据")
        print("=" * 80)
        print(f"Tracking 文件：{tracking_path}")
        confirm = input("\n确认清空 tracking.md 中的所有实验数据？(y/N): ")
        if confirm.lower() == 'y':
            _clear_tracking_md(tracking_path)
            print("\n清空完成！")
        else:
            print("\n已取消清空操作。")
        return

    # 确定 run_dir
    if args.run_dir:
        run_dir = os.path.expanduser(args.run_dir)
    else:
        run_dir = generate_run_dir(args.output_dir)

    # 读取或生成 run_config
    run_config_path = os.path.join(run_dir, "run_config.json")
    if os.path.isfile(run_config_path):
        with open(run_config_path, encoding="utf-8") as f:
            run_config = json.load(f)
    else:
        save_run_config(run_dir, args)
        with open(run_config_path, encoding="utf-8") as f:
            run_config = json.load(f)

    modes = ["A", "B"] if args.mode == "both" else [args.mode]

    print("=" * 80)
    print("Skill Evolution A/B 测试")
    print("=" * 80)
    print(f"Run Dir: {run_dir}")
    print(f"Groups:  {args.group}")
    print(f"Modes:   {modes}")
    print(f"Rounds:  {args.max_rounds}")
    print(f"Device:  {args.device}")
    print(f"DSL:     {run_config.get('dsl', '')} / {run_config.get('backend', '')} / {run_config.get('arch', '')}")
    if "B" in modes:
        print(f"Evolved: {args.evolved_skill_dir}")
    print("=" * 80)

    # ================================================================
    # Phase 1: 运行测试
    # ================================================================
    if not args.collect_only:
        for group in args.group:
            if args.mode == "both":
                source_dir = (
                    project_root / "benchmark" / "akg_kernels_bench" / "thirdparty" / "pytorch" / f"group_{group}"
                )
                if not source_dir.exists():
                    print(f"  Group {group} 目录不存在: {source_dir}")
                    continue

                task_files = sorted([f for f in os.listdir(source_dir) if f.endswith(".py")])
                print(f"\n  Group {group} 共 {len(task_files)} 个算子，开始逐算子 B→A 测试\n")

                for task_file in task_files:
                    print(f"\n{'─'*80}")
                    print(f"  处理算子: {task_file}")
                    print(f"{'─'*80}")

                    _run_single_operator(group, "B", run_dir, args, project_root, task_file)

                    op_name_with_prefix = "akg_agents_" + task_file.replace(".py", "")
                    has_skill = _find_operators_with_skills(run_dir, group, op_name=op_name_with_prefix)
                    if has_skill:
                        print(f"\n  ✓ {task_file} 在 B 组命中 evolved skill，运行 A 组对比")
                        _run_single_operator(group, "A", run_dir, args, project_root, task_file)
                    else:
                        print(f"\n  ✗ {task_file} 在 B 组未命中 evolved skill，跳过 A 组")
            else:
                for mode in modes:
                    run_group(group, mode, run_dir, args, project_root)

    # ================================================================
    # Phase 2: 收集结果
    # ================================================================
    print(f"\n{'='*80}")
    print("收集实验结果...")
    print(f"{'='*80}")

    all_results: List[Dict[str, Any]] = []
    collect_modes = ["B", "A"] if args.mode == "both" else modes
    for group in args.group:
        for mode in collect_modes:
            group_results = collect_group_results(
                group=group,
                ab_mode=mode,
                run_dir=run_dir,
                log_dir_base=args.log_dir,
                max_rounds=args.max_rounds,
            )
            all_results.extend(group_results)
            for r in group_results:
                sp = _fmt(r.get("best_speedup"))
                status = "OK" if r["success"] else "FAIL"
                skills_str = f" skills={r['selected_skills']}" if r["selected_skills"] else ""
                print(f"  Group {group} {mode} | {r['op_name']:25s} | {status:4s} | speedup={sp}{skills_str}")

    # 保存汇总 JSON
    for group in args.group:
        gr = [r for r in all_results if r["group"] == group]
        if gr:
            detail_path = os.path.join(run_dir, f"ab_detail_group{group}.json")
            with open(detail_path, "w", encoding="utf-8") as f:
                json.dump(gr, f, indent=2, ensure_ascii=False, default=str)
            print(f"  Group {group} 详细结果: {detail_path}")

    # ================================================================
    # Phase 3: 更新 tracking.md
    # ================================================================
    if all_results:
        print(f"\n{'='*80}")
        print("更新 tracking.md ...")
        print(f"{'='*80}")
        update_tracking_md(all_results, tracking_path, run_config)

    # ================================================================
    # Phase 4: 汇总
    # ================================================================
    print(f"\n{'='*80}")
    print("测试完成汇总")
    print(f"{'='*80}")
    for group in args.group:
        for mode in collect_modes:
            gr = [r for r in all_results if r["group"] == group and r["ab_mode"] == mode]
            if not gr:
                continue
            n_ok = sum(1 for r in gr if r["success"])
            speedups = [r["best_speedup"] for r in gr if r.get("best_speedup") is not None]
            avg_sp = sum(speedups) / len(speedups) if speedups else 0
            retries = sum(r.get("total_conductor_retries", 0) for r in gr)
            print(f"  Group {group} {mode}: {n_ok}/{len(gr)} 成功, 平均speedup={avg_sp:.2f}x, conductor重试={retries}")

    print(f"\n运行目录: {run_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
