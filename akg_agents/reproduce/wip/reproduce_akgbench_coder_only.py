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
AKGBench Lite 算子生成复现 — 固定文档导入 (coder_only_workflow)

复现目标：
  使用 coder_only_workflow（固定文档拼接）对 AKGBench Lite 中的算子进行端到端
  代码生成。适合作为 kernelgen_only_workflow 的基线对照。

导入方式：
  coder_only_workflow — 将固定参考文档直接拼接到 prompt，不经过 Skill 系统。

默认行为：
  运行 t1/t2/t3 全部算子。可通过 --tiers 选择层级，--cases 指定具体算子。

前置条件：
  - 安装 torch_npu、triton_ascend 及 akg_agents 依赖
  - LLM 服务可访问
  - NPU 设备可用
  - benchmark/akg_kernels_bench_lite/ 目录存在

运行方式：
  # 默认运行全部 tier
  python reproduce/wip/reproduce_akgbench_coder_only.py

  # 仅运行 t1 层级
  python reproduce/wip/reproduce_akgbench_coder_only.py --tiers t1

  # 指定具体算子
  python reproduce/wip/reproduce_akgbench_coder_only.py --cases gelu softmax

  # Pass@3
  python reproduce/wip/reproduce_akgbench_coder_only.py --pass-n 3

  # 多设备并行
  python reproduce/wip/reproduce_akgbench_coder_only.py --device 4 5 6 7 --concurrency 4 --llm-concurrency 8

可调参数：
  --tiers T [T ...]      Tier 列表（默认全部 t1 t2 t3）
  --cases NAME [NAME ..] 指定算子名称（不含 .py）；默认运行该 tier 全部算子
  --device ID [ID ...]   NPU 设备 ID，可多个以池化（默认 $DEVICE_ID 或 0）
  --concurrency N        设备并行度上限（默认 4）
  --llm-concurrency N    LLM 请求并发数（默认与 --concurrency 相同）
  --arch ARCH            硬件架构（默认 ascend910b4）
  --pass-n N             Pass@N：每个算子独立运行 N 次（默认 1）
  --output PATH          JSON 报告输出路径
  --profile              开启性能测试（默认关闭；开启后验证通过的算子自动跑 speedup）

输出格式：
  JSON 文件（默认 ~/.akg/reproduce_log/akgbench_lite_coder_only_<timestamp>.json），
  包含 benchmark="AKGBench_Lite"、stats.op_results（含 profile）等字段。
  详见 reproduce/SPEC.md 中的 JSON 输出规范。
"""

import argparse
import asyncio
from pathlib import Path

from _common import (
    setup_logging, collect_env_spec, print_env_spec,
    run_benchmark, add_common_args,
    default_output_path,
    PROJECT_ROOT,
)

BENCHMARK = "AKGBench_Lite"
DEFAULT_WORKFLOW = "coder_only_workflow"


def _get_bench_lite_dir() -> Path:
    d = PROJECT_ROOT / "benchmark" / "akg_kernels_bench_lite"
    if not d.exists():
        raise FileNotFoundError(
            f"未找到 AKGBench Lite 目录: {d}\n"
            "请确认 benchmark/akg_kernels_bench_lite/ 存在。"
        )
    return d


def _discover_cases(bench_dir: Path, tiers: list, cases: list = None) -> list:
    discovered = []
    for tier in tiers:
        tier_dir = bench_dir / tier
        if not tier_dir.is_dir():
            continue
        for f in sorted(tier_dir.iterdir()):
            if not f.suffix == ".py":
                continue
            case_name = f.stem
            if cases and case_name not in cases:
                continue
            discovered.append((case_name, tier, f))
    return discovered


def parse_args():
    parser = argparse.ArgumentParser(
        description="AKGBench Lite 复现 — 固定文档导入",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--tiers", nargs="+", default=["t1", "t2", "t3"],
                        help="Tier 列表（默认全部 t1 t2 t3）")
    parser.add_argument("--cases", nargs="+", default=None,
                        help="指定算子名称（不含 .py）；默认运行该 tier 全部算子")
    add_common_args(parser)
    return parser.parse_args()


def resolve_ops(args):
    bench_dir = _get_bench_lite_dir()
    cases = _discover_cases(bench_dir, args.tiers, args.cases)
    if not cases:
        raise RuntimeError(f"未找到算子（tiers={args.tiers}, cases={args.cases}）")

    ops = []
    for case_name, tier, file_path in cases:
        task_desc = file_path.read_text(encoding="utf-8")
        display_name = f"AKGBench_{tier}_{case_name}"
        ops.append((display_name, task_desc))
    return ops


async def main():
    setup_logging()
    args = parse_args()

    env_spec = collect_env_spec(args.arch)
    print_env_spec(env_spec)

    ops = resolve_ops(args)
    workflow = DEFAULT_WORKFLOW
    output = args.output or default_output_path("akgbench_lite_coder_only")

    await run_benchmark(
        script_name="akgbench_lite_coder_only",
        workflow=workflow,
        benchmark=BENCHMARK,
        ops=ops,
        framework="torch", dsl="triton_ascend", backend="ascend",
        arch=args.arch, device_ids=args.device,
        max_concurrency=args.concurrency,
        llm_concurrency=args.llm_concurrency,
        pass_n=args.pass_n,
        env_spec=env_spec, output_path=output,
        enable_profile=args.profile,
    )


if __name__ == "__main__":
    asyncio.run(main())
