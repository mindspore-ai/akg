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
  使用 coder_only_workflow（固定文档拼接）对 AKGBench Lite（akg_kernels_bench_lite）
  中的算子进行端到端代码生成，记录生成结果和性能数据。

导入方式：
  coder_only_workflow — 由 config 中 docs_dir.coder 指定的固定文档目录，
  将全部文档内容拼接注入 prompt，不经过 Skill 系统的动态选择。

AKGBench Lite 结构：
  benchmark/akg_kernels_bench_lite/
  ├── t1/   (基础算子: gelu, softmax, matmul_basic, ...)
  ├── t2/   (中等复杂度)
  └── t3/   (高复杂度)

前置条件：
  - source env.sh
  - API key 已配置（AKG_AGENTS_API_KEY 或 settings.json）
  - Ascend NPU 可用（DEVICE_ID 环境变量，默认 0）
  - AKGBench Lite 数据目录存在：benchmark/akg_kernels_bench_lite/

运行方式：
  python reproduce/wip/reproduce_akgbench_coder_only.py --help
  python reproduce/wip/reproduce_akgbench_coder_only.py                       # 默认全部 tier
  python reproduce/wip/reproduce_akgbench_coder_only.py --tiers t1            # 只跑 t1
  python reproduce/wip/reproduce_akgbench_coder_only.py --cases gelu softmax  # 只跑指定算子
  python reproduce/wip/reproduce_akgbench_coder_only.py --device 4 --arch ascend910b3

预期输出：
  控制台打印环境规范 + 每个算子的生成结果（pass/fail、耗时、speedup）。
  JSON 报告保存到 --output 指定路径（默认 ~/.akg/reproduce_log/）。

结果存储格式：
  {
    "script": "akgbench_lite_coder_only",
    "workflow": "coder_only_workflow",
    "ops_count": 6, "elapsed_s": 456.7,
    "env_spec": { "arch", "torch_npu", "triton_ascend", "commit", "llm_model", ... },
    "task_log_dir": "~/akg_agents_logs",
    "stats": { ... }
  }
"""

import argparse
import asyncio
import sys
from pathlib import Path

from _common import (
    setup_logging, collect_env_spec, print_env_spec,
    run_benchmark, add_common_args, default_output_path,
    PROJECT_ROOT,
)


def _get_bench_lite_dir() -> Path:
    d = PROJECT_ROOT / "benchmark" / "akg_kernels_bench_lite"
    if not d.exists():
        raise FileNotFoundError(
            f"未找到 AKGBench Lite 目录: {d}\n"
            "请确认 benchmark/akg_kernels_bench_lite/ 存在。"
        )
    return d


def _discover_cases(bench_dir: Path, tiers: list, cases: list = None) -> list:
    """发现 bench_lite 算子，返回 [(case_name, tier, file_path), ...]"""
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
        description="AKGBench Lite 复现 — 固定文档导入 (coder_only_workflow)",
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
        raise RuntimeError(
            f"未找到算子（tiers={args.tiers}, cases={args.cases}）")

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
    output = args.output or default_output_path("akgbench_lite_coder_only")

    await run_benchmark(
        script_name="akgbench_lite_coder_only",
        workflow="coder_only_workflow",
        ops=ops,
        framework="torch", dsl="triton_ascend", backend="ascend",
        arch=args.arch, device_ids=args.device,
        max_concurrency=args.concurrency,
        env_spec=env_spec, output_path=output,
    )


if __name__ == "__main__":
    asyncio.run(main())
