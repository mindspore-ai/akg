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
KernelBench Level1 算子生成复现 — Skill 系统导入 (kernelgen_only_workflow)

复现目标：
  使用 kernelgen_only_workflow（Skill 系统分阶段动态选择）对 KernelBench Level1
  中指定序号的算子进行端到端代码生成，记录生成结果和性能数据。

导入方式：
  kernelgen_only_workflow — 由 KernelGen 内部按生成阶段（initial / debug /
  optimize）动态调用 LLM 从 SKILL.md 文档库中选择相关 skill，按 category
  分层注入 prompt。不使用固定文档拼接。

默认行为：
  运行 Level1 全部 100 个算子，但排除 54-87 号 conv 算子（triton_ascend
  不支持卷积算子生成）。可通过 --include-conv 包含这些算子。

前置条件：
  - source env.sh
  - API key 已配置（AKG_AGENTS_API_KEY 或 settings.json）
  - Ascend NPU 可用（DEVICE_ID 环境变量，默认 0）
  - KernelBench benchmark 已下载：
      bash akg_agents/download.sh --with_kernelbench

运行方式：
  # 默认运行（排除 conv，pass@1）
  python reproduce/wip/reproduce_kernelbench_kernelgen_skill.py

  # 指定算子序号
  python reproduce/wip/reproduce_kernelbench_kernelgen_skill.py --tasks 1 5 19 42

  # 包含 conv 算子
  python reproduce/wip/reproduce_kernelbench_kernelgen_skill.py --include-conv

  # Pass@3：每个算子独立尝试 3 次
  python reproduce/wip/reproduce_kernelbench_kernelgen_skill.py --pass-n 3

  # 多设备并行 + LLM 并发控制
  python reproduce/wip/reproduce_kernelbench_kernelgen_skill.py --device 4 5 6 7 --concurrency 4 --llm-concurrency 8

可调参数：
  --tasks N [N ...]      KernelBench Level1 任务序号列表（默认全部，排除 conv 54-87）
  --include-conv         包含 54-87 号 conv 算子（默认排除）
  --device ID [ID ...]   NPU 设备 ID，可多个以池化（默认 $DEVICE_ID 或 0）
  --concurrency N        设备并行度上限（默认 4）
  --llm-concurrency N    LLM 请求并发数（默认与 --concurrency 相同）
  --arch ARCH            硬件架构（默认 ascend910b4）
  --pass-n N             Pass@N：每个算子独立运行 N 次（默认 1）
  --output PATH          JSON 报告输出路径
  --profile              开启性能测试（默认关闭；开启后验证通过的算子自动跑 speedup）

输出格式：
  JSON 文件（默认 ~/.akg/reproduce_log/kernelbench_kernelgen_skill_<timestamp>.json），
  包含 benchmark、workflow、pass_n、env_spec、stats.op_results（含 profile）等字段。
  详见 reproduce/SPEC.md 中的 JSON 输出规范。
"""

import argparse
import asyncio

from _common import (
    setup_logging, collect_env_spec, print_env_spec,
    run_benchmark, add_common_args,
    default_output_path,
    ensure_test_utils_importable,
)

BENCHMARK = "KernelBench_Level1"
DEFAULT_WORKFLOW = "kernelgen_only_workflow"
CONV_RANGE = set(range(54, 88))


def _default_task_indices(include_conv: bool) -> list:
    """Level1 全部序号（1-100），默认排除 54-87 conv 算子"""
    all_indices = list(range(1, 101))
    if include_conv:
        return all_indices
    return [i for i in all_indices if i not in CONV_RANGE]


def parse_args():
    parser = argparse.ArgumentParser(
        description="KernelBench Level1 复现 — Skill 系统导入",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--tasks", nargs="+", type=int, default=None,
                        help="KernelBench Level1 任务序号列表（默认全部，排除 conv 54-87）")
    parser.add_argument("--include-conv", action="store_true",
                        help="包含 54-87 号 conv 算子（默认排除）")
    add_common_args(parser)
    return parser.parse_args()


def resolve_ops(args):
    ensure_test_utils_importable()
    from utils import get_kernelbench_op_name, get_kernelbench_task_desc, add_op_prefix

    task_indices = args.tasks if args.tasks else _default_task_indices(args.include_conv)
    names = get_kernelbench_op_name(task_index_list=task_indices, framework="torch")
    if not names:
        raise RuntimeError(f"未找到 KernelBench 任务（tasks={task_indices}）")

    ops = []
    for n in names:
        td = get_kernelbench_task_desc(n, framework="torch")
        ops.append((add_op_prefix(n, benchmark="KernelBench"), td))
    return ops, task_indices


def _benchmark_label(task_indices):
    all_no_conv = set(range(1, 101)) - CONV_RANGE
    if set(task_indices) == all_no_conv:
        return f"{BENCHMARK}_no_Conv"
    if set(task_indices) == set(range(1, 101)):
        return BENCHMARK
    return f"{BENCHMARK}_custom"


async def main():
    setup_logging()
    args = parse_args()

    env_spec = collect_env_spec(args.arch)
    print_env_spec(env_spec)

    ops, task_indices = resolve_ops(args)
    workflow = DEFAULT_WORKFLOW
    output = args.output or default_output_path("kernelbench_kernelgen_skill")

    await run_benchmark(
        script_name="kernelbench_kernelgen_skill",
        workflow=workflow,
        benchmark=_benchmark_label(task_indices),
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
