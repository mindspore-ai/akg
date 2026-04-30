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
NPUKernelBench 算子生成复现 — Skill 系统导入 (kernelgen_only_workflow)

复现目标：
  使用 kernelgen_only_workflow 对 NPUKernelBench 中各个 level 的算子进行
  端到端代码生成。NPUKernelBench 内置 JSONL 动态 shape 列表，
  tests/op/utils.py 的 get_npukernelbench_task_desc 会把每个 case 的
  ``get_input_groups`` 自动 alias 为 AKG 要求的 ``get_inputs_dyn_list``，
  并把 .json 里的 JSONL 内容内联到 task_desc，避免 verify_dir 丢失输入集。

导入方式：
  kernelgen_only_workflow — 由 KernelGen 按生成阶段动态选择 Skill 注入 prompt。

默认行为：
  默认运行 level1 全部算子。可通过 --levels 选择层级，--tasks 指定序号。

前置条件：
  - source env.sh
  - API key 已配置（AKG_AGENTS_API_KEY 或 settings.json）
  - Ascend NPU 可用（DEVICE_ID 环境变量，默认 0）
  - NPUKernelBench benchmark 已就位：
      thirdparty/AscendOpGenAgent/benchmarks/NPUKernelBench/level*/*.py

运行方式：
  # 默认运行 level1 全部算子
  python reproduce/wip/reproduce_npukernelbench_kernelgen_skill.py

  # 指定多个 level
  python reproduce/wip/reproduce_npukernelbench_kernelgen_skill.py --levels level1 level2

  # 指定算子序号（仅与单个 level 搭配使用）
  python reproduce/wip/reproduce_npukernelbench_kernelgen_skill.py --levels level1 --tasks 1 2 3

  # Pass@3
  python reproduce/wip/reproduce_npukernelbench_kernelgen_skill.py --pass-n 3

  # 多设备并行
  python reproduce/wip/reproduce_npukernelbench_kernelgen_skill.py --device 4 5 6 7 --concurrency 4 --llm-concurrency 8

可调参数：
  --levels LV [LV ...]  NPUKernelBench level 列表（默认 level1）
  --tasks N [N ...]     指定算子序号（例如 1 2 3）；仅与单个 --levels 搭配
  --device ID [ID ...]  NPU 设备 ID，可多个以池化（默认 $DEVICE_ID 或 0）
  --concurrency N       设备并行度上限（默认 4）
  --llm-concurrency N   LLM 请求并发数（默认与 --concurrency 相同）
  --arch ARCH           硬件架构（默认 ascend910b4）
  --pass-n N            Pass@N：每个算子独立运行 N 次（默认 1）
  --output PATH         JSON 报告输出路径
  --profile             开启性能测试（默认关闭）

输出格式：
  JSON 文件（默认 ~/.akg/reproduce_log/npukernelbench_kernelgen_skill_<timestamp>.json），
  包含 benchmark="NPUKernelBench_<levels>"、stats.op_results（含 profile）等字段。
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

BENCHMARK_PREFIX = "NPUKernelBench"
DEFAULT_WORKFLOW = "kernelgen_only_workflow"
DEFAULT_LEVELS = ["level1"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="NPUKernelBench 复现 — Skill 系统导入",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--levels", nargs="+", default=DEFAULT_LEVELS,
        choices=["level1", "level2", "level3", "level4"],
        help="NPUKernelBench level 列表（默认 level1）",
    )
    parser.add_argument(
        "--tasks", nargs="+", type=int, default=None,
        help="指定算子序号（仅与单个 --levels 搭配；不指定则跑该 level 全部算子）",
    )
    add_common_args(parser)
    return parser.parse_args()


def resolve_ops(args):
    ensure_test_utils_importable()
    from utils import (
        get_npukernelbench_op_name, get_npukernelbench_task_desc, add_op_prefix,
    )

    if args.tasks is not None and len(args.levels) != 1:
        raise RuntimeError(
            "--tasks 仅支持与单个 --levels 搭配使用；"
            f"当前 --levels={args.levels}",
        )

    ops = []
    for level in args.levels:
        names = get_npukernelbench_op_name(
            task_index_list=args.tasks, level=level,
        )
        if not names:
            raise RuntimeError(
                f"未找到 NPUKernelBench 任务（level={level}, tasks={args.tasks}）",
            )
        for name in names:
            task_desc, aux_files, factory_names = get_npukernelbench_task_desc(
                name, level=level,
            )
            display_name = add_op_prefix(
                f"{level}_{name}", benchmark=BENCHMARK_PREFIX,
            )
            ops.append((display_name, task_desc, aux_files, factory_names))
    return ops


def _benchmark_label(levels):
    return f"{BENCHMARK_PREFIX}_{'_'.join(levels)}"


async def main():
    setup_logging()
    args = parse_args()

    env_spec = collect_env_spec(args.arch)
    print_env_spec(env_spec)

    ops = resolve_ops(args)
    workflow = DEFAULT_WORKFLOW
    output = args.output or default_output_path("npukernelbench_kernelgen_skill")

    await run_benchmark(
        script_name="npukernelbench_kernelgen_skill",
        workflow=workflow,
        benchmark=_benchmark_label(args.levels),
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
