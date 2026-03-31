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
KernelBench Level1 算子生成复现 — 固定文档导入 (coder_only_workflow)

复现目标：
  使用 coder_only_workflow（固定文档拼接）对 KernelBench Level1 中指定序号的
  算子进行端到端代码生成，记录生成结果和性能数据。

导入方式：
  coder_only_workflow — 由 config 中 docs_dir.coder 指定的固定文档目录，
  将全部文档内容拼接注入 prompt，不经过 Skill 系统的动态选择。

默认行为：
  运行 Level1 全部 100 个算子，但排除 54-87 号 conv 算子（triton_ascend
  不支持卷积算子生成）。可通过 --include-conv 包含这些算子。

前置条件：
  - source env.sh
  - API key 已配置（AKG_AGENTS_API_KEY 或 settings.json）
  - Ascend NPU 可用（DEVICE_ID 环境变量，默认 0）
  - KernelBench 子模块已初始化：
      git submodule update --init "akg_agents/thirdparty/*"

运行方式：
  python reproduce/wip/reproduce_kernelbench_coder_only.py --help
  python reproduce/wip/reproduce_kernelbench_coder_only.py                    # 默认全部（排除 conv）
  python reproduce/wip/reproduce_kernelbench_coder_only.py --tasks 1 5 19 42  # 指定序号
  python reproduce/wip/reproduce_kernelbench_coder_only.py --include-conv     # 包含 conv 算子
  python reproduce/wip/reproduce_kernelbench_coder_only.py --device 4 --arch ascend910b3

预期输出：
  控制台打印环境规范 + 每个算子的生成结果（pass/fail、耗时、speedup）。
  JSON 报告保存到 --output 指定路径（默认 ~/.akg/reproduce_log/）。

结果存储格式：
  {
    "script": "kernelbench_coder_only",
    "workflow": "coder_only_workflow",
    "ops_count": 66, "elapsed_s": 1234.5,
    "env_spec": { "arch", "torch_npu", "triton_ascend", "commit", "llm_model", ... },
    "task_log_dir": "~/akg_agents_logs",
    "stats": { ... }
  }
"""

import argparse
import asyncio

from _common import (
    setup_logging, collect_env_spec, print_env_spec,
    run_benchmark, add_common_args, default_output_path,
    ensure_test_utils_importable,
)

CONV_RANGE = set(range(54, 88))


def _default_task_indices(include_conv: bool) -> list:
    """Level1 全部序号（1-100），默认排除 54-87 conv 算子"""
    all_indices = list(range(1, 101))
    if include_conv:
        return all_indices
    return [i for i in all_indices if i not in CONV_RANGE]


def parse_args():
    parser = argparse.ArgumentParser(
        description="KernelBench Level1 复现 — 固定文档导入 (coder_only_workflow)",
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
    return ops


async def main():
    setup_logging()
    args = parse_args()

    env_spec = collect_env_spec(args.arch)
    print_env_spec(env_spec)

    ops = resolve_ops(args)
    output = args.output or default_output_path("kernelbench_coder_only")

    await run_benchmark(
        script_name="kernelbench_coder_only",
        workflow="coder_only_workflow",
        ops=ops,
        framework="torch", dsl="triton_ascend", backend="ascend",
        arch=args.arch, device_ids=args.device,
        max_concurrency=args.concurrency,
        env_spec=env_spec, output_path=output,
    )


if __name__ == "__main__":
    asyncio.run(main())
