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
EvoKernel MHC 算子生成复现 — 固定文档导入 (coder_only_workflow)

复现目标：
  使用 coder_only_workflow（固定文档拼接）对 EvoKernel MHC 中的算子进行端到端
  代码生成。适合作为 kernelgen_only_workflow 的基线对照。

导入方式：
  coder_only_workflow — 将固定参考文档直接拼接到 prompt，不经过 Skill 系统。

默认行为：
  运行 MHC 全部算子。可通过 --op 指定序号运行子集。

前置条件：
  - source env.sh
  - API key 已配置（AKG_AGENTS_API_KEY 或 settings.json）
  - Ascend NPU 可用（DEVICE_ID 环境变量，默认 0）
  - EvoKernel benchmark 数据已就绪：
      bash akg_agents/download.sh --with_evokernel

MHC 算子列表（序号 → 名称）：
  01 SinkhornKnopp       06 MhcUpdate           11 FusedMHCKernels
  02 MhcProjector         07 MHCModule            12 OptimizedMHCLayerWithFusion
  03 StreamWeightedSum    08 MHCBlock2d           13 StaticMHCHyperConnections
  04 StreamMix            09 MHCBlockBottleneck2d  14 MhcPreBlock
  05 StreamWrite          10 OrthostochasticProject 15 MhcPostBlock

运行方式：
  # 默认运行全部 MHC 算子
  python reproduce/wip/reproduce_mhc_coder_only.py

  # 运行单个算子
  python reproduce/wip/reproduce_mhc_coder_only.py --op 5

  # 运行多个算子
  python reproduce/wip/reproduce_mhc_coder_only.py --op 1 3 5 7

  # Pass@3
  python reproduce/wip/reproduce_mhc_coder_only.py --pass-n 3

  # 多设备并行
  python reproduce/wip/reproduce_mhc_coder_only.py --device 4 5 6 7 --concurrency 4 --llm-concurrency 8

可调参数：
  --op N [N ...]         MHC 算子序号（默认全部；指定后只跑对应序号）
  --device ID [ID ...]   NPU 设备 ID，可多个以池化（默认 $DEVICE_ID 或 0）
  --concurrency N        设备并行度上限（默认 4）
  --llm-concurrency N    LLM 请求并发数（默认与 --concurrency 相同）
  --arch ARCH            硬件架构（默认 ascend910b4）
  --pass-n N             Pass@N：每个算子独立运行 N 次（默认 1）
  --output PATH          JSON 报告输出路径
  --profile              开启性能测试（默认关闭；开启后验证通过的算子自动跑 speedup）

输出格式：
  JSON 文件（默认 ~/.akg/reproduce_log/mhc_coder_only_<timestamp>.json），
  包含 benchmark="EvoKernel_MHC"、stats.op_results（含 profile）等字段。
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

BENCHMARK = "EvoKernel_MHC"
DEFAULT_WORKFLOW = "coder_only_workflow"


def _index_to_op_name(index: int) -> str:
    ensure_test_utils_importable()
    from utils import get_evokernel_mhc_op_name
    all_ops = get_evokernel_mhc_op_name()
    if not all_ops:
        raise RuntimeError("未找到任何 MHC 算子，请检查是否已执行 `bash akg_agents/download.sh --with_evokernel`")
    prefix = f"{index:02d}_"
    matched = [op for op in all_ops if op.startswith(prefix)]
    if not matched:
        available = ", ".join(sorted(all_ops))
        raise RuntimeError(f"未找到序号 {index:02d} 对应的 MHC 算子。可用算子: {available}")
    return matched[0]


def parse_args():
    parser = argparse.ArgumentParser(
        description="EvoKernel MHC 复现 — 固定文档导入",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--op", nargs="+", type=int, default=None,
                        help="MHC 算子序号（默认全部；指定后只跑对应序号）")
    add_common_args(parser)
    return parser.parse_args()


def resolve_ops(args):
    ensure_test_utils_importable()
    from utils import get_evokernel_mhc_op_name, get_evokernel_task_desc, add_op_prefix

    if args.op is None:
        names = get_evokernel_mhc_op_name()
    else:
        names = [_index_to_op_name(idx) for idx in args.op]
    if not names:
        raise RuntimeError(f"未找到 MHC 算子（op={args.op}）")

    ops = []
    for n in names:
        td = get_evokernel_task_desc(n, category="MHC")
        ops.append((add_op_prefix(f"MHC_{n}", benchmark="EvoKernel"), td))
    return ops


async def main():
    setup_logging()
    args = parse_args()

    env_spec = collect_env_spec(args.arch)
    print_env_spec(env_spec)

    ops = resolve_ops(args)
    workflow = DEFAULT_WORKFLOW
    output = args.output or default_output_path("mhc_coder_only")

    await run_benchmark(
        script_name="mhc_coder_only",
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
