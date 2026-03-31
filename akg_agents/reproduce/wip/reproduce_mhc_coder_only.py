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
  使用 coder_only_workflow（固定文档拼接）对 EvoKernel MHC benchmark 中的算子
  进行端到端代码生成，记录生成结果和性能数据。

导入方式：
  coder_only_workflow — 由 config 中 docs_dir.coder 指定的固定文档目录，
  将全部文档内容拼接注入 prompt，不经过 Skill 系统的动态选择。

前置条件：
  - source env.sh
  - API key 已配置（AKG_AGENTS_API_KEY 或 settings.json）
  - Ascend NPU 可用（DEVICE_ID 环境变量，默认 0）
  - EvoKernel benchmark 数据已就绪：
      git submodule update --init "akg_agents/thirdparty/*"

MHC 算子列表（序号 → 名称）：
  01 SinkhornKnopp       06 MhcUpdate           11 FusedMHCKernels
  02 MhcProjector         07 MHCModule            12 OptimizedMHCLayerWithFusion
  03 StreamWeightedSum    08 MHCBlock2d           13 StaticMHCHyperConnections
  04 StreamMix            09 MHCBlockBottleneck2d  14 MhcPreBlock
  05 StreamWrite          10 OrthostochasticProject 15 MhcPostBlock

运行方式：
  python reproduce/wip/reproduce_mhc_coder_only.py --help
  python reproduce/wip/reproduce_mhc_coder_only.py                # 默认全部算子
  python reproduce/wip/reproduce_mhc_coder_only.py --op 5         # 只跑序号 05
  python reproduce/wip/reproduce_mhc_coder_only.py --op 1 3 5 7   # 只跑指定序号
  python reproduce/wip/reproduce_mhc_coder_only.py --device 4 5 6 7 --concurrency 4  # 多卡池化

预期输出：
  控制台打印环境规范 + 每个算子的生成结果（pass/fail、耗时、speedup）。
  JSON 报告保存到 --output 指定路径（默认 ~/.akg/reproduce_log/）。

结果存储格式：
  {
    "script": "mhc_coder_only",
    "workflow": "coder_only_workflow",
    "ops_count": 1, "elapsed_s": 123.4,
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


def _index_to_op_name(index: int) -> str:
    """将序号转为 MHC 文件名前缀格式，如 5 -> '05_StreamWrite'"""
    ensure_test_utils_importable()
    from utils import get_evokernel_mhc_op_name
    all_ops = get_evokernel_mhc_op_name()
    if not all_ops:
        raise RuntimeError("未找到任何 MHC 算子，请检查 EvoKernel 子模块是否已初始化")
    prefix = f"{index:02d}_"
    matched = [op for op in all_ops if op.startswith(prefix)]
    if not matched:
        available = ", ".join(sorted(all_ops))
        raise RuntimeError(f"未找到序号 {index:02d} 对应的 MHC 算子。可用算子: {available}")
    return matched[0]


def parse_args():
    parser = argparse.ArgumentParser(
        description="EvoKernel MHC 复现 — 固定文档导入 (coder_only_workflow)",
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
    output = args.output or default_output_path("mhc_coder_only")

    await run_benchmark(
        script_name="mhc_coder_only",
        workflow="coder_only_workflow",
        ops=ops,
        framework="torch", dsl="triton_ascend", backend="ascend",
        arch=args.arch, device_ids=args.device,
        max_concurrency=args.concurrency,
        env_spec=env_spec, output_path=output,
    )


if __name__ == "__main__":
    asyncio.run(main())
