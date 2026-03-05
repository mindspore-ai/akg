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

"""直接运行 akg_agents op workflow。

三种 workflow：
  adaptive_search  自适应搜索（UCB 策略，推荐）
  kernelgen        KernelGen → Verifier → Conductor 迭代
  evolve           进化搜索（岛屿模型）

用法:
  python run_workflow.py --workflow adaptive_search \
    --task-file /path/to/task_desc.py \
    --framework torch --backend cuda --arch a100 --dsl triton_cuda

  python run_workflow.py --workflow kernelgen \
    --task-file /path/to/task_desc.py \
    --framework torch --backend ascend --arch ascend910b4 --dsl triton_ascend \
    --output-path ~/output

  python run_workflow.py --workflow evolve \
    --task-file /path/to/task_desc.py \
    --framework torch --backend cuda --arch a100 --dsl triton_cuda
"""

import asyncio
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


async def run_adaptive_search(op_name, task_desc, framework, backend, arch, dsl, config, output_path, args):
    from akg_agents.op.adaptive_search import adaptive_search
    from akg_agents.core.worker.manager import register_local_worker

    devices = [int(d) for d in args.devices.split(",")] if args.devices else [0]
    await register_local_worker(devices, backend=backend, arch=arch)

    result = await adaptive_search(
        op_name=op_name,
        task_desc=task_desc,
        dsl=dsl,
        framework=framework,
        backend=backend,
        arch=arch,
        config=config,
        max_concurrent=args.max_concurrent,
        initial_task_count=args.initial_tasks,
        max_total_tasks=args.max_tasks,
    )

    best_code = ""
    best_profile = {}
    if result.get("best_implementations"):
        best = result["best_implementations"][0]
        best_code = best.get("impl_code", "")
        best_profile = best.get("profile", {})

    success = result.get("total_success", 0) > 0
    return {
        "code": best_code,
        "profile": best_profile,
        "success": success,
        "total_submitted": result.get("total_submitted", 0),
        "total_success": result.get("total_success", 0),
        "elapsed_time": result.get("elapsed_time", 0),
        "storage_dir": result.get("storage_dir", ""),
    }


async def run_evolve(op_name, task_desc, framework, backend, arch, dsl, config, output_path, args):
    from akg_agents.op.evolve import evolve
    from akg_agents.core.async_pool.task_pool import TaskPool
    from akg_agents.core.worker.manager import register_local_worker

    devices = [int(d) for d in args.devices.split(",")] if args.devices else [0]
    await register_local_worker(devices, backend=backend, arch=arch)

    parallel_num = args.parallel_num
    task_pool = TaskPool(max_concurrency=parallel_num)

    result = await evolve(
        op_name=op_name,
        task_desc=task_desc,
        dsl=dsl,
        framework=framework,
        backend=backend,
        arch=arch,
        config=config,
        task_pool=task_pool,
        max_rounds=args.max_rounds,
        parallel_num=parallel_num,
        num_islands=args.num_islands,
        migration_interval=args.migration_interval,
        elite_size=args.elite_size,
        parent_selection_prob=args.parent_selection_prob,
    )

    best_code = ""
    best_profile = {}
    if result.get("best_implementations"):
        best = result["best_implementations"][0]
        best_code = best.get("impl_code", "")
        best_profile = best.get("profile", {})

    success = result.get("successful_tasks", 0) > 0
    return {
        "code": best_code,
        "profile": best_profile,
        "success": success,
        "total_tasks": result.get("total_tasks", 0),
        "successful_tasks": result.get("successful_tasks", 0),
    }


async def run_kernelgen(op_name, task_desc, framework, backend, arch, dsl, config, output_path, args):
    from akg_agents.op.langgraph_op.task import LangGraphTask
    from akg_agents.utils.task_label import resolve_task_label
    from akg_agents.core.worker.manager import register_local_worker

    devices = [int(d) for d in args.devices.split(",")] if args.devices else [0]
    await register_local_worker(devices, backend=backend, arch=arch)

    config["task_label"] = resolve_task_label(op_name=op_name, parallel_index=1)

    task = LangGraphTask(
        op_name=op_name,
        task_desc=task_desc,
        task_id="0",
        backend=backend,
        arch=arch,
        dsl=dsl,
        config=config,
        framework=framework,
        workflow="kernelgen_only",
    )

    result_op_name, success, final_state = await task.run()

    code = final_state.get("coder_code", "")
    profile_res = final_state.get("profile_res", {})

    return {
        "code": code,
        "profile": profile_res,
        "success": success,
        "error": final_state.get("verifier_error", ""),
    }


WORKFLOW_RUNNERS = {
    "adaptive_search": run_adaptive_search,
    "evolve": run_evolve,
    "kernelgen": run_kernelgen,
}


async def main(args):
    task_file_path = os.path.abspath(os.path.expanduser(args.task_file))
    if not os.path.isfile(task_file_path):
        logger.error(f"task-file 不存在: {task_file_path}")
        return False

    with open(task_file_path, "r", encoding="utf-8") as f:
        task_desc = f.read()

    output_path = args.output_path or os.path.join(
        os.path.expanduser("~/akg_agents_logs"),
        f"workflow_{args.workflow}_{int(time.time())}",
    )
    output_path = os.path.abspath(os.path.expanduser(output_path))
    os.makedirs(output_path, exist_ok=True)

    op_name = args.op_name or Path(task_file_path).stem

    # Build config
    from akg_agents.op.workflows.base_workflow import OpBaseWorkflow
    from akg_agents.op.config.config_validator import load_config

    try:
        config = load_config(dsl=args.dsl, backend=args.backend, workflow=args.workflow)
    except ValueError:
        config = {}

    full_config = OpBaseWorkflow.build_langgraph_task_config(
        dsl=args.dsl,
        backend=args.backend,
        op_name=op_name,
        base_config=config,
    )
    full_config["log_dir"] = str(Path(output_path) / "logs")
    full_config["skip_kernel_gen"] = args.workflow != "kernelgen"

    logger.info(f"Workflow: {args.workflow}")
    logger.info(f"Task: {task_file_path} (op_name={op_name})")
    logger.info(f"Config: framework={args.framework}, backend={args.backend}, arch={args.arch}, dsl={args.dsl}")
    logger.info(f"Output: {output_path}")

    runner = WORKFLOW_RUNNERS[args.workflow]
    result = await runner(
        op_name=op_name,
        task_desc=task_desc,
        framework=args.framework,
        backend=args.backend,
        arch=args.arch,
        dsl=args.dsl,
        config=full_config,
        output_path=output_path,
        args=args,
    )

    # Save results
    code = result.get("code", "")
    if code:
        code_file = os.path.join(output_path, "generated_code.py")
        with open(code_file, "w", encoding="utf-8") as f:
            f.write(code)
        logger.info(f"生成代码: {code_file}")

    summary = {
        "workflow": args.workflow,
        "success": result.get("success", False),
        "op_name": op_name,
        "task_file": task_file_path,
        "output_path": output_path,
        "framework": args.framework,
        "backend": args.backend,
        "arch": args.arch,
        "dsl": args.dsl,
        "profile": str(result.get("profile", {})),
    }
    summary_file = os.path.join(output_path, "summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"摘要: {summary_file}")

    success = result.get("success", False)
    logger.info(f"结果: {'成功' if success else '失败'}")
    return success


def parse_args():
    parser = argparse.ArgumentParser(
        description="直接运行 akg_agents op workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required
    parser.add_argument("--workflow", required=True, choices=list(WORKFLOW_RUNNERS.keys()))
    parser.add_argument("--task-file", required=True, help="KernelBench 格式的 {op_name}.py")
    parser.add_argument("--framework", required=True, help="框架（如 torch）")
    parser.add_argument("--backend", required=True, help="后端（如 cuda, ascend, cpu）")
    parser.add_argument("--arch", required=True, help="架构（如 a100, ascend910b4）")
    parser.add_argument("--dsl", required=True, help="DSL（如 triton_cuda, triton_ascend, cpp）")

    # Optional
    parser.add_argument("--op-name", default=None, help="算子名称（默认从文件名推断）")
    parser.add_argument("--output-path", default=None, help="输出目录")
    parser.add_argument("--devices", default="0", help="设备 ID 列表，逗号分隔（默认 0）")

    # adaptive_search params
    parser.add_argument("--max-concurrent", type=int, default=2)
    parser.add_argument("--initial-tasks", type=int, default=2)
    parser.add_argument("--max-tasks", type=int, default=10)

    # evolve params
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--parallel-num", type=int, default=4)
    parser.add_argument("--num-islands", type=int, default=2)
    parser.add_argument("--migration-interval", type=int, default=2)
    parser.add_argument("--elite-size", type=int, default=5)
    parser.add_argument("--parent-selection-prob", type=float, default=0.5)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    success = asyncio.run(main(args))
    sys.exit(0 if success else 1)
