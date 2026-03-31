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
wip 复现脚本公共模块

提供环境规范采集、报告生成、结果存储等共享功能。
所有 reproduce/wip/ 下的脚本通过 from _common import ... 使用。
"""

import json
import logging
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List

logger = logging.getLogger("reproduce")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TESTS_OP_DIR = PROJECT_ROOT / "tests" / "op"

KERNELGEN_CONFIG_PATH = "./python/akg_agents/op/config/triton_ascend_kernelgen_config.yaml"
DEFAULT_LOG_DIR = os.path.expanduser("~/.akg/reproduce_log")
DEFAULT_MAX_CONCURRENCY = 4


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def ensure_test_utils_importable():
    """将 tests/op 加入 sys.path，使 utils 可导入"""
    p = str(TESTS_OP_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)


# ============================================================
# 环境规范
# ============================================================

def collect_env_spec(arch: str) -> dict:
    """采集当前运行环境信息，写入报告头部"""
    spec = {
        "arch": arch,
        "python": platform.python_version(),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    try:
        import torch_npu
        spec["torch_npu"] = torch_npu.__version__
    except ImportError:
        spec["torch_npu"] = "N/A"

    try:
        import triton
        spec["triton_ascend"] = getattr(triton, "__version__", "unknown")
    except ImportError:
        spec["triton_ascend"] = "N/A"

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=str(PROJECT_ROOT.parent),
        )
        spec["commit"] = result.stdout.strip() if result.returncode == 0 else "unknown"
    except FileNotFoundError:
        spec["commit"] = "unknown"

    llm_model = os.environ.get("AIKG_MODEL_NAME", "")
    if not llm_model:
        try:
            from akg_agents.core_v2.llm.factory import create_llm_client
            client = create_llm_client()
            llm_model = getattr(client, "model_name", "unknown")
        except Exception:
            llm_model = "unknown"
    spec["llm_model"] = llm_model

    return spec


def print_env_spec(spec: dict):
    print("\n" + "=" * 70)
    print("  环境规范")
    print("=" * 70)
    for k, v in spec.items():
        print(f"  {k:20s}: {v}")
    print("=" * 70 + "\n")


# ============================================================
# 通用运行器
# ============================================================

async def run_benchmark(
    *,
    script_name: str,
    workflow: str,
    ops: list,
    framework: str,
    dsl: str,
    backend: str,
    arch: str,
    device_ids: List[int],
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    env_spec: dict,
    output_path: str,
):
    """
    通用 benchmark 运行器。

    Args:
        script_name: 脚本标识名
        workflow: "coder_only_workflow" 或 "kernelgen_only_workflow"
        ops: [(display_name, task_desc), ...] 算子列表
        framework / dsl / backend / arch: 硬件环境参数
        device_ids: NPU 设备 ID 列表，多设备时自动池化
        max_concurrency: 任务并行度上限
        env_spec: collect_env_spec() 返回的环境信息
        output_path: JSON 报告输出路径

    Returns:
        dict: 结构化结果
    """
    from akg_agents.core.async_pool.task_pool import TaskPool
    from akg_agents.op.langgraph_op.task import LangGraphTask as AIKGTask
    from akg_agents.core.worker.manager import register_local_worker
    from akg_agents.op.config.config_validator import load_config
    from akg_agents.utils.environment_check import check_env_for_task

    ensure_test_utils_importable()
    from utils import generate_beautiful_test_report

    if workflow == "kernelgen_only_workflow":
        config = load_config(config_path=KERNELGEN_CONFIG_PATH)
    else:
        config = load_config(dsl=dsl, backend=backend)

    check_env_for_task(framework, backend, dsl, config)
    await register_local_worker(device_ids, backend=backend, arch=arch)

    os.environ['AKG_AGENTS_DATA_COLLECT'] = 'on'
    os.environ['AKG_AGENTS_STREAM_OUTPUT'] = 'on'

    log_dir = config.get("log_dir", "~/akg_agents_logs")
    log_dir_expanded = os.path.expanduser(log_dir)

    print(f"  算子数量:  {len(ops)}")
    print(f"  workflow:  {workflow}")
    print(f"  devices:   {device_ids}")
    print(f"  并行度:    {max_concurrency}")
    print(f"  任务日志:  {log_dir_expanded}\n")

    task_pool = TaskPool(max_concurrency=max_concurrency)
    t0 = time.time()

    for i, (op_display, task_desc) in enumerate(ops):
        task = AIKGTask(
            op_name=op_display,
            task_desc=task_desc,
            task_id=str(i),
            backend=backend, arch=arch, dsl=dsl,
            config=config, framework=framework,
            workflow=workflow,
        )
        task_pool.create_task(task.run)

    results = await task_pool.wait_all()
    elapsed = time.time() - t0

    report_stats = generate_beautiful_test_report(
        results, config, framework, dsl, backend, arch,
    )

    summary = {
        "script": script_name,
        "workflow": workflow,
        "ops_count": len(ops),
        "elapsed_s": round(elapsed, 1),
        "device_ids": device_ids,
        "max_concurrency": max_concurrency,
        "env_spec": env_spec,
        "task_log_dir": log_dir_expanded,
        "stats": report_stats,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 70}")
    print(f"  完成: {script_name}")
    print(f"  算子数: {len(ops)}  |  总耗时: {elapsed:.1f}s")
    print(f"  任务日志: {log_dir_expanded}")
    print(f"  报告: {output_path}")
    print(f"{'=' * 70}\n")

    return summary


# ============================================================
# 通用 argparse 构建
# ============================================================

def add_common_args(parser):
    """为 argparse 添加通用参数"""
    parser.add_argument(
        "--device", nargs="+", type=int,
        default=[int(os.getenv("DEVICE_ID", "0"))],
        help="NPU 设备 ID（可指定多个以池化，如 --device 4 5 6 7；默认 $DEVICE_ID 或 0）",
    )
    parser.add_argument(
        "--concurrency", type=int, default=DEFAULT_MAX_CONCURRENCY,
        help=f"任务并行度上限（默认 {DEFAULT_MAX_CONCURRENCY}）",
    )
    parser.add_argument(
        "--arch", default="ascend910b4",
        help="硬件架构（默认 ascend910b4）",
    )
    parser.add_argument(
        "--output", default=None,
        help="JSON 报告输出路径（默认 ~/.akg/reproduce_log/<script>_<timestamp>.json）",
    )
    return parser


def default_output_path(script_name: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(DEFAULT_LOG_DIR, f"{script_name}_{ts}.json")
