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

通用 CLI 参数（由 add_common_args 注册，所有基础脚本共享）：
  --device ID [ID ...]   NPU 设备 ID，可多个以池化（默认 $DEVICE_ID 或 0）
  --concurrency N        设备并行度上限（默认 4）
  --llm-concurrency N    LLM 请求并发数（默认与 --concurrency 相同）
  --arch ARCH            硬件架构（默认 ascend910b4）
  --pass-n N             Pass@N：每个算子独立运行 N 次（默认 1）
  --output PATH          JSON 报告输出路径（默认 ~/.akg/reproduce_log/<script>_<timestamp>.json）
  --profile              开启性能测试（默认关闭；开启后验证通过的算子自动跑 speedup）
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
from typing import Dict, List, Optional, Any

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

    try:
        from akg_agents.core_v2.llm.factory import create_llm_client
        client = create_llm_client(model_level="standard")
        llm_model = client.provider.model_name
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
# 性能数据提取
# ============================================================

def _extract_perf_data(results: list) -> Dict[str, dict]:
    """从 task 结果中提取每个算子的最优性能数据。

    对于 pass@k 场景（同一算子多次尝试），保留 speedup 最高的一组。
    """
    perf = {}
    for op_name, _success, final_state in results:
        if not isinstance(final_state, dict):
            continue
        profile = final_state.get("profile_res")
        if not profile or not isinstance(profile, dict):
            continue
        existing = perf.get(op_name)
        if existing is None or profile.get("speedup", 0) > existing.get("speedup", 0):
            perf[op_name] = {
                k: v for k, v in profile.items()
                if k in ("gen_time", "base_time", "speedup")
            }
    return perf


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
    benchmark: str = "",
    pass_n: int = 1,
    llm_concurrency: Optional[int] = None,
    enable_profile: bool = False,
):
    """
    通用 benchmark 运行器（仅用于基础 workflow: coder_only / kernelgen）。

    adaptive_search / evolve 请使用 reproduce_adaptive_search.py / reproduce_evolve.py。
    """
    from akg_agents.core.async_pool.task_pool import TaskPool
    from akg_agents.op.langgraph_op.task import LangGraphTask as AIKGTask
    from akg_agents.core.worker.manager import register_local_worker
    from akg_agents.op.config.config_validator import load_config
    from akg_agents.utils.environment_check import check_env_for_task

    ensure_test_utils_importable()
    from utils import generate_beautiful_test_report

    llm_concurrency = llm_concurrency or max_concurrency
    task_type = "profile" if enable_profile else "precision_only"

    if workflow == "kernelgen_only_workflow":
        config = load_config(config_path=KERNELGEN_CONFIG_PATH)
    else:
        config = load_config(dsl=dsl, backend=backend)

    check_env_for_task(framework, backend, dsl, config)
    await register_local_worker(device_ids, backend=backend, arch=arch)

    os.environ['AKG_AGENTS_STREAM_OUTPUT'] = 'on'

    log_dir = config.get("log_dir", os.environ.get("AKG_AGENTS_LOG_DIR", "~/akg_agents_logs").strip() or "~/akg_agents_logs")
    log_dir_expanded = os.path.expanduser(log_dir)

    total_tasks = len(ops) * pass_n
    print(f"  benchmark: {benchmark}")
    print(f"  算子数量:  {len(ops)}  (pass@{pass_n}, 总任务 {total_tasks})")
    print(f"  workflow:  {workflow}")
    print(f"  task_type: {task_type}")
    print(f"  devices:   {device_ids}")
    print(f"  设备并发:  {max_concurrency}")
    print(f"  LLM 并发:  {llm_concurrency}")
    print(f"  任务日志:  {log_dir_expanded}\n")

    task_pool = TaskPool(max_concurrency=max_concurrency)
    t0 = time.time()

    for i, (op_display, task_desc) in enumerate(ops):
        for k in range(pass_n):
            task_id = f"{i}" if pass_n == 1 else f"{i}_k{k}"
            task = AIKGTask(
                op_name=op_display,
                task_desc=task_desc,
                task_id=task_id,
                backend=backend, arch=arch, dsl=dsl,
                config=config, framework=framework,
                workflow=workflow,
                task_type=task_type,
            )
            task_pool.create_task(task.run)

    results = await task_pool.wait_all()
    elapsed = time.time() - t0

    perf_data = _extract_perf_data(results)

    report_stats = generate_beautiful_test_report(
        results, config, framework, dsl, backend, arch,
    )

    op_results = {}
    for op_name, stat in report_stats.get("op_stats", {}).items():
        entry = {"passed": stat["passed"], "total": stat["total"]}
        if op_name in perf_data:
            entry["profile"] = perf_data[op_name]
        op_results[op_name] = entry

    summary = {
        "benchmark": benchmark,
        "script": script_name,
        "workflow": workflow,
        "pass_n": pass_n,
        "ops_count": len(ops),
        "elapsed_s": round(elapsed, 1),
        "device_ids": device_ids,
        "max_concurrency": max_concurrency,
        "llm_concurrency": llm_concurrency,
        "env_spec": env_spec,
        "task_log_dir": log_dir_expanded,
        "stats": {
            "total_ops": report_stats["total_ops"],
            "passed_ops": report_stats["passed_ops"],
            "failed_ops": report_stats["failed_ops"],
            "pass_rate": report_stats["pass_rate"],
            "op_results": op_results,
        },
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 70}")
    print(f"  完成: {script_name}")
    print(f"  算子数: {len(ops)}  |  pass@{pass_n}  |  总耗时: {elapsed:.1f}s")
    print(f"  通过率: {report_stats['passed_ops']}/{report_stats['total_ops']}"
          f" ({report_stats['pass_rate']:.1%})")
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
        help=f"设备并行度上限（默认 {DEFAULT_MAX_CONCURRENCY}）",
    )
    parser.add_argument(
        "--llm-concurrency", type=int, default=None,
        help="LLM 请求并发数（默认与 --concurrency 相同）",
    )
    parser.add_argument(
        "--arch", default="ascend910b4",
        help="硬件架构（默认 ascend910b4）",
    )
    parser.add_argument(
        "--pass-n", type=int, default=1,
        help="Pass@N：每个算子独立运行 N 次（默认 1）",
    )
    parser.add_argument(
        "--output", default=None,
        help="JSON 报告输出路径（默认 ~/.akg/reproduce_log/<script>_<timestamp>.json）",
    )
    parser.add_argument(
        "--profile", action="store_true",
        help="开启性能测试（验证通过后自动跑 speedup / gen_time / base_time）",
    )
    return parser


def default_output_path(script_name: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(DEFAULT_LOG_DIR, f"{script_name}_{ts}.json")
