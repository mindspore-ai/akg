#!/usr/bin/env python3
# Copyright 2025-2026 Huawei Technologies Co., Ltd
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
基于 CUDA 参考数据缓存的自适应搜索 — Triton-CUDA → Triton-Ascend

在 Ascend NPU 环境中，利用 gen_reference_cache.py 预生成的 .pt 参考数据，
通过 adaptive_search 搜索高性能 triton_ascend 算子实现。

与 run_ascend_with_cache.py 的区别：
  - run_ascend_with_cache.py 使用 coder_only_workflow，只做正确性验证（1 次生成）
  - 本脚本使用 adaptive_search，进行多轮进化搜索并跑性能优化

工作流程：
  1. 加载 .pt 参考数据（inputs + outputs）
  2. 将 reference data 注入 config
  3. 调用 adaptive_search，内部流程：
     - LLM 生成 triton_ascend 代码
     - 验证：用 .pt 中的 inputs 跑 forward，与 .pt 中的 outputs 对比（无 CUDA 依赖）
     - Profiling：在 NPU 上实际跑性能
     - UCB 选择 → 进化下一轮
  4. 输出最佳实现及性能数据

前置条件：
  - Ascend NPU 可用
  - source env.sh
  - API key 已配置
  - 已通过 gen_reference_cache.py 生成 .pt 缓存并 scp 到本机

运行方式：
  # 单个算子
  python reproduce/wip/triton-cuda-to-ascend/run_adaptive_search_with_cache.py \\
      --source sglang --op triton_tanh

  # 批量跑全部（自动扫描 cache 目录，断点续存）
  python reproduce/wip/triton-cuda-to-ascend/run_adaptive_search_with_cache.py \\
      --source sglang

  # 批量跑所有 source（不指定 --source 也不指定 --op）
  python reproduce/wip/triton-cuda-to-ascend/run_adaptive_search_with_cache.py

  # 指定缓存文件
  python reproduce/wip/triton-cuda-to-ascend/run_adaptive_search_with_cache.py \\
      --ref-pt ~/.akg/.tmp/reference_data/triton_cuda_cache/sglang/triton_tanh.pt \\
      --benchmark-file benchmark/akg_kernels_bench/thirdparty/sglang/triton_tanh.py

  # 自定义搜索参数
  python reproduce/wip/triton-cuda-to-ascend/run_adaptive_search_with_cache.py \\
      --source sglang --op merge_state_triton \\
      --max-tasks 30 --max-concurrent 4 --devices 0 1 2 3
"""

import asyncio
import argparse
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("adaptive_search_with_cache")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_BASE = PROJECT_ROOT / "benchmark" / "akg_kernels_bench" / "thirdparty"
DEFAULT_CACHE_DIR = os.path.expanduser("~/.akg/.tmp/reference_data/triton_cuda_cache")

SOURCE_BENCHMARK_MAP = {
    "sglang": BENCHMARK_BASE / "sglang",
    "vllm_triton": BENCHMARK_BASE / "vllm" / "triton_ops",
    "vllm_torch": BENCHMARK_BASE / "vllm" / "torch_ops",
}


def resolve_paths(args):
    """解析 .pt 路径和 benchmark 文件路径"""
    if args.ref_pt and args.benchmark_file:
        return args.ref_pt, args.benchmark_file, args.op or Path(args.ref_pt).stem

    if not args.source or not args.op:
        print("错误：需要指定 --source + --op，或者 --ref-pt + --benchmark-file")
        sys.exit(1)

    cache_dir = os.path.expanduser(args.cache_dir)
    pt_path = os.path.join(cache_dir, args.source, f"{args.op}.pt")
    if not os.path.exists(pt_path):
        print(f"错误：参考数据文件不存在: {pt_path}")
        print(f"请先在 CUDA 环境运行 gen_reference_cache.py 生成缓存")
        sys.exit(1)

    benchmark_base = SOURCE_BENCHMARK_MAP.get(args.source)
    if not benchmark_base:
        print(f"错误：未知数据源: {args.source}")
        sys.exit(1)
    benchmark_file = str(benchmark_base / f"{args.op}.py")
    if not os.path.exists(benchmark_file):
        print(f"错误：Benchmark 文件不存在: {benchmark_file}")
        sys.exit(1)

    return pt_path, benchmark_file, args.op


def print_result(result):
    """打印搜索结果"""
    print(f"\n{'='*100}")
    print("自适应搜索结果")
    print(f"{'='*100}")
    print(f"算子名称：{result['op_name']}")
    print(f"终止原因：{result.get('stop_reason', 'Unknown')}")
    print(f"任务统计：提交{result['total_submitted']} / 完成{result.get('total_completed', 0)} / "
          f"成功{result['total_success']} / 失败{result['total_failed']} | "
          f"成功率{result['success_rate']:.1%} | 耗时{result['elapsed_time']:.1f}s")
    print(f"存储目录：{result.get('storage_dir', 'N/A')}")

    task_folder = result.get('task_folder', '')
    if task_folder:
        print(f"Task文件夹：{task_folder}")

    log_dir = result.get('log_dir', '')
    if log_dir:
        print(f"Log目录：{log_dir}")

    lineage_graph = result.get('lineage_graph', '')
    if lineage_graph:
        print(f"谱系图：{lineage_graph}")

    print(f"\n最佳实现（前5个）：")
    best_impls = result.get('best_implementations', [])
    if best_impls:
        for i, impl in enumerate(best_impls[:5], 1):
            task_id = impl.get('id', 'unknown')
            gen_time = impl.get('gen_time', 0)
            profile = impl.get('profile', {})
            base_time = profile.get('base_time', 0) if profile else 0
            speedup = impl.get('speedup', 0)
            generation = impl.get('generation', 0)
            parent_id = impl.get('parent_id', None)
            verify_dir = impl.get('verify_dir', '')
            if generation == 0:
                parent_desc = "初始"
            else:
                parent_desc = f"父代 {parent_id}" if parent_id else f"G{generation}"
            print(f"  {i}. {task_id}（{parent_desc}，个体路径：{verify_dir}，"
                  f"生成代码：{gen_time:.4f}us，基准代码：{base_time:.4f}us，加速比：{speedup:.2f}x）")
    else:
        print("  未找到成功的实现")
    print(f"\n{'='*100}")


async def run_single(op_name, pt_path, benchmark_file, args, worker_registered=False):
    """运行单个算子的自适应搜索

    Args:
        op_name: 算子名称
        pt_path: .pt 参考数据文件路径
        benchmark_file: benchmark 源码路径
        args: 命令行参数
        worker_registered: Worker 是否已注册（批量模式下只注册一次）

    Returns:
        dict: 搜索结果（可 JSON 序列化的摘要）
    """
    dsl = "triton_ascend"
    framework = "torch"
    backend = "ascend"
    arch = args.arch

    print(f"\n{'='*70}")
    print(f"  自适应搜索 + 参考数据模式 (Triton-CUDA → Triton-Ascend)")
    print(f"{'='*70}")
    print(f"  算子:       {op_name}")
    print(f"  参考数据:   {pt_path}")
    print(f"  Benchmark:  {benchmark_file}")
    print(f"  目标 DSL:   {dsl}")
    print(f"  架构:       {arch}")
    print(f"  设备:       {args.devices}")
    print(f"  最大任务:   {args.max_tasks}")
    print(f"  并发数:     {args.max_concurrent}")
    print(f"{'='*70}\n")

    with open(pt_path, "rb") as f:
        ref_bytes = f.read()
    print(f"  参考数据加载完成: {len(ref_bytes)} bytes")

    with open(benchmark_file, "r", encoding="utf-8") as f:
        task_desc = f.read()

    if not worker_registered:
        from akg_agents.core.worker.manager import register_worker
        await register_worker(backend=backend, arch=arch, device_ids=args.devices)

    from akg_agents.op.config.config_validator import load_config
    from akg_agents.utils.environment_check import check_env_for_task
    from akg_agents import get_project_root

    config_path = str(Path(get_project_root()) / "op" / "config" / "triton_ascend_evolve_config.yaml")
    try:
        config = load_config(config_path=config_path)
    except ValueError:
        config = load_config(dsl=dsl, backend=backend)

    config["use_reference_data"] = True
    config["use_reference_inputs"] = True
    config["reference_data"] = ref_bytes

    from akg_agents.utils.task_label import resolve_task_label
    config["task_label"] = resolve_task_label(op_name=op_name, parallel_index=1)

    check_env_for_task(framework, backend, dsl, config)

    from akg_agents.op.adaptive_search import adaptive_search

    print(f"\n  开始自适应搜索...\n")

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
        exploration_coef=args.exploration_coef,
        random_factor=args.random_factor,
        inspiration_sample_num=args.inspiration_num,
        use_tiered_sampling=True,
        handwrite_sample_num=2,
        handwrite_decay_rate=2.0,
        use_evolution_controller=True
    )

    print_result(result)
    return result


# ===================== 断点续存 =====================

DEFAULT_PROGRESS_DIR = str(Path(__file__).resolve().parent / ".tmp")


def _progress_file_path(progress_dir):
    return os.path.join(progress_dir, "batch_progress.json")


def load_progress(progress_dir):
    """加载进度 JSON，不存在则返回空结构"""
    path = _progress_file_path(progress_dir)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"cases": {}, "summary": {}}


def save_progress(progress_dir, progress):
    """原子写入进度 JSON（先写 tmp 再 rename）"""
    os.makedirs(progress_dir, exist_ok=True)
    path = _progress_file_path(progress_dir)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def _result_summary(result):
    """从 adaptive_search 结果中提取可 JSON 序列化的摘要"""
    best = result.get("best_implementations", [])
    top1 = best[0] if best else {}
    return {
        "total_submitted": result.get("total_submitted", 0),
        "total_completed": result.get("total_completed", 0),
        "total_success": result.get("total_success", 0),
        "total_failed": result.get("total_failed", 0),
        "success_rate": result.get("success_rate", 0),
        "elapsed_time": result.get("elapsed_time", 0),
        "stop_reason": result.get("stop_reason", ""),
        "storage_dir": result.get("storage_dir", ""),
        "log_dir": result.get("log_dir", ""),
        "best_speedup": top1.get("speedup", 0),
        "best_gen_time": top1.get("gen_time", 0),
    }


def discover_cases(cache_dir, sources=None):
    """扫描 cache 目录，返回 [(source, op_name, pt_path, benchmark_file), ...]"""
    cases = []
    if sources is None:
        sources = list(SOURCE_BENCHMARK_MAP.keys())
    for source in sources:
        source_cache = os.path.join(cache_dir, source)
        if not os.path.isdir(source_cache):
            continue
        benchmark_base = SOURCE_BENCHMARK_MAP.get(source)
        if not benchmark_base:
            continue
        for pt_file in sorted(Path(source_cache).glob("*.pt")):
            op_name = pt_file.stem
            benchmark_file = str(benchmark_base / f"{op_name}.py")
            if not os.path.exists(benchmark_file):
                logger.warning(f"跳过 {source}/{op_name}: benchmark 文件不存在 {benchmark_file}")
                continue
            cases.append((source, op_name, str(pt_file), benchmark_file))
    return cases


async def run_batch(args):
    """批量模式：串行跑所有 cases，断点续存"""
    cache_dir = os.path.expanduser(args.cache_dir)
    sources = [args.source] if args.source else None
    cases = discover_cases(cache_dir, sources)

    if not cases:
        print("错误：未发现任何可用的 .pt 缓存文件")
        print(f"  缓存目录: {cache_dir}")
        print(f"  数据源:   {sources or '全部'}")
        sys.exit(1)

    progress_dir = args.progress_dir
    progress = load_progress(progress_dir)

    # 统计
    total = len(cases)
    done = sum(1 for c in cases if progress["cases"].get(f"{c[0]}/{c[1]}", {}).get("status") in ("done", "error"))
    print(f"\n{'='*70}")
    print(f"  批量自适应搜索 (Triton-CUDA → Triton-Ascend)")
    print(f"{'='*70}")
    print(f"  总 cases:    {total}")
    print(f"  已完成:      {done}")
    print(f"  待执行:      {total - done}")
    print(f"  进度文件:    {_progress_file_path(progress_dir)}")
    print(f"{'='*70}\n")

    # 注册 Worker（只注册一次）
    from akg_agents.core.worker.manager import register_worker
    await register_worker(backend="ascend", arch=args.arch, device_ids=args.devices)

    succeeded, failed, skipped = 0, 0, 0

    for idx, (source, op_name, pt_path, benchmark_file) in enumerate(cases, 1):
        case_key = f"{source}/{op_name}"
        case_record = progress["cases"].get(case_key, {})

        if case_record.get("status") in ("done", "error"):
            skipped += 1
            print(f"[{idx}/{total}] 跳过（{case_record['status']}）: {case_key}")
            continue

        print(f"\n[{idx}/{total}] 开始: {case_key}")
        start_ts = time.time()

        try:
            result = await run_single(
                op_name=op_name,
                pt_path=pt_path,
                benchmark_file=benchmark_file,
                args=args,
                worker_registered=True
            )
            elapsed = time.time() - start_ts
            progress["cases"][case_key] = {
                "status": "done",
                "source": source,
                "op_name": op_name,
                "finished_at": datetime.now().isoformat(),
                "elapsed_time": round(elapsed, 1),
                "result": _result_summary(result),
            }
            save_progress(progress_dir, progress)
            succeeded += 1
            best_sp = progress["cases"][case_key]["result"]["best_speedup"]
            print(f"[{idx}/{total}] 完成: {case_key}  "
                  f"best_speedup={best_sp:.2f}x  耗时={elapsed:.0f}s")

        except Exception as e:
            elapsed = time.time() - start_ts
            err_msg = traceback.format_exc()
            logger.error(f"[{idx}/{total}] {case_key} 失败: {e}")
            progress["cases"][case_key] = {
                "status": "error",
                "source": source,
                "op_name": op_name,
                "finished_at": datetime.now().isoformat(),
                "elapsed_time": round(elapsed, 1),
                "error": str(e),
                "traceback": err_msg,
            }
            save_progress(progress_dir, progress)
            failed += 1
            print(f"[{idx}/{total}] 失败: {case_key}  error={e}")

    # 更新汇总
    progress["summary"] = {
        "total": total,
        "succeeded": succeeded,
        "failed": failed,
        "skipped": skipped,
        "finished_at": datetime.now().isoformat(),
    }
    save_progress(progress_dir, progress)

    print(f"\n{'='*70}")
    print(f"  批量搜索完成")
    print(f"{'='*70}")
    print(f"  总计: {total}  成功: {succeeded}  失败: {failed}  跳过(已完成): {skipped}")
    print(f"  进度文件: {_progress_file_path(progress_dir)}")
    print(f"{'='*70}\n")


async def run(args):
    """入口：根据参数决定走单个还是批量"""
    # 指定了具体文件 → 单个
    if args.ref_pt and args.benchmark_file:
        pt_path, benchmark_file, op_name = resolve_paths(args)
        return await run_single(op_name, pt_path, benchmark_file, args)

    # 指定了 --op → 单个
    if args.op:
        pt_path, benchmark_file, op_name = resolve_paths(args)
        return await run_single(op_name, pt_path, benchmark_file, args)

    # 否则 → 批量模式
    return await run_batch(args)


def parse_args():
    parser = argparse.ArgumentParser(
        description="基于 CUDA 参考数据缓存的自适应搜索 (Triton-CUDA → Triton-Ascend)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 单个算子
  python run_adaptive_search_with_cache.py --source sglang --op triton_tanh

  # 批量跑某个 source 下所有 cache（断点续存，重跑自动跳过已完成）
  python run_adaptive_search_with_cache.py --source sglang

  # 批量跑全部 source
  python run_adaptive_search_with_cache.py

  # 重置某个失败的 case 后重跑（手动编辑进度 JSON 删掉该条目即可）

  # 手动指定文件路径
  python run_adaptive_search_with_cache.py \\
      --ref-pt /path/to/triton_tanh.pt \\
      --benchmark-file /path/to/triton_tanh.py

  # 多设备并行 + 更多搜索
  python run_adaptive_search_with_cache.py \\
      --source sglang --op merge_state_triton \\
      --devices 0 1 2 3 --max-concurrent 4 --max-tasks 40
        """,
    )

    # 算子选择（方式一：source + op）
    parser.add_argument("--source", choices=list(SOURCE_BENCHMARK_MAP.keys()),
                        help="数据源（不指定 --op 时批量跑该 source 下所有 cache）")
    parser.add_argument("--op", help="算子名称（不指定则进入批量模式）")

    # 算子选择（方式二：直接指定文件路径）
    parser.add_argument("--ref-pt", help=".pt 参考数据文件路径")
    parser.add_argument("--benchmark-file", help="Benchmark 源代码文件路径")

    # 缓存目录
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR,
                        help=f"参考数据缓存目录（默认 {DEFAULT_CACHE_DIR}）")

    # 断点续存
    parser.add_argument("--progress-dir", default=DEFAULT_PROGRESS_DIR,
                        help="批量模式进度文件目录（默认脚本同级 .tmp/）")

    # 硬件配置
    parser.add_argument("--devices", type=int, nargs="+", default=[0, 1, 2, 3],
                        help="NPU 设备 ID 列表（默认 [0,1,2,3]）")
    parser.add_argument("--arch", default="ascend910b4",
                        help="硬件架构（默认 ascend910b4）")

    # 搜索参数
    parser.add_argument("--max-concurrent", type=int, default=4,
                        help="最大并发任务数（默认 4）")
    parser.add_argument("--initial-tasks", type=int, default=4,
                        help="初始任务数量（默认 4）")
    parser.add_argument("--max-tasks", type=int, default=20,
                        help="最大总任务数（默认 20）")
    parser.add_argument("--exploration-coef", type=float, default=1.414,
                        help="UCB 探索系数（默认 1.414）")
    parser.add_argument("--random-factor", type=float, default=0.1,
                        help="选择时的随机扰动（默认 0.1）")
    parser.add_argument("--inspiration-num", type=int, default=3,
                        help="灵感采样数量（默认 3）")

    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
