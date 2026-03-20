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
verify_evolved_skill.py - 验证特定 evolved skill 对算子生成的效果

针对指定算子任务，使用 kernelgen_only workflow 对比：
  A 组 (baseline): 仅使用 guides/ 下的原始 skill（KernelGen 默认行为）
  B 组 (treatment): 在原始 skill 基础上，强制注入用户指定的 skill

--skill-paths 支持三种形式（可混合使用）：
  - 目录路径: 递归加载目录下所有 SKILL.md
  - 文件路径: 直接加载单个 SKILL.md
  - 多个路径: 空格分隔，任意混合目录和文件

用法:
  # 指定单个 skill 文件
  python verify_evolved_skill.py --task-file /path/to/op.py --skill-paths /path/to/error-fix/SKILL.md

  # 指定 skill 目录（递归加载）
  python verify_evolved_skill.py --task-file /path/to/op.py --skill-paths /path/to/evolved/

  # 混合多个路径
  python verify_evolved_skill.py --task-file /path/to/op.py --skill-paths /path/to/error-fix/SKILL.md /path/to/exp-skill/

  # 指定其他 DSL/backend
  python verify_evolved_skill.py --task-file /path/to/op.py --skill-paths /path/to/SKILL.md --dsl triton_cuda --backend cuda

task-file 格式：标准 bench_lite 算子文件（包含 class Model, get_inputs, get_init_inputs）。
"""

import argparse
import asyncio
import importlib.util
import json
import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def get_project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "python" / "akg_agents").is_dir():
            return parent
    raise FileNotFoundError("无法定位项目根目录")


PROJECT_ROOT = get_project_root()
sys.path.insert(0, str(PROJECT_ROOT / "python"))

SKILLS_DIR = PROJECT_ROOT / "python" / "akg_agents" / "op" / "resources" / "skills"


def load_task_file(task_file: str) -> tuple:
    """从算子文件中提取 op_name 和 task_desc。"""
    path = Path(task_file).resolve()
    if not path.exists():
        raise FileNotFoundError(f"算子文件不存在: {path}")

    spec = importlib.util.spec_from_file_location("_task_module", str(path))
    mod = importlib.util.module_from_spec(spec)

    task_desc = path.read_text(encoding="utf-8")
    op_name = path.stem
    return op_name, task_desc


def load_skills_from_paths(paths: list) -> list:
    """从用户指定的路径列表加载 skill。

    每个路径可以是：
      - 目录: 递归加载目录下所有 SKILL.md
      - 文件: 直接加载单个 SKILL.md（路径必须指向 SKILL.md 文件）
    """
    from akg_agents.core_v2.skill import SkillLoader

    loader = SkillLoader()
    skills = []
    for raw in paths:
        p = Path(raw).resolve()
        if not p.exists():
            logger.warning(f"路径不存在，跳过: {p}")
            continue
        if p.is_dir():
            loaded = loader.load_from_directory(p)
            logger.info(f"从目录 {p} 加载了 {len(loaded)} 个 skill")
            skills.extend(loaded)
        elif p.is_file() and p.name == "SKILL.md":
            loaded = loader.load_single(p)
            if loaded:
                logger.info(f"加载 skill: {loaded.name} ({p})")
                skills.append(loaded)
            else:
                logger.warning(f"解析失败: {p}")
        else:
            logger.warning(f"不是 SKILL.md 文件也不是目录，跳过: {p}")

    for s in skills:
        logger.info(f"  - {s.name}: {s.description[:80]}...")
    return skills


def inject_evolved_skills(task, dsl: str, evolved_skills: list):
    """将 evolved skills 强制注入 KernelGen 的 skill 缓存。

    KernelGen._load_skills_by_dsl 只从 guides/ 加载，这里直接往缓存里追加
    evolved skills，使其参与后续的 skill 选择和 prompt 组装。
    """
    kernel_gen = task.agents.get("kernel_gen")
    if not kernel_gen:
        logger.error("LangGraphTask 中未找到 kernel_gen agent")
        return

    dsl_key = dsl.replace("_", "-")
    existing = kernel_gen._skills_cache.get(dsl_key, [])
    if not existing:
        kernel_gen._load_skills_by_dsl(dsl)
        existing = kernel_gen._skills_cache.get(dsl_key, [])

    existing_names = {s.name for s in existing}
    new_skills = [s for s in evolved_skills if s.name not in existing_names]
    kernel_gen._skills_cache[dsl_key] = existing + new_skills
    logger.info(
        f"注入 {len(new_skills)} 个 evolved skill 到 KernelGen 缓存 "
        f"(总计 {len(existing) + len(new_skills)} 个)"
    )


async def run_single(
    op_name: str,
    task_desc: str,
    dsl: str,
    backend: str,
    arch: str,
    framework: str,
    ab_mode: str,
    evolved_skills: list,
    device: int,
    workflow: str,
    task_type: str,
) -> dict:
    """运行单次测试，返回结果字典。"""
    from akg_agents.op.config.config_validator import load_config
    from akg_agents.op.langgraph_op.task import LangGraphTask
    from akg_agents.core.worker.manager import register_local_worker

    await register_local_worker([device], backend=backend, arch=arch)

    config = load_config(dsl, backend=backend, workflow=workflow.replace("_workflow", ""))

    task = LangGraphTask(
        op_name=op_name,
        task_desc=task_desc,
        task_id=f"verify_{ab_mode}",
        dsl=dsl,
        backend=backend,
        arch=arch,
        config=config,
        framework=framework,
        workflow=workflow,
        task_type=task_type,
    )

    if ab_mode == "B" and evolved_skills:
        inject_evolved_skills(task, dsl, evolved_skills)

    t0 = time.time()
    _, success, final_state = await task.run()
    elapsed = time.time() - t0

    result = {
        "ab_mode": ab_mode,
        "op_name": op_name,
        "success": success,
        "elapsed_s": round(elapsed, 1),
        "speedup": final_state.get("best_speedup"),
        "gen_time_us": final_state.get("best_gen_time"),
        "evolved_skills_injected": [s.name for s in evolved_skills] if ab_mode == "B" else [],
    }
    return result


def _fmt_val(val, fmt="{:.2f}", suffix=""):
    if val is None:
        return "N/A"
    return f"{fmt.format(float(val))}{suffix}"


def print_results(results: list):
    """打印对比结果。"""
    print(f"\n{'='*80}")
    print("  Evolved Skill 验证结果")
    print(f"{'='*80}")
    print(f"  {'组别':20s} | 状态 | {'Speedup':>10s} | {'GenTime':>12s} | 耗时")
    print(f"  {'-'*20}-+------+{'-'*12}+{'-'*14}+-------")
    for r in results:
        mode_label = "A (baseline)" if r["ab_mode"] == "A" else "B (+ skill)"
        status = "PASS" if r["success"] else "FAIL"
        sp = _fmt_val(r.get("speedup"), "{:.2f}", "x")
        gt = _fmt_val(r.get("gen_time_us"), "{:.1f}", "us")
        print(f"  {mode_label:20s} | {status:4s} | {sp:>10s} | {gt:>12s} | {r['elapsed_s']}s")
        if r.get("evolved_skills_injected"):
            print(f"  {'':20s}   注入: {', '.join(r['evolved_skills_injected'])}")
    print(f"{'='*80}")

    a = next((r for r in results if r["ab_mode"] == "A"), None)
    b = next((r for r in results if r["ab_mode"] == "B"), None)
    if a and b:
        diffs = []
        if a.get("speedup") is not None and b.get("speedup") is not None:
            diffs.append(f"Speedup: {b['speedup'] - a['speedup']:+.2f}x")
        if a.get("gen_time_us") is not None and b.get("gen_time_us") is not None:
            diffs.append(f"GenTime: {b['gen_time_us'] - a['gen_time_us']:+.1f}us")
        if diffs:
            print(f"  差异 (B - A): {' | '.join(diffs)}")
            print(f"{'='*80}")


async def main_async(args):
    op_name, task_desc = load_task_file(args.task_file)
    logger.info(f"算子: {op_name}, DSL: {args.dsl}, backend: {args.backend}")

    evolved_skills = load_skills_from_paths(args.skill_paths) if args.skill_paths else []
    if not evolved_skills:
        logger.warning("未指定 skill 或加载为空，B 组将与 A 组相同")

    modes = {"A": "baseline", "B": "evolved"}
    if args.mode != "both":
        modes = {args.mode: modes[args.mode]}

    results = []
    for mode in modes:
        logger.info(f"\n{'─'*60}")
        logger.info(f"运行 {mode} 组 ({modes[mode]})")
        logger.info(f"{'─'*60}")
        r = await run_single(
            op_name=op_name,
            task_desc=task_desc,
            dsl=args.dsl,
            backend=args.backend,
            arch=args.arch,
            framework=args.framework,
            ab_mode=mode,
            evolved_skills=evolved_skills,
            device=args.device,
            workflow=args.workflow,
            task_type=args.task_type,
        )
        results.append(r)

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    result_file = output_path / f"verify_result_{op_name}.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"结果保存: {result_file}")

    print_results(results)


def main():
    parser = argparse.ArgumentParser(
        description="验证 evolved skill 对特定算子的效果",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--task-file", required=True, help="算子任务文件路径（标准 bench_lite 格式的 .py）")
    parser.add_argument("--skill-paths", nargs="+", default=None,
                        help="要验证的 skill 路径（支持目录、SKILL.md 文件，可多个混合传入）")
    parser.add_argument("--dsl", default="triton_ascend", help="DSL 类型（默认 triton_ascend）")
    parser.add_argument("--backend", default="ascend", help="硬件后端（默认 ascend）")
    parser.add_argument("--arch", default="", help="硬件架构（如 ascend910b4, a100）")
    parser.add_argument("--framework", default="torch", help="框架（默认 torch）")
    parser.add_argument("--task-type", choices=["profile", "precision_only"], default="profile",
                        help="任务类型: profile=正确性+性能, precision_only=仅正确性（默认 profile）")
    parser.add_argument("--mode", choices=["A", "B", "both"], default="both",
                        help="A=baseline, B=注入指定skill, both=先A后B对比（默认）")
    parser.add_argument("--device", type=int, default=0, help="设备 ID（默认 0）")
    parser.add_argument("--workflow", default="kernelgen_only_workflow",
                        help="workflow 名称（默认 kernelgen_only_workflow）")
    parser.add_argument("--output-dir", "-o", default="~/.akg/skill_verify_results",
                        help="结果输出目录（默认 ~/.akg/skill_verify_results）")
    args = parser.parse_args()
    args.output_dir = os.path.expanduser(args.output_dir)
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
