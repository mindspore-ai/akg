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
A/B 测试工具函数

包含 Skill Evolution A/B 测试的核心功能：
- Run ID 和目录管理
- 配置生成
- 测试运行
- 日志文件系统解析
- 结果收集
- Tracking.md 更新
"""

import argparse
import glob
import json
import os
import re
import secrets
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


# ============================================================================
# Run ID 和目录管理
# ============================================================================

def generate_run_dir(output_base: str) -> str:
    """生成唯一的运行目录: {output_base}/run_{timestamp}_{random4hex}/"""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    rand = secrets.token_hex(2)
    run_dir = os.path.expanduser(f"{output_base}/run_{ts}_{rand}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def save_run_config(run_dir: str, args: argparse.Namespace) -> None:
    """将运行元信息保存到 run_config.json。"""
    config = {
        "run_dir": run_dir,
        "start_time": datetime.now().isoformat(),
        "groups": args.group,
        "mode": args.mode,
        "max_rounds": args.max_rounds,
        "device": args.device,
        "dsl": "triton_ascend",
        "backend": "ascend",
        "arch": "ascend910b4",
        "evolved_skill_dir": args.evolved_skill_dir,
        "agent_config": args.agent_config,
        "log_dir": args.log_dir,
    }
    with open(os.path.join(run_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


# ============================================================================
# 配置生成
# ============================================================================

def build_evolve_config(
    group: int,
    ab_mode: str,
    run_dir: str,
    device: int,
    evolved_skill_dir: str,
    base_config_path: str,
    max_rounds: int,
    project_root: Path,
    task_dir: Optional[str] = None,
) -> str:
    """生成临时 evolve_config.yaml 并返回路径。"""
    if task_dir is None:
        task_dir = str(
            project_root / "benchmark" / "akg_kernels_bench" / "thirdparty" / "pytorch" / f"group_{group}"
        )
    output_dir = os.path.join(run_dir, f"group_{group}_{ab_mode}")

    config = {
        "base": {
            "dsl": "triton_ascend",
            "framework": "torch",
            "backend": "ascend",
            "arch": "ascend910b4",
            "config_path": base_config_path,
        },
        "evolve": {
            "max_rounds": max_rounds,
            "parallel_num": 1,
        },
        "island": {
            "num_islands": 1,
            "migration_interval": 0,
            "elite_size": 0,
        },
        "devices": {
            "device_list": [device],
        },
        "batch": {
            "parallel_num": 1,
            "device_pool": [device],
            "task_dir": task_dir,
            "output_dir": output_dir,
        },
    }

    tmp_dir = tempfile.mkdtemp(prefix="ab_test_")
    config_path = os.path.join(tmp_dir, "evolve_config.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    agent_config_src = base_config_path
    if not Path(agent_config_src).is_absolute():
        akg_root = project_root / "python" / "akg_agents"
        agent_config_src = str(akg_root / agent_config_src)

    if not Path(agent_config_src).exists():
        print(f"警告: agent config 不存在: {agent_config_src}")
        return config_path

    with open(agent_config_src, "r", encoding="utf-8") as f:
        agent_config = yaml.safe_load(f) or {}

    agent_config["ab_test_mode"] = True
    if ab_mode == "B" and evolved_skill_dir:
        base = evolved_skill_dir
        if not Path(base).is_absolute():
            base = str(project_root / base)
        fix_dir = os.path.join(base, "evolved-fix")
        imp_dir = os.path.join(base, "evolved-improvement")
        dirs = [d for d in (fix_dir, imp_dir) if os.path.isdir(d)]
        agent_config["evolved_skill_dirs"] = dirs
        agent_config["evolved_skill_dir"] = dirs[0] if dirs else base

    agent_config_dst = os.path.join(tmp_dir, "agent_config.yaml")
    with open(agent_config_dst, "w", encoding="utf-8") as f:
        yaml.dump(agent_config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    config["base"]["config_path"] = agent_config_dst
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    return config_path


# ============================================================================
# 运行
# ============================================================================

def run_group(
    group: int,
    ab_mode: str,
    run_dir: str,
    args: argparse.Namespace,
    project_root: Path,
    task_dir: Optional[str] = None,
):
    label = "无外部知识注入 (baseline)" if ab_mode == "A" else "注入 evolved skill (treatment)"
    suffix = ""
    if task_dir:
        n = len([f for f in os.listdir(task_dir) if f.endswith(".py")])
        suffix = f"  ({n} 个算子)"
    print(f"\n{'='*80}")
    print(f"  Group {group} — {ab_mode} 组  {label}{suffix}")
    print(f"{'='*80}\n")

    config_path = build_evolve_config(
        group=group,
        ab_mode=ab_mode,
        run_dir=run_dir,
        device=args.device,
        evolved_skill_dir=args.evolved_skill_dir,
        base_config_path=args.agent_config,
        max_rounds=args.max_rounds,
        project_root=project_root,
        task_dir=task_dir,
    )

    try:
        import asyncio
        sys.path.insert(0, str(project_root / "python"))
        from akg_agents.op.utils.evolve.runner_manager import run_batch_evolve
        asyncio.run(run_batch_evolve(config_path=config_path))
    finally:
        tmp_dir = str(Path(config_path).parent)
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _find_operators_with_skills(run_dir: str, group: int, op_name: Optional[str] = None) -> Union[List[str], bool]:
    """从 B 组输出中找出有 evolved skill 命中的算子文件名列表，或检查单个算子是否命中。

    参数:
        run_dir: 运行目录
        group: 组号
        op_name: 如果指定，只检查该算子是否命中 skill，返回 bool；否则返回所有命中的算子列表

    解析 output_{op_name}_*.txt 查找 skill selection 日志。
    """
    b_dir = os.path.join(run_dir, f"group_{group}_B")
    if not os.path.isdir(b_dir):
        return [] if op_name is None else False

    if op_name:
        pattern = os.path.join(b_dir, f"output_{op_name}_*.txt")
        files = sorted(glob.glob(pattern))
        if not files:
            return False
        with open(files[-1], encoding="utf-8", errors="ignore") as f:
            for line in f:
                if "skills selected:" in line and "Evolved skill selection complete" in line:
                    m = re.search(r"skills selected:\s*\[([^\]]*)\]", line)
                    if m and m.group(1).strip():
                        return True
        return False

    matched_files: List[str] = []
    for out_file in sorted(glob.glob(os.path.join(b_dir, "output_akg_agents_*.txt"))):
        has_skills = False
        task_file_name = None
        with open(out_file, encoding="utf-8", errors="ignore") as f:
            for line in f:
                if "skills selected:" in line and "Evolved skill selection complete" in line:
                    m = re.search(r"skills selected:\s*\[([^\]]*)\]", line)
                    if m and m.group(1).strip():
                        has_skills = True
                if "任务文件:" in line:
                    m = re.search(r"任务文件:\s*(\S+)", line)
                    if m:
                        task_file_name = os.path.basename(m.group(1))
        if has_skills and task_file_name:
            matched_files.append(task_file_name)

    return matched_files


def _create_filtered_task_dir(
    project_root: Path, group: int, allowed_files: List[str]
) -> Optional[str]:
    """创建临时目录，只包含 allowed_files 中的算子文件（符号链接）。"""
    if not allowed_files:
        return None
    source_dir = (
        project_root / "benchmark" / "akg_kernels_bench" / "thirdparty" / "pytorch" / f"group_{group}"
    )
    tmp_dir = tempfile.mkdtemp(prefix=f"ab_filtered_g{group}_")
    for fname in allowed_files:
        src = source_dir / fname
        if src.exists():
            os.symlink(str(src), os.path.join(tmp_dir, fname))
    return tmp_dir


def _run_single_operator(
    group: int,
    ab_mode: str,
    run_dir: str,
    args: argparse.Namespace,
    project_root: Path,
    task_file: str,
):
    """运行单个算子的 A/B 测试。"""
    source_dir = (
        project_root / "benchmark" / "akg_kernels_bench" / "thirdparty" / "pytorch" / f"group_{group}"
    )
    tmp_dir = tempfile.mkdtemp(prefix=f"ab_single_g{group}_")
    src = source_dir / task_file
    if src.exists():
        os.symlink(str(src), os.path.join(tmp_dir, task_file))

    label = "无外部知识注入 (baseline)" if ab_mode == "A" else "注入 evolved skill (treatment)"
    print(f"\n{'='*80}")
    print(f"  Group {group} — {ab_mode} 组 — {task_file} {label}")
    print(f"{'='*80}\n")

    config_path = build_evolve_config(
        group=group,
        ab_mode=ab_mode,
        run_dir=run_dir,
        device=args.device,
        evolved_skill_dir=args.evolved_skill_dir,
        base_config_path=args.agent_config,
        max_rounds=args.max_rounds,
        project_root=project_root,
        task_dir=tmp_dir,
    )

    try:
        import asyncio
        sys.path.insert(0, str(project_root / "python"))
        from akg_agents.op.utils.evolve.runner_manager import run_batch_evolve
        asyncio.run(run_batch_evolve(config_path=config_path))
    finally:
        tmp_dir_cfg = str(Path(config_path).parent)
        shutil.rmtree(tmp_dir_cfg, ignore_errors=True)
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ============================================================================
# 日志文件系统解析
# ============================================================================

def _parse_speed_up_record(record_path: str) -> Dict[str, Dict[str, Any]]:
    """解析 speed_up_record.txt，返回 {task_id: {base_time, gen_time, speedup}}。"""
    result: Dict[str, Dict[str, Any]] = {}
    if not os.path.isfile(record_path):
        return result
    with open(record_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = {}
            for seg in line.split(", "):
                if ": " in seg:
                    k, v = seg.split(": ", 1)
                    parts[k.strip()] = v.strip()
            tid = parts.get("task_id", "")
            if not tid:
                continue
            try:
                result[tid] = {
                    "base_time_us": float(parts.get("base_time", "0").replace(" us", "")),
                    "gen_time_us": float(parts.get("generation_time", "0").replace(" us", "")),
                    "speedup": float(parts.get("speedup", "0").replace("x", "")),
                    "unique_dir": parts.get("unique_dir", ""),
                }
            except ValueError:
                continue
    return result


def _count_conductor_retries(conductor_dir: str, round_idx: int) -> int:
    if not os.path.isdir(conductor_dir):
        return 0
    pattern = os.path.join(conductor_dir, f"Iteration{round_idx}_Task*_conductor_decision_result.txt")
    return len(glob.glob(pattern))


def _classify_error(error_log_dir: str, round_idx: int, op_name: str) -> Optional[str]:
    pattern = os.path.join(error_log_dir, f"Iteration{round_idx}_Task*_{op_name}_verifier_error_log.txt")
    logs = glob.glob(pattern)
    if not logs:
        return None
    try:
        with open(logs[-1], encoding="utf-8", errors="ignore") as f:
            content = f.read(2000)
    except OSError:
        return None
    low = content.lower()
    if "syntaxerror" in low or "compileerror" in low or "编译" in low:
        return "compile"
    if "验证超时" in low or "timed out" in low or "timeouterror" in low:
        return "timeout"
    if "验证失败" in content or "err_cnt" in content or "输出不一致" in content:
        return "accuracy"
    if content.strip():
        return "runtime"
    return None


def _parse_selected_skills(output_file: str) -> List[str]:
    skills: List[str] = []
    if not os.path.isfile(output_file):
        return skills
    try:
        with open(output_file, encoding="utf-8", errors="ignore") as f:
            for line in f:
                if "Evolved skill selection complete" in line and "skills selected:" in line:
                    m = re.search(r"skills selected:\s*\[([^\]]*)\]", line)
                    if m:
                        raw = m.group(1)
                        skills = [s.strip().strip("'\"") for s in raw.split(",") if s.strip()]
                    break
    except OSError:
        pass
    return skills


def _find_task_folders(output_dir: str) -> Dict[str, str]:
    """返回 op_name -> task_folder 的映射。"""
    mapping: Dict[str, str] = {}
    for txt in sorted(glob.glob(os.path.join(output_dir, "realtime_results_*.txt"))):
        current_op = None
        with open(txt, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("akg_agents_"):
                    current_op = line
                m = re.match(r"Task文件夹:\s*(\S+)", line)
                if m and current_op:
                    mapping[current_op] = m.group(1)
                    current_op = None
    return mapping


def _find_task_folders_list(output_dir: str) -> List[Dict[str, Any]]:
    """返回按顺序排列的 task_folder 列表，每个元素包含 op_name 和 task_folder。"""
    folders: List[Dict[str, Any]] = []
    for txt in sorted(glob.glob(os.path.join(output_dir, "realtime_results_*.txt"))):
        current_op = None
        with open(txt, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("akg_agents_"):
                    current_op = line
                m = re.match(r"Task文件夹:\s*(\S+)", line)
                if m and current_op:
                    folders.append({
                        "op_name": current_op,
                        "task_folder": m.group(1)
                    })
                    current_op = None
    return folders


def _parse_realtime_results(realtime_file: str) -> Dict[int, Dict[str, Any]]:
    """解析单个 realtime_results_*.txt 文件，返回按索引排列的映射。"""
    result: Dict[int, Dict[str, Any]] = {}
    idx = 0
    current_op = None
    with open(realtime_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("akg_agents_"):
                current_op = line
            m = re.match(r"Task文件夹:\s*(\S+)", line)
            if m and current_op:
                result[idx] = {
                    "op_name": current_op,
                    "task_folder": m.group(1)
                }
                idx += 1
                current_op = None
    return result


def _find_output_file(output_dir: str, op_name: str) -> Optional[str]:
    pattern = os.path.join(output_dir, f"output_{op_name}_*.txt")
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


# ============================================================================
# 结果收集
# ============================================================================

def collect_group_results(
    group: int,
    ab_mode: str,
    run_dir: str,
    log_dir_base: str,
    max_rounds: int,
) -> List[Dict[str, Any]]:
    """从日志文件系统收集一个 group+mode 的全部算子详细结果。

    支持两种模式：
    1. 批量运行：读取最后一个 batch_summary_*.json
    2. 逐算子运行：合并所有 batch_summary_*.json 文件
    """
    output_dir = os.path.join(run_dir, f"group_{group}_{ab_mode}")
    if not os.path.isdir(output_dir):
        return []

    log_base = os.path.expanduser(log_dir_base)

    summaries = sorted(glob.glob(os.path.join(output_dir, "batch_summary_*.json")))
    if not summaries:
        return []

    all_task_results: List[Dict[str, Any]] = []
    total_execution_time = 0.0
    for summary_path in summaries:
        try:
            with open(summary_path, encoding="utf-8") as f:
                batch = json.load(f)
                all_task_results.extend(batch.get("task_results", []))
                batch_info = batch.get("batch_info", {})
                total_execution_time += batch_info.get("total_execution_time_seconds", 0)
        except (json.JSONDecodeError, OSError):
            continue

    if not all_task_results:
        return []

    task_folder_list = _find_task_folders_list(output_dir)

    results: List[Dict[str, Any]] = []
    for i, tr in enumerate(all_task_results):
        op_name = tr["op_name"]
        task = task_folder_list[i] if i < len(task_folder_list) else None
        task_folder = task.get("task_folder", "") if task else ""
        short_name = re.sub(r"^akg_agents_", "", op_name)

        op_log_dir = os.path.join(log_base, task_folder, op_name) if task_folder else ""
        profiling_dir = os.path.join(op_log_dir, "profiling") if op_log_dir else ""
        conductor_dir = os.path.join(op_log_dir, "conductor") if op_log_dir else ""
        record_path = os.path.join(profiling_dir, "speed_up_record.txt") if profiling_dir else ""

        speed_records = _parse_speed_up_record(record_path)

        output_file = _find_output_file(output_dir, op_name)
        selected_skills = _parse_selected_skills(output_file) if output_file else []

        rounds_data: List[Dict[str, Any]] = []
        for r_idx in range(1, max_rounds + 1):
            tid = f"{r_idx}_Task0"
            rec = speed_records.get(tid)
            success = rec is not None
            conductor_retries = _count_conductor_retries(conductor_dir, r_idx) if op_log_dir else 0
            error_type = None
            if not success and op_log_dir:
                error_type = _classify_error(op_log_dir, r_idx, op_name)
            rounds_data.append({
                "round": r_idx,
                "success": success,
                "speedup": rec["speedup"] if rec else None,
                "gen_time_us": rec["gen_time_us"] if rec else None,
                "base_time_us": rec["base_time_us"] if rec else None,
                "conductor_retries": conductor_retries,
                "error_type": error_type,
            })

        first_success_round = next(
            (rd["round"] for rd in rounds_data if rd["success"]), None
        )
        total_retries = sum(rd["conductor_retries"] for rd in rounds_data)
        error_types = list({rd["error_type"] for rd in rounds_data if rd["error_type"]})

        entry = {
            "op_name": short_name,
            "group": group,
            "ab_mode": ab_mode,
            "success": tr.get("success", False),
            "best_speedup": tr.get("best_speedup"),
            "execution_time_s": tr.get("execution_time"),
            "first_success_round": first_success_round,
            "total_conductor_retries": total_retries,
            "error_types": ", ".join(sorted(error_types)) if error_types else "",
            "selected_skills": selected_skills,
            "task_folder": task_folder,
            "rounds": rounds_data,
        }
        results.append(entry)

    if results and total_execution_time > 0:
        results[0]["_batch_total_execution_time"] = total_execution_time

    return results


# ============================================================================
# Tracking.md 更新
# ============================================================================

def _fmt(val, fmt_str="{:.2f}") -> str:
    if val is None:
        return "-"
    try:
        return fmt_str.format(float(val))
    except (ValueError, TypeError):
        return str(val)


TRACKING_ROUND_COLS = 3


def _split_by_group(content: str) -> Dict[int, str]:
    sec4_pos = content.find("## 4.")
    search_end = sec4_pos if sec4_pos > 0 else len(content)

    result: Dict[int, str] = {}
    pattern = re.compile(r"^### Group (\d+)", re.MULTILINE)
    matches = list(pattern.finditer(content, 0, search_end))
    for i, m in enumerate(matches):
        group_num = int(m.group(1))
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else search_end
        next_section = re.search(r"^## \d+\.", content[start:end], re.MULTILINE)
        if next_section:
            end = start + next_section.start()
        result[group_num] = content[start:end]
    return result


def _add_group_section(
    content: str, group: int, all_results: List[Dict[str, Any]]
) -> str:
    """为 tracking.md 动态添加一个缺失的 Group section。"""
    ops_in_group = sorted({
        r["op_name"] for r in all_results if r["group"] == group
    })
    header = (
        f"\n### Group {group}\n\n"
        "| 算子 | A/B | R1 成功 | R1 Speedup | R2 成功 | R2 Speedup | R3 成功 | R3 Speedup "
        "| 首次成功轮次 | 总生成时间(s) | 重试次数 | 错误类型 | 采样到的 Skill |\n"
        "|------|-----|--------|-----------|--------|-----------|--------|-----------|"
        "------------|-------------|---------|---------|--------------|"
    )
    rows = []
    for op in ops_in_group:
        for ab in ["A", "B"]:
            rows.append(f"| {op} | {ab} | - | - | - | - | - | - | - | - | - | - | - |")

    section_text = header + "\n" + "\n".join(rows) + "\n"

    sec4_marker = "## 4. 汇总统计"
    pos = content.find(sec4_marker)
    if pos > 0:
        content = content[:pos] + section_text + "\n" + content[pos:]
    else:
        content += section_text

    return content


def update_tracking_md(
    all_results: List[Dict[str, Any]],
    tracking_path: str,
    run_config: Dict[str, Any],
):
    """更新 tracking.md: 逐算子详情 + 实验运行记录 + 汇总统计。"""
    if not os.path.isfile(tracking_path):
        print(f"  tracking.md 不存在: {tracking_path}，跳过更新")
        return

    with open(tracking_path, "r", encoding="utf-8") as f:
        content = f.read()

    # --- 1. 逐算子详情 ---
    needed_groups = {entry["group"] for entry in all_results}
    group_sections = _split_by_group(content)
    for g in sorted(needed_groups):
        if g not in group_sections:
            content = _add_group_section(content, g, all_results)
    group_sections = _split_by_group(content)

    for entry in all_results:
        group = entry["group"]
        op_name = entry["op_name"]
        ab = entry["ab_mode"]
        rounds = entry.get("rounds", [])

        cells = [op_name, ab]
        for r_idx in range(1, TRACKING_ROUND_COLS + 1):
            rd = next((rd for rd in rounds if rd["round"] == r_idx), None)
            if rd and rd["success"]:
                cells.append("Y")
                cells.append(_fmt(rd["speedup"]))
            elif rd:
                cells.append("N")
                cells.append("-")
            else:
                cells.append("-")
                cells.append("-")

        cells.append(str(entry["first_success_round"]) if entry.get("first_success_round") else "-")
        cells.append(_fmt(entry.get("execution_time_s"), "{:.1f}"))
        cells.append(str(entry.get("total_conductor_retries", 0)))
        cells.append(entry.get("error_types") or "-")
        skills = entry.get("selected_skills", [])
        cells.append(", ".join(skills) if skills else "-")

        new_row = "| " + " | ".join(cells) + " |"
        row_pattern = re.compile(
            r"\| *" + re.escape(op_name) + r" *\| *" + re.escape(ab) + r" *\|[^\n]*"
        )
        if group in group_sections:
            old_section = group_sections[group]
            new_section = row_pattern.sub(new_row, old_section)
            if new_section != old_section:
                content = content.replace(old_section, new_section)
                group_sections[group] = new_section

    # --- 2. 实验运行记录 (Section 2) ---
    content = _update_experiment_record(content, all_results, run_config)

    # --- 3. 汇总统计 (Section 4) ---
    content = _update_summary_stats(content, all_results)

    # --- 4. 算子级 Speedup 对比 (Section 5) ---
    content = _update_speedup_comparison(content, all_results)

    with open(tracking_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  tracking.md 已更新: {tracking_path}")


def _update_experiment_record(
    content: str,
    all_results: List[Dict[str, Any]],
    run_config: Dict[str, Any],
) -> str:
    """在 '## 2. 实验运行记录' 表中追加本次运行。"""
    run_id = os.path.basename(run_config.get("run_dir", ""))
    dsl = run_config.get("dsl", "")
    backend = run_config.get("backend", "")
    arch = run_config.get("arch", "")
    max_rounds = run_config.get("max_rounds", 3)
    date = datetime.now().strftime("%Y-%m-%d")

    groups_seen: Dict[int, Dict[str, List]] = {}
    for r in all_results:
        g = r["group"]
        ab = r["ab_mode"]
        groups_seen.setdefault(g, {}).setdefault(ab, []).append(r)

    new_rows: List[str] = []
    for g in sorted(groups_seen):
        for ab in ["B", "A"]:
            entries = groups_seen.get(g, {}).get(ab, [])
            if not entries:
                continue
            n_ok = sum(1 for e in entries if e["success"])
            sr = f"{n_ok}/{len(entries)}"
            speedups = [e["best_speedup"] for e in entries if e.get("best_speedup") is not None]
            avg_sp = _fmt(sum(speedups) / len(speedups) if speedups else 0)
            if entries and "_batch_total_execution_time" in entries[0]:
                total_time = entries[0]["_batch_total_execution_time"]
            else:
                total_time = sum(e.get("execution_time_s", 0) or 0 for e in entries)
            skill_ver = "v0.1" if ab == "B" else "N/A"
            note = f"{dsl}/{backend}/{arch}"
            row = (
                f"| {run_id} | {date} | {g} | {ab} | {skill_ver} | {max_rounds} | 1 "
                f"| {sr} | {avg_sp} | {_fmt(total_time, '{:.0f}')}s | {note} |"
            )
            new_rows.append(row)

    if not new_rows:
        return content

    sec2_start = content.find("## 2. 实验运行记录")
    if sec2_start < 0:
        return content

    sep_match = re.search(r"\|[-:| ]+\|\n", content[sec2_start:])
    if not sep_match:
        return content

    insert_pos = sec2_start + sep_match.end()

    existing_rows = ""
    rest_start = insert_pos
    while rest_start < len(content) and content[rest_start] in ['\n', '\r']:
        rest_start += 1
    while rest_start < len(content) and content[rest_start] == '|':
        line_end = content.find('\n', rest_start)
        if line_end < 0:
            line_end = len(content)
        existing_rows += content[rest_start:line_end + 1]
        rest_start = line_end + 1

    all_row_text = existing_rows + "\n".join(new_rows) + "\n"
    content = content[:insert_pos] + all_row_text + content[rest_start:]

    return content


def _parse_summary_header_groups(content: str) -> List[int]:
    """从汇总统计表头解析出包含的 Group 编号列表。"""
    sec4 = content.find("## 4. 汇总统计")
    if sec4 < 0:
        return [1, 2, 3]
    header_area = content[sec4:sec4 + 500]
    return sorted(set(int(m.group(1)) for m in re.finditer(r"Group (\d+)", header_area)))


def _ensure_summary_groups(content: str, needed_groups: List[int]) -> str:
    """确保汇总统计表头包含所有需要的 Group 列，不足则重建表头。"""
    existing = _parse_summary_header_groups(content)
    all_groups = sorted(set(existing) | set(needed_groups))
    if all_groups == existing:
        return content

    sec4 = content.find("## 4. 汇总统计")
    if sec4 < 0:
        return content
    sec5 = content.find("## 5.", sec4 + 1)
    if sec5 < 0:
        sec5 = len(content)

    cols = []
    for g in all_groups:
        cols.extend([f"Group {g} A", f"Group {g} B"])
    cols.extend(["总 A", "总 B"])

    header = "| 指标 | " + " | ".join(cols) + " |"
    sep_cells = ["------"] * (len(cols) + 1)
    sep = "|" + "|".join(sep_cells) + "|"
    metric_rows = [
        "成功率", "平均首次成功轮次", "平均 Speedup", "平均生成时间", "Skill 命中率"
    ]
    data_rows = []
    for m in metric_rows:
        data_rows.append("| " + m + " | " + " | ".join(["-"] * len(cols)) + " |")

    new_table = "\n".join([
        "## 4. 汇总统计", "", header, sep, *data_rows, ""
    ])
    content = content[:sec4] + new_table + "\n" + content[sec5:]
    return content


def _update_summary_stats(content: str, all_results: List[Dict[str, Any]]) -> str:
    """更新 '## 4. 汇总统计' 中的指标。

    增量更新：只覆盖当前运行包含的 group 列，保留之前运行写入的其他 group 数据，
    并从所有 per-group 列重新计算汇总列（总 A / 总 B）。
    """
    result_groups = sorted({r["group"] for r in all_results})
    content = _ensure_summary_groups(content, result_groups)
    header_groups = _parse_summary_header_groups(content)

    stats: Dict[str, Dict[str, Any]] = {}
    for r in all_results:
        key = f"Group {r['group']} {r['ab_mode']}"
        stats.setdefault(key, {"ok": 0, "total": 0, "speedups": [], "times": [], "first_rounds": [], "skill_hits": 0})
        s = stats[key]
        s["total"] += 1
        if r["success"]:
            s["ok"] += 1
        if r.get("best_speedup") is not None:
            s["speedups"].append(r["best_speedup"])
        if r.get("execution_time_s") is not None:
            s["times"].append(r["execution_time_s"])
        if r.get("first_success_round") is not None:
            s["first_rounds"].append(r["first_success_round"])
        if r.get("selected_skills"):
            s["skill_hits"] += 1

    def _col(key: str, metric: str) -> str:
        if key not in stats:
            return "-"
        s = stats[key]
        if metric == "sr":
            return f"{s['ok']}/{s['total']}" if s["total"] else "-"
        if metric == "sp":
            return _fmt(sum(s["speedups"]) / len(s["speedups"])) if s["speedups"] else "-"
        if metric == "fr":
            return _fmt(sum(s["first_rounds"]) / len(s["first_rounds"]), "{:.1f}") if s["first_rounds"] else "-"
        if metric == "time":
            return _fmt(sum(s["times"]) / len(s["times"]), "{:.0f}") if s["times"] else "-"
        if metric == "skill":
            return f"{s.get('skill_hits', 0)}/{s['total']}" if s["total"] else "-"
        return "-"

    sec4_start = content.find("## 4. 汇总统计")
    sec5_start = content.find("## 5.", sec4_start + 1) if sec4_start >= 0 else -1
    if sec4_start < 0:
        return content
    sec4_end = sec5_start if sec5_start > 0 else len(content)

    sec4_text = content[sec4_start:sec4_end]

    def _parse_existing_row(metric_name: str) -> List[str]:
        for line in sec4_text.split('\n'):
            if not line.strip().startswith('|'):
                continue
            parts = [p.strip() for p in line.split('|')]
            if len(parts) >= 2 and parts[1] == metric_name:
                return parts[2:-1] if (parts and parts[-1] == '') else parts[2:]
        return []

    def _recalc_total(cells: List[str], metric_key: str) -> str:
        if metric_key in ("sr", "skill"):
            total_ok, total_n = 0, 0
            for c in cells:
                c = c.strip()
                if c in ("-", "N/A"):
                    continue
                m = re.match(r"(\d+)/(\d+)", c)
                if m:
                    total_ok += int(m.group(1))
                    total_n += int(m.group(2))
            return f"{total_ok}/{total_n}" if total_n > 0 else "-"
        vals = []
        for c in cells:
            c = c.strip()
            if c in ("-", "N/A"):
                continue
            try:
                vals.append(float(c))
            except ValueError:
                continue
        if not vals:
            return "-"
        avg = sum(vals) / len(vals)
        if metric_key == "time":
            return _fmt(avg, "{:.0f}")
        if metric_key == "fr":
            return _fmt(avg, "{:.1f}")
        return _fmt(avg)

    def _build_row(metric_name: str, metric_key: str) -> None:
        nonlocal sec4_text

        existing = _parse_existing_row(metric_name)

        group_cells: List[str] = []
        col_idx = 0
        for g in header_groups:
            for ab in ["A", "B"]:
                k = f"Group {g} {ab}"
                if k in stats:
                    if metric_key == "skill" and ab == "A":
                        group_cells.append("N/A")
                    else:
                        group_cells.append(_col(k, metric_key))
                else:
                    group_cells.append(existing[col_idx] if col_idx < len(existing) else "-")
                col_idx += 1

        a_cells = [group_cells[j] for j in range(0, len(group_cells), 2)]
        b_cells = [group_cells[j] for j in range(1, len(group_cells), 2)]

        total_a = "N/A" if metric_key == "skill" else _recalc_total(a_cells, metric_key)
        total_b = _recalc_total(b_cells, metric_key)

        all_vals = group_cells + [total_a, total_b]
        new_row = f"| {metric_name} | " + " | ".join(all_vals) + " |"

        lines = sec4_text.split('\n')
        for i, line in enumerate(lines):
            if line.strip().startswith('|') and metric_name in line:
                parts = [p.strip() for p in line.split('|')]
                if parts and parts[1] == metric_name:
                    lines[i] = new_row
                    break
        sec4_text = '\n'.join(lines)

    _build_row("成功率", "sr")
    _build_row("平均首次成功轮次", "fr")
    _build_row("平均 Speedup", "sp")
    _build_row("平均生成时间", "time")
    _build_row("Skill 命中率", "skill")

    content = content[:sec4_start] + sec4_text + content[sec4_end:]
    return content


def _parse_speedup_groups(
    section_text: str,
) -> Dict[int, Dict[str, Dict[str, Optional[float]]]]:
    """从已有的 '算子级 Speedup 对比' section 文本中解析出 per-group 数据。"""
    result: Dict[int, Dict[str, Dict[str, Optional[float]]]] = {}
    current_group: Optional[int] = None
    for line in section_text.split('\n'):
        gm = re.match(r"^### Group (\d+)", line)
        if gm:
            current_group = int(gm.group(1))
            result[current_group] = {}
            continue
        if current_group is None or not line.startswith('|') or '---' in line:
            continue
        cells = [c.strip() for c in line.split('|')]
        cells = [c for c in cells if c]
        if len(cells) < 3 or cells[0] == '算子':
            continue
        op_name = cells[0]
        op_data: Dict[str, Optional[float]] = {}
        for idx, key in [(1, "A"), (2, "B")]:
            try:
                val = cells[idx].strip()
                if val not in ('-', ''):
                    op_data[key] = float(val)
            except (ValueError, IndexError):
                pass
        result[current_group][op_name] = op_data
    return result


def _update_speedup_comparison(content: str, all_results: List[Dict[str, Any]]) -> str:
    """更新 '算子级 Speedup 对比' section —— 每个 Group 输出逐算子最佳 Speedup 对比表。

    增量更新：只覆盖当前运行包含的 group，保留之前运行写入的其他 group 数据。
    """
    new_groups: Dict[int, Dict[str, Dict[str, Optional[float]]]] = {}
    for r in all_results:
        g = r["group"]
        op = r["op_name"]
        ab = r["ab_mode"]
        new_groups.setdefault(g, {}).setdefault(op, {})[ab] = r.get("best_speedup")

    if not new_groups:
        return content

    existing_match = re.search(
        r"^(## \d+\.)\s*算子级 Speedup 对比", content, re.MULTILINE
    )

    if existing_match:
        sec_start = existing_match.start()
        next_sec = re.search(r"^## \d+\.", content[sec_start + 1:], re.MULTILINE)
        sec_end = (sec_start + 1 + next_sec.start()) if next_sec else len(content)
        section_title = existing_match.group(0).rstrip()
        existing_groups = _parse_speedup_groups(content[sec_start:sec_end])
    else:
        all_sec_nums = re.findall(r"^## (\d+)\.", content, re.MULTILINE)
        max_num = max(int(n) for n in all_sec_nums) if all_sec_nums else 0
        section_title = f"## {max_num + 1}. 算子级 Speedup 对比"
        sec_start = None
        sec_end = None
        existing_groups = {}

    merged = dict(existing_groups)
    merged.update(new_groups)

    lines = [section_title, ""]
    for g in sorted(merged):
        lines.append(f"### Group {g}")
        lines.append("")
        lines.append("| 算子 | A Speedup | B Speedup | 差异 (B-A) |")
        lines.append("|------|-----------|-----------|------------|")
        for op in sorted(merged[g]):
            a_sp = merged[g][op].get("A")
            b_sp = merged[g][op].get("B")
            a_str = _fmt(a_sp)
            b_str = _fmt(b_sp)
            if a_sp is not None and b_sp is not None:
                diff = b_sp - a_sp
                diff_str = f"{diff:+.2f}"
                if diff > 0:
                    diff_str = f"**{diff_str}**"
            else:
                diff_str = "-"
            lines.append(f"| {op} | {a_str} | {b_str} | {diff_str} |")
        lines.append("")

    new_section = "\n".join(lines) + "\n"

    if sec_start is not None:
        content = content[:sec_start] + new_section + content[sec_end:]
    else:
        content = content.rstrip() + "\n\n" + new_section

    return content


def _clear_tracking_md(tracking_path: str) -> bool:
    """清空 tracking.md 中的所有实验数据，恢复到初始空表状态。

    保留文件结构（section 标题、表头、算子名列表），只重置数据单元格。
    Section 1 及 Section 5+ 完整保留不动。
    """
    if not os.path.isfile(tracking_path):
        print(f"  tracking.md 不存在：{tracking_path}，跳过清空")
        return False

    with open(tracking_path, "r", encoding="utf-8") as f:
        content = f.read()

    # === Section 2: 清空实验运行记录数据行，保留表头 ===
    sec2_start = content.find("## 2. 实验运行记录")
    if sec2_start >= 0:
        sec3_start = content.find("## 3.", sec2_start + 1)
        if sec3_start < 0:
            sec3_start = len(content)
        sec2_text = content[sec2_start:sec3_start]
        sep_match = re.search(r'\|[-:| ]+\|\n', sec2_text)
        if sep_match:
            content = content[:sec2_start + sep_match.end()] + "\n\n" + content[sec3_start:]

    # === Section 3: 保留 Group 结构和算子名，数据列重置为 "-" ===
    sec3_start = content.find("## 3. 逐算子详情")
    if sec3_start >= 0:
        sec4_start = content.find("## 4.", sec3_start + 1)
        if sec4_start < 0:
            sec4_start = len(content)
        sec3_text = content[sec3_start:sec4_start]

        def _reset_data_row(m: re.Match) -> str:
            parts = [c.strip() for c in m.group(0).split('|')]
            n_data = len(parts) - 4
            if n_data <= 0:
                return m.group(0)
            return f"| {parts[1]} | {parts[2]} | " + " | ".join(["-"] * n_data) + " |"

        sec3_text = re.sub(
            r'^\|[^|]+\| *(?:A|B) *\|[^\n]*',
            _reset_data_row, sec3_text, flags=re.MULTILINE,
        )
        content = content[:sec3_start] + sec3_text + content[sec4_start:]

    # === Section 4: 重置汇总统计指标行 ===
    sec4_start = content.find("## 4. 汇总统计")
    if sec4_start >= 0:
        sec5_start = content.find("## 5.", sec4_start + 1)
        if sec5_start < 0:
            sec5_start = len(content)
        sec4_text = content[sec4_start:sec5_start]

        metric_names = {"成功率", "平均首次成功轮次", "平均 Speedup", "平均生成时间", "Skill 命中率"}
        lines = sec4_text.split('\n')
        for i, line in enumerate(lines):
            if not line.strip().startswith('|'):
                continue
            parts = [c.strip() for c in line.split('|')]
            if len(parts) < 3 or parts[1] not in metric_names:
                continue
            n_data = len(parts) - 3
            if parts[1] == "Skill 命中率":
                vals = ["N/A" if j % 2 == 0 else "-" for j in range(n_data)]
            else:
                vals = ["-"] * n_data
            lines[i] = f"| {parts[1]} | " + " | ".join(vals) + " |"
        content = content[:sec4_start] + '\n'.join(lines) + content[sec5_start:]

    # === Section 5: 清空 Speedup 对比数据行，保留 Group 表头 ===
    existing_match = re.search(
        r"^(## \d+\.)\s*算子级 Speedup 对比", content, re.MULTILINE
    )
    if existing_match:
        sec5_start = existing_match.start()
        next_sec = re.search(r"^## \d+\.", content[sec5_start + 1:], re.MULTILINE)
        sec5_end = (sec5_start + 1 + next_sec.start()) if next_sec else len(content)
        sec5_text = content[sec5_start:sec5_end]
        section_title = existing_match.group(0).rstrip()

        group_nums = sorted(set(
            int(gm.group(1))
            for gm in re.finditer(r"^### Group (\d+)", sec5_text, re.MULTILINE)
        ))
        if not group_nums:
            group_nums = [1, 2, 3]

        lines = [section_title, ""]
        for g in group_nums:
            lines.extend([
                f"### Group {g}", "",
                "| 算子 | A Speedup | B Speedup | 差异 (B-A) |",
                "|------|-----------|-----------|------------|", "",
            ])
        lines.append("")
        content = content[:sec5_start] + "\n".join(lines) + content[sec5_end:]

    with open(tracking_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"  tracking.md 已清空：{tracking_path}")
    return True
