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
run_ab_test.py - Skill Evolution A/B 测试批量运行器

用法：
  # 运行 Group 1 的 A/B 测试（先 B 后 A，B 中无 skill 命中的算子自动跳过 A）
  python run_ab_test.py --group 1 --mode both --device 0

  # 仅收集某次运行的结果（必须指定 --run-dir）
  python run_ab_test.py --collect-only --run-dir ~/akg_eval_results/run_20260309_155303_a1b2

每次运行会在 output-dir 下创建带时间戳的目录，保证不同运行互不干扰：
  ~/akg_eval_results/run_20260309_155303_a1b2/
    ├── run_config.json          # 运行元信息（dsl/backend/arch 等）
    ├── group_1_B/               # B 组结果
    ├── group_1_A/               # A 组结果（仅包含 B 中有 skill 命中的算子）
    └── ab_detail_group1.json    # 汇总的 A/B 详细结果
"""

import argparse
import glob
import json
import os
import re
import secrets
import sys
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import yaml


# ============================================================================
# 项目路径
# ============================================================================

def get_project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "python" / "akg_agents").is_dir():
            return parent
    raise FileNotFoundError("无法定位项目根目录")


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
        "dsl": "triton_cuda",
        "backend": "cuda",
        "arch": "rtx3090",
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
            "dsl": "triton_cuda",
            "framework": "torch",
            "backend": "cuda",
            "arch": "rtx3090",
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
        agent_config["evolved_skill_dir"] = evolved_skill_dir

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


def _find_operators_with_skills(run_dir: str, group: int) -> List[str]:
    """从 B 组输出中找出有 evolved skill 命中的算子文件名列表。

    解析 output_{op_name}_*.txt 查找 skill selection 日志，
    返回有 skill 命中的算子对应的原始 task 文件名（如 01_LayerNorm.py）。
    """
    b_dir = os.path.join(run_dir, f"group_{group}_B")
    if not os.path.isdir(b_dir):
        return []

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
    """从日志文件系统收集一个 group+mode 的全部算子详细结果。"""
    output_dir = os.path.join(run_dir, f"group_{group}_{ab_mode}")
    if not os.path.isdir(output_dir):
        return []

    log_base = os.path.expanduser(log_dir_base)

    summaries = sorted(glob.glob(os.path.join(output_dir, "batch_summary_*.json")))
    if not summaries:
        return []
    with open(summaries[-1], encoding="utf-8") as f:
        batch = json.load(f)

    task_folders = _find_task_folders(output_dir)

    results: List[Dict[str, Any]] = []
    for tr in batch.get("task_results", []):
        op_name = tr["op_name"]
        task_folder = task_folders.get(op_name, "")
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
    result: Dict[int, str] = {}
    pattern = re.compile(r"^### Group (\d+)", re.MULTILINE)
    matches = list(pattern.finditer(content))
    for i, m in enumerate(matches):
        group_num = int(m.group(1))
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
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

    sep_match = re.search(r"\|[-|]+\|\n", content[sec2_start:])
    if not sep_match:
        return content

    insert_pos = sec2_start + sep_match.end()

    existing_rows = ""
    rest_start = insert_pos
    while rest_start < len(content) and content[rest_start] == '|':
        line_end = content.find('\n', rest_start)
        if line_end < 0:
            line_end = len(content)
        rest_start = line_end + 1
    existing_rows = content[insert_pos:rest_start]

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
    sep = "|------" + "|".join(["------"] * len(cols)) + "|"
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
    """更新 '## 4. 汇总统计' 中的指标。"""
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

    totals = {"A": {"ok": 0, "total": 0, "speedups": [], "first_rounds": []},
              "B": {"ok": 0, "total": 0, "speedups": [], "first_rounds": [], "skill_hits": 0}}
    for key, s in stats.items():
        ab = "B" if key.endswith("B") else "A"
        totals[ab]["ok"] += s["ok"]
        totals[ab]["total"] += s["total"]
        totals[ab]["speedups"].extend(s["speedups"])
        totals[ab]["first_rounds"].extend(s["first_rounds"])
        if ab == "B":
            totals[ab]["skill_hits"] = totals[ab].get("skill_hits", 0) + s.get("skill_hits", 0)

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

    def _total(ab: str, metric: str) -> str:
        t = totals[ab]
        if metric == "sr":
            return f"{t['ok']}/{t['total']}" if t["total"] else "-"
        if metric == "sp":
            return _fmt(sum(t["speedups"]) / len(t["speedups"])) if t["speedups"] else "-"
        if metric == "fr":
            return _fmt(sum(t["first_rounds"]) / len(t["first_rounds"]), "{:.1f}") if t["first_rounds"] else "-"
        if metric == "skill":
            return f"{t.get('skill_hits', 0)}/{t['total']}" if t["total"] else "-"
        return "-"

    sec4_start = content.find("## 4. 汇总统计")
    sec5_start = content.find("## 5.", sec4_start + 1) if sec4_start >= 0 else -1
    if sec4_start < 0:
        return content
    sec4_end = sec5_start if sec5_start > 0 else len(content)

    sec4_text = content[sec4_start:sec4_end]

    def _build_row(metric_name: str, metric_key: str) -> None:
        nonlocal sec4_text
        row_pat = re.compile(r"\| *" + re.escape(metric_name) + r" *\|[^\n]*")
        vals = []
        for g in header_groups:
            for ab in ["A", "B"]:
                k = f"Group {g} {ab}"
                if metric_key == "skill" and ab == "A":
                    vals.append("N/A")
                else:
                    vals.append(_col(k, metric_key))
        vals.append(_total("A", metric_key) if metric_key != "skill" else "N/A")
        vals.append(_total("B", metric_key))
        new_row = f"| {metric_name} | " + " | ".join(vals) + " |"
        sec4_text = row_pat.sub(new_row, sec4_text)

    _build_row("成功率", "sr")
    _build_row("平均首次成功轮次", "fr")
    _build_row("平均 Speedup", "sp")
    _build_row("平均生成时间", "time")
    _build_row("Skill 命中率", "skill")

    content = content[:sec4_start] + sec4_text + content[sec4_end:]
    return content


# ============================================================================
# 主入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Skill Evolution A/B 测试运行器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--group", type=int, nargs="+", required=True, choices=[1, 2, 3, 4],
        help="要测试的 group 编号（可多选）",
    )
    parser.add_argument(
        "--mode", choices=["A", "B", "both"], default="both",
        help="A=baseline, B=evolved skill, both=先 B 后 A（B 无 skill 命中则跳过 A）",
    )
    parser.add_argument(
        "--device", type=int, default=0,
        help="NPU/GPU 设备 ID（默认 0）",
    )
    parser.add_argument(
        "--max-rounds", type=int, default=3,
        help="evolve 迭代轮数（默认 3）",
    )
    parser.add_argument(
        "--output-dir", default="~/akg_eval_results",
        help="输出根目录（默认 ~/akg_eval_results）",
    )
    parser.add_argument(
        "--evolved-skill-dir",
        default="python/akg_agents/op/resources/skills/triton-cuda/evolved",
        help="evolved SKILL.md 所在目录（B 组使用）",
    )
    parser.add_argument(
        "--agent-config",
        default="op/config/triton_cuda_evolve_config.yaml",
        help="agent 模型配置文件（相对于 python/akg_agents/）",
    )
    parser.add_argument(
        "--log-dir", default="~/akg_agents_logs",
        help="agent 日志根目录（默认 ~/akg_agents_logs）",
    )
    parser.add_argument(
        "--tracking-md", default=None,
        help="tracking.md 路径（默认自动检测）",
    )
    parser.add_argument(
        "--run-dir", default=None,
        help="指定已有的运行目录（用于 --collect-only）",
    )
    parser.add_argument(
        "--collect-only", action="store_true",
        help="仅收集已有结果（必须配合 --run-dir 使用）",
    )
    args = parser.parse_args()

    if args.collect_only and not args.run_dir:
        parser.error("--collect-only 必须配合 --run-dir 使用")

    project_root = get_project_root()

    tracking_path = args.tracking_md or str(
        project_root / "benchmark" / "akg_kernels_bench" / "thirdparty" / "pytorch" / "tracking.md"
    )

    # 确定 run_dir
    if args.run_dir:
        run_dir = os.path.expanduser(args.run_dir)
    else:
        run_dir = generate_run_dir(args.output_dir)

    # 读取或生成 run_config
    run_config_path = os.path.join(run_dir, "run_config.json")
    if os.path.isfile(run_config_path):
        with open(run_config_path, encoding="utf-8") as f:
            run_config = json.load(f)
    else:
        save_run_config(run_dir, args)
        with open(run_config_path, encoding="utf-8") as f:
            run_config = json.load(f)

    modes = ["A", "B"] if args.mode == "both" else [args.mode]

    print("=" * 80)
    print("Skill Evolution A/B 测试")
    print("=" * 80)
    print(f"Run Dir: {run_dir}")
    print(f"Groups:  {args.group}")
    print(f"Modes:   {modes}")
    print(f"Rounds:  {args.max_rounds}")
    print(f"Device:  {args.device}")
    print(f"DSL:     {run_config.get('dsl', '')} / {run_config.get('backend', '')} / {run_config.get('arch', '')}")
    if "B" in modes:
        print(f"Evolved: {args.evolved_skill_dir}")
    print("=" * 80)

    # ================================================================
    # Phase 1: 运行测试
    # ================================================================
    if not args.collect_only:
        for group in args.group:
            if args.mode == "both":
                # both 模式: B 先跑，根据 B 的 skill 命中情况过滤 A 的算子
                run_group(group, "B", run_dir, args, project_root)

                matched_files = _find_operators_with_skills(run_dir, group)
                if matched_files:
                    print(f"\n  B 组中 {len(matched_files)} 个算子有 skill 命中，将为这些算子运行 A 组")
                    filtered_dir = _create_filtered_task_dir(project_root, group, matched_files)
                    try:
                        run_group(group, "A", run_dir, args, project_root, task_dir=filtered_dir)
                    finally:
                        if filtered_dir:
                            shutil.rmtree(filtered_dir, ignore_errors=True)
                else:
                    print(f"\n  B 组中无算子命中 evolved skill，跳过 Group {group} A 组")
            else:
                for mode in modes:
                    run_group(group, mode, run_dir, args, project_root)

    # ================================================================
    # Phase 2: 收集结果
    # ================================================================
    print(f"\n{'='*80}")
    print("收集实验结果...")
    print(f"{'='*80}")

    all_results: List[Dict[str, Any]] = []
    collect_modes = ["B", "A"] if args.mode == "both" else modes
    for group in args.group:
        for mode in collect_modes:
            group_results = collect_group_results(
                group=group,
                ab_mode=mode,
                run_dir=run_dir,
                log_dir_base=args.log_dir,
                max_rounds=args.max_rounds,
            )
            all_results.extend(group_results)
            for r in group_results:
                sp = _fmt(r.get("best_speedup"))
                status = "OK" if r["success"] else "FAIL"
                skills_str = f" skills={r['selected_skills']}" if r["selected_skills"] else ""
                print(f"  Group {group} {mode} | {r['op_name']:25s} | {status:4s} | speedup={sp}{skills_str}")

    # 保存汇总 JSON
    for group in args.group:
        gr = [r for r in all_results if r["group"] == group]
        if gr:
            detail_path = os.path.join(run_dir, f"ab_detail_group{group}.json")
            with open(detail_path, "w", encoding="utf-8") as f:
                json.dump(gr, f, indent=2, ensure_ascii=False, default=str)
            print(f"  Group {group} 详细结果: {detail_path}")

    # ================================================================
    # Phase 3: 更新 tracking.md
    # ================================================================
    if all_results:
        print(f"\n{'='*80}")
        print("更新 tracking.md ...")
        print(f"{'='*80}")
        update_tracking_md(all_results, tracking_path, run_config)

    # ================================================================
    # Phase 4: 汇总
    # ================================================================
    print(f"\n{'='*80}")
    print("测试完成汇总")
    print(f"{'='*80}")
    for group in args.group:
        for mode in collect_modes:
            gr = [r for r in all_results if r["group"] == group and r["ab_mode"] == mode]
            if not gr:
                continue
            n_ok = sum(1 for r in gr if r["success"])
            speedups = [r["best_speedup"] for r in gr if r.get("best_speedup") is not None]
            avg_sp = sum(speedups) / len(speedups) if speedups else 0
            retries = sum(r.get("total_conductor_retries", 0) for r in gr)
            print(f"  Group {group} {mode}: {n_ok}/{len(gr)} 成功, 平均speedup={avg_sp:.2f}x, conductor重试={retries}")

    print(f"\n运行目录: {run_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
