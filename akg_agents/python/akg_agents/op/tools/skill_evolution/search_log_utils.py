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
Skill 自进化系统 - 搜索日志分析 (search_log 模式)

从 adaptive_search 产生的 3 个日志文件收集数据，经单调栈压缩后生成进化链 diff。

数据源：
  - verification_results.jsonl  → task_id, passed, verify_dir, dsl/backend/arch
  - speed_up_record.txt         → task_id, gen_time, speedup
  - {op_name}_lineage_graph.md  → task_id, parent_id, generation
  - verify_dir/*_impl.py        → 代码
"""

import json
import logging
import os
import re
from collections import defaultdict
from dataclasses import asdict
from typing import Any, Dict, List, Set, Tuple

from .common import (
    CompressedData, EvolutionStep, TaskRecord,
    code_diff,
)

logger = logging.getLogger(__name__)

MIN_GEN_TIME_IMPROVE_PCT = 0.01


# ==================== 收集 ====================


def collect(log_dir: str, op_name: str) -> Tuple[List[TaskRecord], Dict]:
    """收集进化数据

    Args:
        log_dir: 节点 logs 目录（如 .../node_004/logs）
        op_name: 算子名称（如 relu）

    Returns:
        (records, metadata) 其中 metadata 包含 dsl/backend/arch
    """
    logger.info(f"[SearchLog:Collect] 开始: log_dir={log_dir}, op_name={op_name}")

    vr_path = os.path.join(log_dir, "verification_results.jsonl")
    vr_map, metadata = _parse_verification_results(vr_path)
    logger.info(f"[SearchLog:Collect] verification_results: {len(vr_map)} 个 passed 任务")
    if not vr_map and os.path.isfile(vr_path):
        logger.warning("[SearchLog:Collect] verification_results 文件存在但未解析到 passed 任务")

    sr_path = os.path.join(log_dir, op_name, "profiling", "speed_up_record.txt")
    perf_map = _parse_speedup_record(sr_path)
    logger.info(f"[SearchLog:Collect] speed_up_record: {len(perf_map)} 条记录")
    if not perf_map and os.path.isfile(sr_path):
        logger.warning("[SearchLog:Collect] speed_up_record 文件存在但未解析到性能数据")

    lg_path = os.path.join(log_dir, f"{op_name}_lineage_graph.md")
    lineage_map = _parse_lineage_table(lg_path)
    logger.info(f"[SearchLog:Collect] lineage_graph: {len(lineage_map)} 条记录")
    if not lineage_map and os.path.isfile(lg_path):
        logger.warning("[SearchLog:Collect] lineage_graph 文件存在但未解析到谱系数据")

    all_task_ids = set(lineage_map.keys()) | set(vr_map.keys()) | set(perf_map.keys())
    records: List[TaskRecord] = []

    for task_id in sorted(all_task_ids):
        lineage = lineage_map.get(task_id, {})
        perf = perf_map.get(task_id, {})
        vr = vr_map.get(task_id)

        speedup = perf.get("speedup", lineage.get("speedup", 0.0))
        gen_time = perf.get("gen_time", lineage.get("gen_time", float("inf")))
        if speedup <= 0 and not vr:
            continue

        code = ""
        if vr and vr.get("verify_dir"):
            resolved = _resolve_verify_dir(vr["verify_dir"], log_dir)
            code = _read_impl_file(resolved)

        record = TaskRecord(
            task_id=task_id,
            parent_id=lineage.get("parent_id", ""),
            generation=lineage.get("generation", 0),
            code=code,
            speedup=speedup,
            gen_time=gen_time,
        )
        records.append(record)

    logger.info(f"[SearchLog:Collect] 共 {len(records)} 条记录")
    return records, metadata


# ==================== 压缩 ====================


def compress(
    records: List[TaskRecord],
    metadata: Dict,
) -> CompressedData:
    """压缩 records → CompressedData"""
    record_map = {r.task_id: r for r in records}

    best = min(records, key=_perf_sort_key) if records else None
    evolution_chains = _build_evolution_chains(records, record_map)
    summary = _build_performance_summary(records)

    result = CompressedData(
        op_name=metadata.get("op_name", ""),
        dsl=metadata.get("dsl", ""),
        backend=metadata.get("backend", ""),
        arch=metadata.get("arch", ""),
        best_task_id=best.task_id if best else "",
        best_speedup=best.speedup if best else 0.0,
        best_gen_time=best.gen_time if best else float("inf"),
        best_code=best.code if best else "",
        evolution_chains=evolution_chains,
        performance_summary=summary,
        total_tasks=len(records),
        success_count=len(records),
    )

    logger.info(
        f"[SearchLog:Compress] 最佳={result.best_task_id} "
        f"({result.best_gen_time}us/{result.best_speedup:.2f}x), "
        f"进化链={len(evolution_chains)} 步"
    )
    return result


# ==================== 进化链构建 ====================


def _build_evolution_chains(
    records: List[TaskRecord],
    record_map: Dict[str, TaskRecord],
) -> List[EvolutionStep]:
    """建树 → DFS 收集路径 → 每条路径单调栈 → 注释剥离 diff"""
    if not records:
        return []

    children: Dict[str, List[str]] = defaultdict(list)
    all_ids = {r.task_id for r in records}
    for r in records:
        if r.parent_id and r.parent_id in record_map:
            children[r.parent_id].append(r.task_id)

    roots = [r.task_id for r in records
             if not r.parent_id or r.parent_id not in record_map]

    all_paths: List[List[str]] = []

    def _dfs(nid: str, path: List[str]) -> None:
        path.append(nid)
        kids = [c for c in children.get(nid, []) if c in all_ids]
        if not kids:
            all_paths.append(list(path))
        else:
            for kid in kids:
                _dfs(kid, path)
        path.pop()

    for root in roots:
        _dfs(root, [])

    seen: Set[Tuple[str, str]] = set()
    steps: List[EvolutionStep] = []

    for path in all_paths:
        mono: List[str] = []
        for nid in path:
            rec = record_map.get(nid)
            if not rec:
                continue
            if not mono:
                mono.append(nid)
                continue
            if rec.gen_time < record_map[mono[-1]].gen_time:
                mono.append(nid)

        path_steps_added = 0
        for i in range(len(mono) - 1):
            pid, cid = mono[i], mono[i + 1]
            if (pid, cid) in seen:
                continue
            seen.add((pid, cid))

            parent, child = record_map[pid], record_map[cid]
            pct = _gen_time_improve_pct(parent, child)

            if pct < MIN_GEN_TIME_IMPROVE_PCT:
                continue

            step = _try_make_step(parent, child)
            if step:
                steps.append(step)
                path_steps_added += 1

        if path_steps_added == 0 and len(mono) >= 2:
            head, tail = record_map[mono[0]], record_map[mono[-1]]
            pct = _gen_time_improve_pct(head, tail)
            pair = (mono[0], mono[-1])
            if pct >= MIN_GEN_TIME_IMPROVE_PCT and pair not in seen:
                seen.add(pair)
                step = _try_make_step(head, tail)
                if step:
                    steps.append(step)

    steps.sort(key=lambda s: _gen_time_improve_pct_raw(
        s.parent_gen_time, s.child_gen_time), reverse=True)
    return steps


def _gen_time_improve_pct(parent: TaskRecord, child: TaskRecord) -> float:
    return _gen_time_improve_pct_raw(parent.gen_time, child.gen_time)


def _gen_time_improve_pct_raw(parent_gt: float, child_gt: float) -> float:
    if parent_gt <= 0 or parent_gt == float("inf"):
        return 0.0
    return (parent_gt - child_gt) / parent_gt


def _try_make_step(parent: TaskRecord, child: TaskRecord):
    pid, cid = parent.task_id, child.task_id
    if not parent.code or not child.code:
        return None

    diff = code_diff(parent.code, child.code, pid, cid)
    if not diff or diff == "(代码相同)":
        return None

    return EvolutionStep(
        parent_id=pid,
        child_id=cid,
        parent_speedup=parent.speedup,
        child_speedup=child.speedup,
        parent_gen_time=parent.gen_time,
        child_gen_time=child.gen_time,
        code_diff=diff,
    )


def _perf_sort_key(r: TaskRecord) -> Tuple[float, float]:
    return (r.gen_time, -r.speedup)


def _build_performance_summary(records: List[TaskRecord]) -> str:
    if not records:
        return ""
    speedups = [r.speedup for r in records]
    gen_times = [r.gen_time for r in records if r.gen_time < float("inf")]
    lines = [
        f"总任务数: {len(records)}",
        f"最佳加速比: {max(speedups):.2f}x",
    ]
    if len(speedups) > 1:
        lines.append(f"加速比范围: {min(speedups):.2f}x ~ {max(speedups):.2f}x")
    if gen_times:
        lines.append(f"最佳执行时间: {min(gen_times):.2f} us")
    return "\n".join(lines)


# ==================== 日志文件解析 ====================


def _parse_verification_results(path: str) -> Tuple[Dict[str, Dict], Dict]:
    vr_map: Dict[str, Dict] = {}
    metadata: Dict = {"dsl": "", "backend": "", "arch": ""}

    content = _read_text(path)
    if not content:
        return vr_map, metadata

    for block in re.split(r"\n\s*\n", content.strip()):
        block = block.strip()
        if not block:
            continue
        try:
            entry = json.loads(block)
        except json.JSONDecodeError:
            continue

        if not metadata["dsl"] and entry.get("dsl"):
            metadata["dsl"] = entry["dsl"]
            metadata["backend"] = entry.get("backend", "")
            metadata["arch"] = entry.get("arch", "")

        tid = entry.get("task_id", "")
        if tid and entry.get("passed"):
            vr_map[tid] = {"verify_dir": entry.get("verify_dir", "")}

    return vr_map, metadata


def _parse_speedup_record(path: str) -> Dict[str, Dict]:
    result: Dict[str, Dict] = {}
    content = _read_text(path)
    if not content:
        return result

    for line in content.strip().splitlines():
        parts = {}
        for segment in line.split(", "):
            if ": " in segment:
                key, val = segment.split(": ", 1)
                parts[key.strip()] = val.strip()

        tid = parts.get("task_id", "")
        if not tid:
            continue

        try:
            speedup = float(parts.get("speedup", "0").replace("x", ""))
        except ValueError:
            speedup = 0.0
        try:
            gen_time = float(parts.get("generation_time", "inf").replace(" us", ""))
        except ValueError:
            gen_time = float("inf")

        if tid not in result or gen_time < result[tid].get("gen_time", float("inf")):
            result[tid] = {"speedup": speedup, "gen_time": gen_time}

    return result


def _parse_lineage_table(path: str) -> Dict[str, Dict]:
    result: Dict[str, Dict] = {}
    content = _read_text(path)
    if not content:
        return result

    in_table = False
    for line in content.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            in_table = False
            continue

        cells = [c.strip() for c in line.split("|") if c.strip()]
        if not cells or len(cells) < 5:
            continue

        if "---" in cells[0] or cells[0] in ("任务ID", "任务 ID", "task_id", "Task"):
            in_table = True
            continue
        if not in_table:
            in_table = True
            continue

        task_id = cells[0]

        gen_str = cells[1]
        if "初始" in gen_str or gen_str == "0":
            generation = 0
        else:
            m = re.search(r"(\d+)", gen_str)
            generation = int(m.group(1)) if m else 0

        try:
            gen_time = float(cells[2].replace("us", "").strip())
        except (ValueError, TypeError):
            gen_time = float("inf")

        try:
            speedup = float(cells[3].replace("x", "").strip())
        except (ValueError, TypeError):
            speedup = 0.0

        parent_id = cells[4].strip()
        if parent_id == "-":
            parent_id = ""

        result[task_id] = {
            "parent_id": parent_id,
            "generation": generation,
            "gen_time": gen_time,
            "speedup": speedup,
        }

    return result


def _resolve_verify_dir(verify_dir: str, log_dir: str) -> str:
    """处理跨平台路径差异：若 verify_dir 不存在，尝试从 log_dir 拼接相对路径"""
    expanded = os.path.expanduser(verify_dir)
    if os.path.isdir(expanded):
        return expanded
    normalized = verify_dir.replace("\\", "/")
    marker = "/logs/"
    idx = normalized.rfind(marker)
    if idx >= 0:
        relative = normalized[idx + len(marker):]
        candidate = os.path.join(log_dir, relative)
        if os.path.isdir(candidate):
            return candidate
    return verify_dir


def _read_impl_file(verify_dir: str) -> str:
    verify_dir = os.path.expanduser(verify_dir)
    if not os.path.isdir(verify_dir):
        return ""
    try:
        for fname in os.listdir(verify_dir):
            if fname.endswith("_impl.py"):
                return _read_text(os.path.join(verify_dir, fname))
    except OSError:
        pass
    return ""


def _read_text(path: str) -> str:
    try:
        with open(os.path.expanduser(path), "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


# ==================== Prompt 变量构建 ====================


def to_prompt_vars(compressed: CompressedData) -> Dict[str, Any]:
    """将 CompressedData 转换为 analyze_search_log.j2 的 Jinja2 模板变量"""
    return {
        "op_name": compressed.op_name,
        "dsl": compressed.dsl,
        "backend": compressed.backend,
        "arch": compressed.arch,
        "task_desc": compressed.task_desc,
        "total_tasks": compressed.total_tasks,
        "success_count": compressed.success_count,
        "best_speedup": compressed.best_speedup,
        "best_task_id": compressed.best_task_id,
        "best_gen_time": compressed.best_gen_time,
        "best_code": compressed.best_code,
        "evolution_chains": [asdict(s) for s in compressed.evolution_chains],
        "performance_summary": compressed.performance_summary,
    }
