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
Skill 自进化系统 - 数据压缩器

核心：
- 选取 gen_time 最小的方案作为最佳实现
- 对每条进化路径维护单调栈，注释剥离后生成相邻 diff
"""

import difflib
import logging
import re
from collections import defaultdict
from typing import Dict, List, Set, Tuple

from .models import CompressedData, EvolutionStep, TaskRecord

logger = logging.getLogger(__name__)

DIFF_MAX_LINES = 60
MIN_GEN_TIME_IMPROVE_PCT = 0.01  # gen_time 提升 ≥ 1% 才生成 diff


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
        f"[Compressor] 最佳={result.best_task_id} "
        f"({result.best_gen_time}us/{result.best_speedup:.2f}x), "
        f"进化链={len(evolution_chains)} 步"
    )
    return result


# ==================== 进化链 ====================


def _build_evolution_chains(
    records: List[TaskRecord],
    record_map: Dict[str, TaskRecord],
) -> List[EvolutionStep]:
    """建树 → DFS 收集路径 → 每条路径单调栈 → 注释剥离 diff"""
    if not records:
        return []

    # 1. 建树
    children: Dict[str, List[str]] = defaultdict(list)
    all_ids = {r.task_id for r in records}
    for r in records:
        if r.parent_id and r.parent_id in record_map:
            children[r.parent_id].append(r.task_id)

    roots = [r.task_id for r in records
             if not r.parent_id or r.parent_id not in record_map]

    logger.info(f"[Compressor] 进化树: {len(records)} 节点, {len(roots)} 根")

    # 2. DFS 收集根→叶路径
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

    # 3. 每条路径 → 单调栈 → diff
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

        _log_mono(path, mono, record_map)

        path_steps_added = 0
        for i in range(len(mono) - 1):
            pid, cid = mono[i], mono[i + 1]
            if (pid, cid) in seen:
                continue
            seen.add((pid, cid))

            parent, child = record_map[pid], record_map[cid]
            pct = _gen_time_improve_pct(parent, child)

            if pct < MIN_GEN_TIME_IMPROVE_PCT:
                logger.info(
                    f"[Compressor]   {pid}→{cid}: 跳过 "
                    f"gen_time 提升={pct:.2%}"
                )
                continue

            step = _try_make_step(parent, child)
            if step:
                steps.append(step)
                path_steps_added += 1

        # 路径级回退：逐步 diff 全被过滤时，生成首→尾总 diff
        if path_steps_added == 0 and len(mono) >= 2:
            head, tail = record_map[mono[0]], record_map[mono[-1]]
            pct = _gen_time_improve_pct(head, tail)
            pair = (mono[0], mono[-1])
            if pct >= MIN_GEN_TIME_IMPROVE_PCT and pair not in seen:
                seen.add(pair)
                logger.info(
                    f"[Compressor]   路径回退 {mono[0]}→{mono[-1]}: "
                    f"gen_time 总提升={pct:.2%}"
                )
                step = _try_make_step(head, tail)
                if step:
                    steps.append(step)

    steps.sort(key=lambda s: _gen_time_improve_pct_raw(
        s.parent_gen_time, s.child_gen_time), reverse=True)
    logger.info(f"[Compressor] 进化链: {len(steps)} 步")
    return steps


def _gen_time_improve_pct(parent: TaskRecord, child: TaskRecord) -> float:
    """计算 gen_time 提升百分比：(parent - child) / parent"""
    return _gen_time_improve_pct_raw(parent.gen_time, child.gen_time)


def _gen_time_improve_pct_raw(parent_gt: float, child_gt: float) -> float:
    if parent_gt <= 0 or parent_gt == float("inf"):
        return 0.0
    return (parent_gt - child_gt) / parent_gt


def _try_make_step(parent: TaskRecord, child: TaskRecord):
    """尝试生成一个 EvolutionStep，无代码或代码相同则返回 None"""
    pid, cid = parent.task_id, child.task_id
    if not parent.code or not child.code:
        logger.info(
            f"[Compressor]   {pid}→{cid}: 跳过 无代码 "
            f"({len(parent.code)}/{len(child.code)})"
        )
        return None

    diff = _code_diff(parent.code, child.code, pid, cid)
    if not diff or diff == "(代码相同)":
        logger.info(f"[Compressor]   {pid}→{cid}: 跳过 代码相同")
        return None

    logger.info(f"[Compressor]   {pid}→{cid}: diff {len(diff)} 字符")
    return EvolutionStep(
        parent_id=pid,
        child_id=cid,
        parent_speedup=parent.speedup,
        child_speedup=child.speedup,
        parent_gen_time=parent.gen_time,
        child_gen_time=child.gen_time,
        code_diff=diff,
    )


def _log_mono(
    path: List[str], mono: List[str], rm: Dict[str, TaskRecord]
) -> None:
    """打印单调栈过滤结果"""
    def _fmt(nid: str) -> str:
        r = rm.get(nid)
        if not r:
            return nid
        gt = f"{r.gen_time:.1f}us" if r.gen_time < float("inf") else "∞"
        return f"{nid}({gt}/{r.speedup:.2f}x)"

    logger.info(
        f"[Compressor] path: {' → '.join(_fmt(n) for n in path)} "
        f"| 单调栈({len(path)}→{len(mono)}): "
        f"{' → '.join(_fmt(n) for n in mono)}"
    )


# ==================== 性能排序 / 比较 ====================


def _perf_sort_key(r: TaskRecord) -> Tuple[float, float]:
    """gen_time 升序, speedup 降序"""
    return (r.gen_time, -r.speedup)



# ==================== 注释剥离 + diff ====================


def _strip_comments(code: str) -> str:
    """移除注释和 docstring，用于干净 diff"""
    if not code:
        return ""
    code = re.sub(r'"""[\s\S]*?"""', '', code)
    code = re.sub(r"'''[\s\S]*?'''", '', code)

    cleaned: List[str] = []
    for line in code.splitlines():
        if line.strip().startswith("#"):
            continue
        result: List[str] = []
        in_sq, in_dq = False, False
        i = 0
        while i < len(line):
            ch = line[i]
            if ch == '\\' and i + 1 < len(line):
                result.append(ch)
                result.append(line[i + 1])
                i += 2
                continue
            if ch == '"' and not in_sq:
                in_dq = not in_dq
            elif ch == "'" and not in_dq:
                in_sq = not in_sq
            elif ch == '#' and not in_sq and not in_dq:
                break
            result.append(ch)
            i += 1
        cleaned.append("".join(result).rstrip())

    merged: List[str] = []
    prev_blank = False
    for line in cleaned:
        blank = not line.strip()
        if blank and prev_blank:
            continue
        prev_blank = blank
        merged.append(line)

    return "\n".join(merged).strip() + "\n" if merged else ""


def _code_diff(
    base: str, target: str, base_label: str, target_label: str,
) -> str:
    """注释剥离后生成 unified diff"""
    base = _strip_comments(base)
    target = _strip_comments(target)

    diff_lines = list(difflib.unified_diff(
        base.splitlines(keepends=True),
        target.splitlines(keepends=True),
        fromfile=f"a/{base_label}",
        tofile=f"b/{target_label}",
        n=1,
    ))
    if not diff_lines:
        return "(代码相同)"

    if len(diff_lines) > DIFF_MAX_LINES:
        total = len(diff_lines)
        diff_lines = diff_lines[:DIFF_MAX_LINES]
        diff_lines.append(f"... (截断，共 {total} 行)\n")

    return "".join(diff_lines).rstrip()


# ==================== 性能摘要 ====================


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
