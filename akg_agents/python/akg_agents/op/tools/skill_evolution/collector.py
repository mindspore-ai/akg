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
Skill 自进化系统 - 数据收集器

从 3 个文件收集数据：
- verification_results.jsonl  → task_id, passed, verify_dir, dsl/backend/arch
- speed_up_record.txt         → task_id, gen_time, speedup
- {op_name}_lineage_graph.md  → task_id, parent_id, generation

对 passed 的任务，从 verify_dir 读取 *_impl.py 作为代码。
"""

import json
import logging
import os
import re
from typing import Dict, List, Tuple

from .models import TaskRecord

logger = logging.getLogger(__name__)


def collect(log_dir: str, op_name: str) -> Tuple[List[TaskRecord], Dict]:
    """收集进化数据

    Args:
        log_dir: 节点 logs 目录（如 .../node_004/logs）
        op_name: 算子名称（如 relu）

    Returns:
        (records, metadata) 其中 metadata 包含 dsl/backend/arch
    """
    logger.info(f"[Collector] 开始: log_dir={log_dir}, op_name={op_name}")

    # 1. verification_results.jsonl → passed 任务 + verify_dir + 环境信息
    vr_path = os.path.join(log_dir, "verification_results.jsonl")
    vr_map, metadata = _parse_verification_results(vr_path)
    logger.info(
        f"[Collector] verification_results: {len(vr_map)} 个 passed 任务"
    )

    # 2. speed_up_record.txt → gen_time + speedup
    sr_path = os.path.join(log_dir, op_name, "profiling", "speed_up_record.txt")
    perf_map = _parse_speedup_record(sr_path)
    logger.info(f"[Collector] speed_up_record: {len(perf_map)} 条记录")

    # 3. lineage_graph.md → parent_id + generation
    lg_path = os.path.join(log_dir, f"{op_name}_lineage_graph.md")
    lineage_map = _parse_lineage_table(lg_path)
    logger.info(f"[Collector] lineage_graph: {len(lineage_map)} 条记录")

    # 4. 合并：以 lineage_map 为全集（它包含所有任务）
    all_task_ids = set(lineage_map.keys()) | set(vr_map.keys()) | set(perf_map.keys())
    records: List[TaskRecord] = []

    for task_id in sorted(all_task_ids):
        lineage = lineage_map.get(task_id, {})
        perf = perf_map.get(task_id, {})
        vr = vr_map.get(task_id)

        # 只保留有性能数据的任务（即成功的）
        speedup = perf.get("speedup", lineage.get("speedup", 0.0))
        gen_time = perf.get("gen_time", lineage.get("gen_time", float("inf")))
        if speedup <= 0 and not vr:
            continue

        # 从 verify_dir 读取代码
        code = ""
        if vr and vr.get("verify_dir"):
            code = _read_impl_file(vr["verify_dir"])

        record = TaskRecord(
            task_id=task_id,
            parent_id=lineage.get("parent_id", ""),
            generation=lineage.get("generation", 0),
            code=code,
            speedup=speedup,
            gen_time=gen_time,
        )
        records.append(record)

        gt_str = f"{gen_time:.2f}us" if gen_time < float("inf") else "∞"
        logger.info(
            f"[Collector]   {task_id} (gen={record.generation}, "
            f"parent={record.parent_id or '-'}, {gt_str}/{speedup:.2f}x, "
            f"code={'Y' if code else 'N'}({len(code)}))"
        )

    logger.info(f"[Collector] 共 {len(records)} 条记录")
    return records, metadata


# ==================== 解析函数 ====================


def _parse_verification_results(path: str) -> Tuple[Dict[str, Dict], Dict]:
    """解析 verification_results.jsonl

    返回 (vr_map, metadata):
    - vr_map: task_id → {verify_dir} （仅 passed=True 的，取最后一条）
    - metadata: {dsl, backend, arch}
    """
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
    """解析 speed_up_record.txt → {task_id: {speedup, gen_time}}"""
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
    """解析 lineage_graph.md 中的表格 → {task_id: {parent_id, generation, speedup, gen_time}}"""
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

        # 分隔行 / 表头行 → 标记进入表格
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


def _read_impl_file(verify_dir: str) -> str:
    """从 verify 目录读取 *_impl.py 文件"""
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
    """读取文本文件，失败返回空字符串"""
    try:
        with open(os.path.expanduser(path), "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""
