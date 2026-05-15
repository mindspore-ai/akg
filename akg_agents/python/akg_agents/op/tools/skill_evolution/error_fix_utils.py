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
Skill 自进化系统 - 错误修复经验提取 (error_fix 模式)

从搜索日志目录中提取"失败→成功"的修复记录，包括：
  - 失败代码 vs 成功代码的 diff（完整，不截断）
  - 最后一条错误日志

只关注成功修复的案例，每个 Task 只记录一对
"成功前最后一版失败代码 → 成功代码"。
"""

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .common import code_diff

logger = logging.getLogger(__name__)

MAX_ERROR_LOG_CHARS = 1000


@dataclass
class SuccessfulFixRecord:
    """成功修复记录（Task 级别）"""
    task_id: str
    op_name: str
    error_log: str = ""
    error_step: int = 0
    failed_code: str = ""
    success_code: str = ""
    diff: str = ""
    dsl: str = ""
    backend: str = ""
    arch: str = ""


# ==================== 公开接口 ====================


def collect(
    log_dir: str, op_name: str,
) -> Tuple[List[SuccessfulFixRecord], Dict[str, str]]:
    """从 logs 目录收集成功修复记录

    解析 verification_results.jsonl，对每个有 passed=true 的 task：
    找到成功前最后一个失败验证，提取失败/成功代码、错误日志和完整 diff。

    Returns:
        (records, metadata)
    """
    jsonl_path = os.path.join(log_dir, "verification_results.jsonl")
    if not os.path.isfile(jsonl_path):
        raise FileNotFoundError(f"verification_results.jsonl 不存在: {jsonl_path}")

    all_entries = _parse_verification_jsonl(jsonl_path)

    metadata: Dict[str, str] = {
        "op_name": op_name, "dsl": "", "backend": "", "arch": "",
    }

    tasks: Dict[str, List[Dict]] = {}
    for entry in all_entries:
        tid = entry.get("task_id", "")
        if not tid:
            continue
        tasks.setdefault(tid, []).append(entry)
        if not metadata["dsl"] and entry.get("dsl"):
            metadata["dsl"] = entry["dsl"]
        if not metadata["backend"] and entry.get("backend"):
            metadata["backend"] = entry["backend"]
        if not metadata["arch"] and entry.get("arch"):
            metadata["arch"] = entry["arch"]

    records: List[SuccessfulFixRecord] = []

    for tid, entries in tasks.items():
        entries.sort(key=lambda e: e.get("step", 0))
        record = _extract_fix_record(
            tid, entries, op_name, log_dir, metadata,
        )
        if record:
            records.append(record)

    logger.info(
        f"[ErrorFix:Collect] {len(tasks)} 个 Task, "
        f"提取到 {len(records)} 个成功修复记录"
    )
    return records, metadata


def to_prompt_vars(
    records: List[SuccessfulFixRecord],
    metadata: Dict[str, str],
) -> Dict[str, Any]:
    """构建 analyze_error_fix.j2 模板变量

    注意：
      - conductor 建议不注入 prompt（避免误导 LLM，diff 已足够）
      - diff 使用完整版本，不截断
    """
    fix_cases: List[str] = []
    for i, r in enumerate(records, 1):
        parts = [
            f"### 修复案例 {i}: {r.task_id}",
            "",
            f"**错误 Step**: {r.error_step}",
            f"**错误日志** (截取):",
            f"```",
            f"{r.error_log}",
            f"```",
            "",
            f"**代码 Diff** (失败→成功):",
            f"```diff",
            f"{r.diff}",
            f"```",
        ]
        fix_cases.append("\n".join(parts))

    return {
        "op_name": metadata.get("op_name", ""),
        "dsl": metadata.get("dsl", ""),
        "backend": metadata.get("backend", ""),
        "arch": metadata.get("arch", ""),
        "fix_count": len(records),
        "fix_cases": "\n\n".join(fix_cases),
    }


# ==================== 数据解析 ====================


def _parse_verification_jsonl(path: str) -> List[Dict]:
    """解析 verification_results.jsonl（每行一个独立 JSON 对象）"""
    entries: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    for raw in re.split(r'\n\s*(?=\{)', content.strip()):
        raw = raw.strip()
        if not raw:
            continue
        try:
            entries.append(json.loads(raw))
        except json.JSONDecodeError:
            pass
    return entries


def _extract_fix_record(
    task_id: str,
    entries: List[Dict],
    op_name: str,
    log_dir: str,
    metadata: Dict[str, str],
) -> Optional[SuccessfulFixRecord]:
    """对单个 Task 的验证记录列表，提取修复记录"""
    first_success = None
    last_failure_before_success = None

    for entry in entries:
        if entry.get("passed"):
            first_success = entry
            break
        last_failure_before_success = entry

    if not first_success or not last_failure_before_success:
        return None

    fail_step = last_failure_before_success.get("step", 0)
    success_step = first_success.get("step", 0)

    fail_code = _read_impl_from_entry(last_failure_before_success, log_dir)
    success_code = _read_impl_from_entry(first_success, log_dir)

    if not fail_code and not success_code:
        logger.debug(f"[ErrorFix] {task_id}: 无法读取代码，跳过")
        return None

    error_log = last_failure_before_success.get("error_log", "")
    if len(error_log) > MAX_ERROR_LOG_CHARS:
        error_log = error_log[-MAX_ERROR_LOG_CHARS:]

    diff = code_diff(
        fail_code, success_code,
        f"{task_id}_step{fail_step}", f"{task_id}_step{success_step}",
        truncate=False,
    )

    return SuccessfulFixRecord(
        task_id=task_id,
        op_name=op_name,
        error_log=error_log,
        error_step=fail_step,
        failed_code=fail_code,
        success_code=success_code,
        diff=diff,
        dsl=metadata.get("dsl", ""),
        backend=metadata.get("backend", ""),
        arch=metadata.get("arch", ""),
    )


# ==================== 文件读取 ====================


def _read_impl_from_entry(entry: Dict, log_dir: str) -> str:
    """从 verification entry 的 verify_dir 读取 *_impl.py"""
    verify_dir = entry.get("verify_dir", "")
    if not verify_dir:
        return ""
    resolved = _resolve_verify_dir(verify_dir, log_dir)
    return _read_impl_file(resolved)


def _resolve_verify_dir(verify_dir: str, log_dir: str) -> str:
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
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except OSError:
        return ""
