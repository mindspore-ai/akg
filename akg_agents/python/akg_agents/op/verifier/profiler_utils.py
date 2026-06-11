# Copyright 2025 Huawei Technologies Co., Ltd
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
Profiling utilities shared between KernelVerifier and LocalWorker.
Contains methods for running msprof, nsys, and analyzing profiling data.
"""

import os
import re
import json
import logging
import math
import subprocess
from typing import Tuple, Optional, Dict, Any, List
from pathlib import Path
from datetime import datetime
import pandas as pd
from akg_agents.utils.process_utils import run_command

logger = logging.getLogger(__name__)


# Canonical per-section schema returned by both the python-script profile
# path here and the msprof/nsys path in LocalWorker. ``per_case_us`` is the
# load-bearing field: a single-element list for static-shape ops keeps the
# downstream consumer iteration uniform with multi-shape ops. ``avg_us`` is
# the arithmetic mean (== per_case_us[0] for the static case) — present so
# legacy callers asking for "the aggregate timing" don't need to redo the
# sum themselves.
#
#   {
#     "avg_us": float,            # mean of per_case_us
#     "per_case_us": [float, ...],# length 1+ ; never empty when present
#     "method": str | None,       # timer name (e.g. "msprof", "loop_timer")
#   }
#
# `run_profile_scripts_and_collect_results` returns
#   {"base": Section | None, "gen": Section | None}
# where ``base is None`` covers the "skipped / no base script / measurement
# failed" cases uniformly. ``gen is None`` means the generation profile
# couldn't be measured (subprocess failed or JSON missing). Callers see one
# None-check, not two sentinel-value branches.


def make_profile_section(avg_us: float,
                         per_case_us: Optional[List[float]] = None,
                         method: Optional[str] = None) -> Dict[str, Any]:
    """Build a canonical profile section. Use this everywhere we synthesize
    a per-shape section from a single aggregate measurement (override
    baseline, msprof/nsys path, etc.) so the schema stays consistent."""
    if per_case_us is None or not per_case_us:
        per_case_us = [float(avg_us)]
    return {
        "avg_us": float(avg_us),
        "per_case_us": [float(t) for t in per_case_us],
        "method": method,
    }


def _finite(x: Any) -> Optional[float]:
    """Coerce to a finite float; None on inf/nan/non-numeric. Used to
    sanitize values read out of profile JSON before they propagate."""
    if isinstance(x, (int, float)) and math.isfinite(float(x)):
        return float(x)
    return None


def read_profile_result_from_json(verify_dir: str,
                                  json_filename: str) -> Optional[Dict[str, Any]]:
    """Read a profile-result JSON written by ``prof_{base,generation}_template_refactored.j2``.

    Returns a canonical section dict (see module docstring) or ``None`` when
    the file is absent / unparsable / inf-only. Templates emit
    ``per_case_us`` (always a list, length 1 for static-shape); we fall back
    to wrapping ``execution_time_us`` so older JSON written by the previous
    template revision still parses (transitional — drop once all task dirs
    have been re-profiled)."""
    json_path = os.path.join(verify_dir, json_filename)
    if not os.path.exists(json_path):
        logger.error(f"profile JSON not found: {json_path}")
        return None
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"profile JSON unreadable {json_filename}: {e}")
        return None

    avg = _finite(data.get("avg_time_us")) or _finite(data.get("execution_time_us"))
    if avg is None:
        return None

    raw_per_case = data.get("per_case_us")
    if isinstance(raw_per_case, list) and raw_per_case:
        per_case = [c for c in (_finite(t) for t in raw_per_case) if c is not None]
    else:
        per_case = []
    if not per_case:
        per_case = [avg]
    return {
        "avg_us": avg,
        "per_case_us": per_case,
        "method": data.get("method"),
    }


def run_profile_scripts_and_collect_results(
    verify_dir: str, op_name: str, task_id: str = "0",
    override_base_time_us: Optional[float] = None,
) -> Dict[str, Optional[Dict[str, Any]]]:
    """Run base + generation profile scripts and collect their canonical
    per-shape sections.

    Returns ``{"base": Section | None, "gen": Section | None}``. ``base``
    is ``None`` when the base script is absent (cross-backend / cached) and
    no override was provided, or when its measurement otherwise failed.
    ``gen`` is ``None`` when generation measurement failed (subprocess
    non-zero or JSON missing); callers treat that as an infra error.

    Thread-safe: uses ``cwd=`` instead of ``os.chdir()``.
    """
    base_section: Optional[Dict[str, Any]] = None

    # Cached / cross-backend path: caller already has the baseline. Wrap as
    # single-case section so downstream array consumers see the same shape.
    if (override_base_time_us is not None and override_base_time_us > 0
            and override_base_time_us < float("inf")):
        base_section = make_profile_section(
            override_base_time_us, method="override")
        logger.info(f"[{op_name}: {task_id}] 使用缓存的 baseline: "
                    f"{base_section['avg_us']:.2f} us")
    else:
        base_script = f"profile_{op_name}_base.py"
        base_script_path = os.path.join(verify_dir, base_script)
        if os.path.exists(base_script_path):
            base_result = run_command(
                ["python", base_script], cmd_msg="base_profile",
                timeout=600, cwd=verify_dir,
            )
            if not base_result[0]:
                logger.error(f"[{op_name}: {task_id}] 基准性能脚本执行失败: "
                             f"{base_result[1]}")
            else:
                base_section = read_profile_result_from_json(
                    verify_dir, "base_profile_result.json")
        else:
            logger.info(f"[{op_name}: {task_id}] 基准性能脚本不存在"
                        f"（使用缓存 baseline 或跨后端场景），跳过 base profile")

    gen_section: Optional[Dict[str, Any]] = None
    gen_script = f"profile_{op_name}_generation.py"
    gen_script_path = os.path.join(verify_dir, gen_script)
    if os.path.exists(gen_script_path):
        gen_result = run_command(
            ["python", gen_script], cmd_msg="generation_profile",
            timeout=600, cwd=verify_dir,
        )
        if not gen_result[0]:
            logger.error(f"[{op_name}: {task_id}] 生成代码性能脚本执行失败: "
                         f"{gen_result[1]}")
        else:
            gen_section = read_profile_result_from_json(
                verify_dir, "generation_profile_result.json")
    else:
        logger.info(f"[{op_name}: {task_id}] 生成代码性能脚本不存在，"
                    "跳过 generation profile")

    base_avg = base_section["avg_us"] if base_section else float("inf")
    gen_avg = gen_section["avg_us"] if gen_section else float("inf")
    logger.info(f"[{op_name}: {task_id}] Read profile results: "
                f"base={base_avg:.2f} us, gen={gen_avg:.2f} us "
                f"(base_cases={len(base_section['per_case_us']) if base_section else 0}, "
                f"gen_cases={len(gen_section['per_case_us']) if gen_section else 0})")
    return {"base": base_section, "gen": gen_section}


def run_msprof(script_path: str, op_name: str = "", task_id: str = "0", timeout: int = 600) -> Tuple[bool, str, Optional[str]]:
    """运行msprof性能分析
    
    Args:
        script_path: Python脚本路径
        op_name: 算子名称（用于日志）
        task_id: 任务ID（用于日志）
        timeout: 超时时间（秒）
        
    Returns:
        (success, error_msg, prof_path): 是否成功，错误信息，prof数据路径
    """
    try:
        process = subprocess.run(
            f'msprof --application="python {script_path}"',
            shell=True, capture_output=True, text=True, timeout=timeout
        )

        for line in process.stdout.split('\n'):
            if "[INFO] Process profiling data complete. Data is saved in" in line:
                match = re.search(r"Data is saved in (.+)$", line)
                if match:
                    return True, "", match.group(1).strip()

        return False, "未找到数据保存路径", None
    except Exception as e:
        logger.error(f"[{task_id}:{op_name}] msprof执行错误: {e}")
        return False, f"执行错误: {str(e)}", None


def analyze_prof_data(prof_path: str, warmup_times: int, run_times: int, op_name: str = "", task_id: str = "0") -> Tuple[bool, str, float]:
    """分析PROF数据
    
    Args:
        prof_path: prof数据目录路径
        warmup_times: 预热次数
        run_times: 实际运行次数
        op_name: 算子名称（用于日志）
        task_id: 任务ID（用于日志）
        
    Returns:
        (success, error_msg, avg_time_us): 是否成功，错误信息，平均时间（微秒）
    """
    try:
        csv_files = list(Path(prof_path).glob("mindstudio_profiler_output/op_summary_*.csv"))
        if not csv_files:
            return False, "未找到CSV文件", 0.0

        df = pd.read_csv(csv_files[0])

        # 移除特定的Op
        df_filtered = df[~df["Op Name"].str.contains("aclnnIsClose_IsCloseAiCpu_IsClose|aclnnAll_ReduceAll_ReduceAll",
                                                     regex=True, na=False)]

        total_count = warmup_times + run_times
        op_counts = df_filtered["Op Name"].value_counts()
        valid_ops = op_counts[op_counts == total_count]

        if len(valid_ops) == 0:
            return False, "没有找到符合预期次数的Op", float('inf')

        # 检查不匹配的Op
        invalid_ops = op_counts[op_counts != total_count]
        if len(invalid_ops) > 0:
            logger.warning(f"[{task_id}:{op_name}] 发现{len(invalid_ops)}个Op次数不匹配")

        # 计算平均时间
        df_valid = df_filtered[df_filtered["Op Name"].isin(valid_ops.index)]
        total_avg_time = 0.0

        for op_name_iter in valid_ops.index:
            op_data = df_valid[df_valid["Op Name"] == op_name_iter]["Task Duration(us)"].tolist()
            if len(op_data) > warmup_times:
                valid_data = op_data[warmup_times:]
                avg_time = sum(valid_data) / len(valid_data)
                total_avg_time += avg_time

        return True, "", total_avg_time

    except Exception as e:
        logger.error(f"[{task_id}:{op_name}] 分析prof数据时出错: {e}")
        return False, f"分析数据时出错: {str(e)}", float('inf')


def run_nsys(script_path: str, op_name: str = "", task_id: str = "0", timeout: int = 600) -> Tuple[bool, str, Optional[str]]:
    """运行nsys性能分析
    
    Args:
        script_path: Python脚本路径
        op_name: 算子名称（用于日志）
        task_id: 任务ID（用于日志）
        timeout: 超时时间（秒）
        
    Returns:
        (success, error_msg, rep_path): 是否成功，错误信息，nsys报告文件路径
    """
    try:
        output_name = "nsys_report_" + os.path.basename(script_path).replace(".py", "")
        cmd = f'nsys profile --output={output_name} python {script_path}'
        logger.debug(f"[{task_id}:{op_name}] Running nsys profile: {cmd}")
        process = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        report_path = os.path.join(os.path.dirname(script_path), output_name + ".nsys-rep")

        if os.path.exists(report_path):
            return True, "", report_path
        return False, "未找到nsys报告文件", None
    except Exception as e:
        logger.error(f"[{task_id}:{op_name}] nsys执行错误: {e}")
        return False, f"执行错误: {str(e)}", None


def analyze_nsys_data(rep_path: str, warmup_times: int, run_times: int, profile_type: str = "", op_name: str = "", task_id: str = "0") -> Tuple[bool, str, float]:
    """分析nsys生成的rep文件，返回平均耗时(us)
    
    Args:
        rep_path: nsys报告文件路径
        warmup_times: 预热次数
        run_times: 实际运行次数
        profile_type: profile类型标识（用于CSV文件命名）
        op_name: 算子名称（用于日志）
        task_id: 任务ID（用于日志）
        
    Returns:
        (success, error_msg, avg_time_us): 是否成功，错误信息，平均时间（微秒）
    """
    try:
        dir_plib = Path(rep_path).resolve().parent
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 在CSV文件名中添加profile_type标识
        type_suffix = f"_{profile_type}" if profile_type else ""
        csv_base = f"nsys_report_{timestamp}{type_suffix}"
        csv_path = dir_plib / csv_base
        
        # 导出csv
        cmd = f'nsys stats --report gputrace --timeunit us --format csv --output {csv_path} {rep_path}'
        logger.debug(f"[{task_id}:{op_name}] Running nsys stats: {cmd}")
        subprocess.run(cmd, shell=True, check=True)
        csv_path = dir_plib / f"{csv_base}_gputrace.csv"

        if not os.path.exists(csv_path):
            return False, "未生成csv文件", float('inf')
            
        df = pd.read_csv(csv_path)
        
        # 兼容不同nsys版本的列名
        name_col = None
        for col in df.columns:
            if col.lower() in ["name", "function name", "kernel name", "Name"]:
                name_col = col
                break
        if not name_col:
            # 兜底找包含name的列
            for col in df.columns:
                if "name" in col.lower():
                    name_col = col
                    break
                    
        time_col = None
        for col in df.columns:
            if "time (ns)" in col.lower() or "average" in col.lower() or "duration" in col.lower():
                time_col = col
                break
                
        if not name_col or not time_col:
            return False, "未找到kernel名或耗时列", float('inf')
            
        total_count = warmup_times + run_times
        op_counts = df[name_col].value_counts()
        valid_ops = op_counts[op_counts == total_count]
        
        if len(valid_ops) == 0:
            return False, "没有找到符合预期次数的kernel", float('inf')
            
        df_valid = df[df[name_col].isin(valid_ops.index)]
        total_avg_time = 0.0
        
        for op_name_iter in valid_ops.index:
            op_data = df_valid[df_valid[name_col] == op_name_iter][time_col].tolist()
            if len(op_data) > warmup_times:
                valid_data = op_data[warmup_times:]
                avg_time = sum(valid_data) / len(valid_data)
                total_avg_time += avg_time  # timeunit us
                
        return True, "", total_avg_time
        
    except Exception as e:
        logger.error(f"[{task_id}:{op_name}] 分析nsys数据时出错: {e}")
        return False, f"分析nsys数据时出错: {str(e)}", float('inf')
