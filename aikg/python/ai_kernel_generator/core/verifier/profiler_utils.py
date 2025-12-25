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
import subprocess
from typing import Tuple, Optional
from pathlib import Path
from datetime import datetime
import pandas as pd
from ai_kernel_generator.utils.process_utils import run_command

logger = logging.getLogger(__name__)


def run_profile_scripts_and_collect_results(verify_dir: str, op_name: str, task_id: str = "0") -> Tuple[float, float]:
    """运行性能测试脚本并收集结果

    Args:
        verify_dir: 验证目录，包含性能测试脚本
        op_name: 算子名称
        task_id: 任务ID

    Returns:
        (base_time_us, gen_time_us): 基准时间和生成时间（微秒）
        - 如果 base 脚本不存在（跨后端场景），base_time_us 返回 inf
    
    注意: 此函数是线程安全的，不使用 os.chdir()。
    多个任务可以在线程池中并发执行而不会互相干扰。
    """
    try:
        base_time_us = float('inf')
        
        # 步骤1：运行基准性能测试脚本（如果存在）
        # 跨后端场景（使用参考数据）下，base 脚本可能不存在
        base_script = f"profile_{op_name}_base.py"
        base_script_path = os.path.join(verify_dir, base_script)
        
        if os.path.exists(base_script_path):
            # 使用 cwd 参数指定工作目录（线程安全），不使用 os.chdir()
            base_result = run_command(["python", base_script], cmd_msg="base_profile", timeout=300, cwd=verify_dir)
            if not base_result[0]:
                logger.error(f"[{op_name}: {task_id}] 基准性能脚本执行失败: {base_result[1]}")
                # base 失败不影响 generation 的执行
            else:
                base_time_us = read_profile_result_from_json(verify_dir, "base_profile_result.json")
        else:
            logger.info(f"[{op_name}: {task_id}] 基准性能脚本不存在（跨后端场景），跳过 base profile")

        # 步骤2：运行生成代码性能测试脚本
        gen_script = f"profile_{op_name}_generation.py"
        gen_result = run_command(["python", gen_script], cmd_msg="generation_profile", timeout=300, cwd=verify_dir)
        if not gen_result[0]:
            logger.error(f"[{op_name}: {task_id}] 生成代码性能脚本执行失败: {gen_result[1]}")
            return base_time_us, float('inf')

        # 步骤3：从JSON文件读取性能数据
        gen_time_us = read_profile_result_from_json(verify_dir, "generation_profile_result.json")
        
        logger.info(f"[{op_name}: {task_id}] Read profile results: base={base_time_us:.2f} us, gen={gen_time_us:.2f} us")

        return base_time_us, gen_time_us

    except Exception as e:
        logger.error(f"[{op_name}: {task_id}] 性能脚本执行和结果收集失败: {e}")
        return float('inf'), float('inf')


def read_profile_result_from_json(verify_dir: str, json_filename: str) -> float:
    """从JSON文件读取性能结果
    
    Args:
        verify_dir: 验证目录
        json_filename: JSON文件名
        
    Returns:
        平均时间（微秒）
    """
    json_path = os.path.join(verify_dir, json_filename)
    try:
        if not os.path.exists(json_path):
            error_msg = f"JSON file not found: {json_path}"
            logger.error(error_msg)
            print(f"[DEBUG] {error_msg}")
            
            cwd_msg = f"Current directory: {os.getcwd()}"
            logger.error(cwd_msg)
            print(f"[DEBUG] {cwd_msg}")
            
            files_msg = f"Files in verify_dir: {os.listdir(verify_dir) if os.path.exists(verify_dir) else 'N/A'}"
            logger.error(files_msg)
            print(f"[DEBUG] {files_msg}")
            return float('inf')
            
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Try multiple possible field names for compatibility
            avg_time = data.get('avg_time_us') or data.get('execution_time_us') or float('inf')
            print(f"[DEBUG] Successfully read {json_filename}: avg_time={avg_time} us, data_keys={list(data.keys())}")
            return avg_time
    except Exception as e:
        error_msg = f"Failed to read {json_filename}: {e}"
        logger.error(error_msg)
        print(f"[DEBUG] {error_msg}")
        import traceback
        traceback.print_exc()
        return float('inf')


def run_msprof(script_path: str, op_name: str = "", task_id: str = "0") -> Tuple[bool, str, Optional[str]]:
    """运行msprof性能分析
    
    Args:
        script_path: Python脚本路径
        op_name: 算子名称（用于日志）
        task_id: 任务ID（用于日志）
        
    Returns:
        (success, error_msg, prof_path): 是否成功，错误信息，prof数据路径
    """
    try:
        process = subprocess.run(
            f'msprof --application="python {script_path}"',
            shell=True, capture_output=True, text=True, timeout=600
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


def run_nsys(script_path: str, op_name: str = "", task_id: str = "0") -> Tuple[bool, str, Optional[str]]:
    """运行nsys性能分析
    
    Args:
        script_path: Python脚本路径
        op_name: 算子名称（用于日志）
        task_id: 任务ID（用于日志）
        
    Returns:
        (success, error_msg, rep_path): 是否成功，错误信息，nsys报告文件路径
    """
    try:
        output_name = "nsys_report_" + os.path.basename(script_path).replace(".py", "")
        cmd = f'nsys profile --output={output_name} python {script_path}'
        logger.debug(f"[{task_id}:{op_name}] Running nsys profile: {cmd}")
        process = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=600)
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

