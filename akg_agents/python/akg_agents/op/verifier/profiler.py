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
NPU Profiler 模块。

提供 NPU 性能分析功能，支持：
- 精确的执行时间测量
- L2 cache 清除（可选）
- 自动过滤无关的 warning 输出
"""

import os
import sys
import contextlib
import re
import shutil
import time
from typing import Callable, Tuple, Optional, Literal
import torch
import torch_npu
import pandas as pd

# 导入 L2 cache 清除相关功能
from .l2_cache_clear import (
    DslType,
    L2_CACHE_CLEAR_KERNEL_NAME,
    clear_l2_cache,
    get_l2_cache_warnings,
    clear_l2_cache_warnings,
)

try:
    from akg_agents.op.utils.triton_autotune_patch import AKG_RESTORE_COPY_KERNEL_NAME
except ImportError:
    AKG_RESTORE_COPY_KERNEL_NAME = "AKG_restore_copy"

# 预编译正则表达式，提高性能
# 过滤 profiler 相关的噪声输出
_FILTER_PATTERNS = re.compile(
    r'('
    r'Please DO NOT tune args|'
    r'Invalid parameter export_type|'
    r'Start parsing profiling data|'
    r'CANN profiling data parsed|'
    r'All profiling data parsed|'
    r'\[WARNING\]|'
    r'\[INFO\]|'
    r'profiler\.py:|'
    # 过滤 triton 编译相关的 warning
    r'WARNING:\s*Grid.*physical limit|'
    r'WARNING:\s*Grid.*performance'
    r')'
)

_SYMBOL_PATTERN = re.compile(r'^[\\\|/\-_=+*#~`!@$%^&()\[\]{}.,;:\'"<>?\s]+$')
_DECORATION_PATTERN = re.compile(r'[\\\|\-=/]{3,}')


def suppress_output():
    """
    创建输出抑制上下文管理器，过滤特定的 WARNING/INFO 输出。
    
    注意：此过滤器不会过滤 L2 cache 相关的警告消息，
    这些消息通过 l2_cache_clear 模块收集并在 profiler 结束后输出。
    """
    class OutputFilter:
        def __init__(self, original_stream):
            self.original_stream = original_stream
            self.suppress_next_lines = 0

        def write(self, text):
            # 如果正在抑制后续行，减少计数器
            if self.suppress_next_lines > 0:
                self.suppress_next_lines -= 1
                if not text.strip():
                    return

            # 使用预编译的正则表达式快速匹配
            if _FILTER_PATTERNS.search(text):
                self.suppress_next_lines = 2
                return

            stripped_text = text.strip()

            # 完全空行
            if not stripped_text:
                return

            # 使用正则表达式快速检查符号行
            if len(stripped_text) <= 50 and _SYMBOL_PATTERN.match(stripped_text):
                unique_chars = set(stripped_text.replace(' ', '').replace('\t', ''))
                if len(unique_chars) <= 3:
                    return

            # 使用正则表达式检查装饰线
            if _DECORATION_PATTERN.search(stripped_text):
                return

            # 其他内容正常输出
            self.original_stream.write(text)

        def flush(self):
            if hasattr(self.original_stream, 'flush'):
                self.original_stream.flush()

        def __getattr__(self, name):
            return getattr(self.original_stream, name)

    @contextlib.contextmanager
    def output_suppressor():
        old_stdout = sys.stdout
        old_stderr = sys.stderr

        try:
            sys.stdout = OutputFilter(old_stdout)
            sys.stderr = OutputFilter(old_stderr)
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    return output_suppressor()


def profiler_npu_core(fn: Callable, warmup: int = 25, active: int = 100, 
                      prof_dir_name: Optional[str] = None,
                      clear_l2_cache_flag: bool = False,
                      dsl: DslType = "other",
                      filter_restore_copy: bool = False) -> Tuple[float, str]:
    """
    NPU profiler 核心函数。
    
    Args:
        fn: 要 profile 的函数
        warmup: warmup 次数
        active: 有效测量次数
        prof_dir_name: profile 结果目录名
        clear_l2_cache_flag: 是否在每次迭代前清除 L2 cache
        dsl: DSL 类型，决定 L2 cache 清除方式
             - "triton_ascend": 使用专用 triton kernel（推荐，可精确过滤）
             - 其他: 使用 tensor.zero_()（fallback，有误判风险）
    
    Returns:
        Tuple[float, str]: (执行时间(微秒), profile结果目录路径)
    """
    fn()
    torch.npu.synchronize()

    experimental_config = torch_npu.profiler._ExperimentalConfig(
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        l2_cache=False,
        data_simplification=False
    )
    # 将 warmup 步数放入 skip_first，保持预热效果但避免 warmup_trace 的副作用
    # fn() 依然会在所有 skip_first 步骤中执行，确保充分预热
    skip_first = 1 + warmup
    wait = 0
    warmup_prof = 0
    repeat = 1
    total = skip_first + (wait + warmup_prof + active) * repeat

    timestamp = int(time.time() * 1000)

    if prof_dir_name is not None:
        profile_path = os.path.join(os.getcwd(), f"{prof_dir_name}_{timestamp}")
    else:
        profile_path = os.path.join(os.getcwd(), f"profile_results_{timestamp}")

    # 预初始化 L2 cache buffer（如果需要清除）
    if clear_l2_cache_flag:
        # 预热 L2 cache 清除操作
        clear_l2_cache(dsl)

    with torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.NPU
        ],
        schedule=torch_npu.profiler.schedule(wait=wait, warmup=warmup_prof, active=active,
                                             repeat=repeat, skip_first=skip_first),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(profile_path),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        with_flops=False,
        with_modules=False,
        experimental_config=experimental_config,
    ) as prof:
        for _ in range(total):
            if clear_l2_cache_flag:
                clear_l2_cache(dsl)
            fn()
            prof.step()
            torch.npu.synchronize()

    exec_time = collect_time(profile_path, active, clear_l2_cache_flag=clear_l2_cache_flag,
                             dsl=dsl, filter_restore_copy=filter_restore_copy)
    return exec_time, profile_path


def profiler_npu(fn: Callable, warmup: int = 25, active: int = 100, prof_dir_name: Optional[str] = None,
                 keep_res: bool = False, suppress_warnings: bool = True,
                 clear_l2_cache: bool = False, dsl: DslType = "other",
                 filter_restore_copy: bool = False) -> float:
    """
    NPU profiler 主函数。

    Args:
        fn: 要 profile 的函数
        warmup: warmup 次数
        active: 有效测量次数
        prof_dir_name: profile 结果目录名
        keep_res: 是否保留结果文件
        suppress_warnings: 是否抑制 WARNING/INFO 输出
        clear_l2_cache: 是否在每次迭代前清除 L2 cache
        dsl: DSL 类型，决定 L2 cache 清除方式
             - "triton_ascend": 使用专用 triton kernel（推荐，可精确过滤）
             - 其他: 使用 tensor.zero_()（fallback，有误判风险）
    
    Returns:
        float: 平均执行时间（微秒）
    """
    # 清空之前的警告消息
    clear_l2_cache_warnings()
    
    if suppress_warnings:
        with suppress_output():
            exec_time, profile_path = profiler_npu_core(
                fn, warmup, active, prof_dir_name, 
                clear_l2_cache_flag=clear_l2_cache, dsl=dsl,
                filter_restore_copy=filter_restore_copy,
            )
    else:
        exec_time, profile_path = profiler_npu_core(
            fn, warmup, active, prof_dir_name,
            clear_l2_cache_flag=clear_l2_cache, dsl=dsl,
            filter_restore_copy=filter_restore_copy,
        )
    
    # profiler 结束后输出收集的 L2 cache 警告消息（绕过 suppress_output）
    warnings_list = get_l2_cache_warnings()
    if warnings_list:
        for warning_msg in warnings_list:
            # 使用 sys.__stderr__ 绕过任何 stderr 重定向
            print(f"[WARN] {warning_msg}", file=sys.__stderr__)

    # 清理结果文件
    if not keep_res and os.path.exists(profile_path):
        shutil.rmtree(profile_path)

    return exec_time


def collect_time(base_dir: str, active: int, clear_l2_cache_flag: bool = False,
                 dsl: DslType = "other", filter_restore_copy: bool = False) -> float:
    """
    从 profiling 结果中收集时间信息。

    Args:
        base_dir: profiling 结果目录
        active: 有效测量次数
        clear_l2_cache_flag: 是否启用了 L2 cache 清除
        dsl: DSL 类型，决定如何过滤 L2 cache 清除操作
             - "triton_ascend": 过滤名为 "AKG_l2cache_clear" 的 kernel
             - 其他: 过滤 "ZerosLike" 类型的操作

    Returns:
        float: 平均执行时间(微秒)，失败时返回 float('inf')
    """
    if not os.path.exists(base_dir):
        print(f"Base directory not found: {base_dir}")
        return float('inf')

    for root, _, files in os.walk(base_dir):
        for file in files:
            if file != 'op_statistic.csv':
                continue

            target_file = os.path.join(root, file)
            try:
                df = pd.read_csv(target_file)
            except (pd.errors.EmptyDataError, pd.errors.ParserError, FileNotFoundError) as e:
                print(f"Failed to read {target_file}: {e}")
                continue

            # 检查必需的列
            required_columns = ['Count', 'Total Time(us)']
            if not all(col in df.columns for col in required_columns):
                print(f"Missing required columns in {target_file}. Found: {list(df.columns)}")
                continue

            # 过滤有效操作
            try:
                if clear_l2_cache_flag or filter_restore_copy:
                    df = _filter_l2_cache_clear_ops(df, dsl,
                                                    filter_restore_copy=filter_restore_copy)
                
                valid_ops = df[df['Count'] % active == 0].copy()

                if valid_ops.empty:
                    print(f"No valid ops found in {target_file}")
                    continue

                total_time_sum = valid_ops['Total Time(us)'].sum()
                if pd.isna(total_time_sum) or total_time_sum <= 0:
                    print(f"Invalid timing data in {target_file}")
                    continue

                average_time = total_time_sum / active
                return average_time

            except (KeyError, ValueError, ZeroDivisionError) as e:
                print(f"Error processing timing data in {target_file}: {e}")
                continue

    print(f"No valid timing data found in {base_dir}")
    return float('inf')


def _filter_l2_cache_clear_ops(df: pd.DataFrame, dsl: DslType,
                                filter_restore_copy: bool = False) -> pd.DataFrame:
    """
    从 profiling 结果中过滤掉 AKG 框架内部操作。

    过滤内容：
      - L2 cache 清除 kernel（AKG_l2cache_clear / ZerosLike）
      - restore_value 的 copy kernel（filter_restore_copy=True 时，
        按 kernel 名字 AKG_restore_copy 精确过滤，与 l2_cache_clear 同一模式）

    Args:
        df: profiling 数据 DataFrame
        dsl: DSL 类型
        filter_restore_copy: 是否过滤 restore_value 产生的 copy 操作。
             仅在 autotune benchmark 阶段开启，最终 profiling 不开。

    Returns:
        pd.DataFrame: 过滤后的 DataFrame
    """
    if dsl == "triton_ascend":
        col = None
        if 'OP Type' in df.columns:
            col = 'OP Type'
        elif 'Name' in df.columns:
            col = 'Name'

        if col is not None:
            keep = pd.Series(True, index=df.index)
            keep &= ~df[col].str.contains(
                L2_CACHE_CLEAR_KERNEL_NAME, case=False, na=False, regex=False)
            if filter_restore_copy:
                keep &= ~df[col].str.contains(
                    AKG_RESTORE_COPY_KERNEL_NAME, case=False, na=False, regex=False)
            filtered_df = df[keep]
        else:
            filtered_df = df
    else:
        if 'OP Type' in df.columns:
            filter_cond = ~df['OP Type'].str.contains(
                r'^ZerosLike$', case=False, na=False, regex=True
            )
            filtered_df = df[filter_cond]
        elif 'Type' in df.columns:
            filter_cond = ~df['Type'].str.contains(
                r'^ZerosLike$', case=False, na=False, regex=True
            )
            filtered_df = df[filter_cond]
        else:
            filtered_df = df

    return filtered_df
