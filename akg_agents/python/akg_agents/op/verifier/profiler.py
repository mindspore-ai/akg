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

import os
import sys
import contextlib
import re
import shutil
import time
from typing import Callable, Tuple, Optional
import torch
import torch_npu
import pandas as pd


# 预编译正则表达式，提高性能
_FILTER_PATTERNS = re.compile(r'(Please DO NOT tune args|Invalid parameter export_type|'
                              r'Start parsing profiling data|CANN profiling data parsed|'
                              r'All profiling data parsed|\[WARNING\]|\[INFO\]|profiler\.py:)')

_SYMBOL_PATTERN = re.compile(r'^[\\\|/\-_=+*#~`!@$%^&()\[\]{}.,;:\'"<>?\s]+$')
_DECORATION_PATTERN = re.compile(r'[\\\|\-=/]{3,}')


def suppress_output():
    """
    创建输出抑制上下文管理器，过滤特定的WARNING/INFO输出
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


def profiler_npu_core(fn: Callable, warmup: int = 25, active: int = 100, prof_dir_name: Optional[str] = None) -> Tuple[float, str]:
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
    warmup = 0
    repeat = 1
    total = skip_first + (wait + warmup + active) * repeat

    timestamp = int(time.time() * 1000)

    if prof_dir_name is not None:
        profile_path = os.path.join(os.getcwd(), f"{prof_dir_name}_{timestamp}")
    else:
        profile_path = os.path.join(os.getcwd(), f"profile_results_{timestamp}")

    with torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.NPU
        ],
        schedule=torch_npu.profiler.schedule(wait=wait, warmup=warmup, active=active,
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
            fn()
            prof.step()
            torch.npu.synchronize()

    exec_time = collect_time(profile_path, active)
    return exec_time, profile_path


def profiler_npu(fn: Callable, warmup: int = 25, active: int = 100, prof_dir_name: Optional[str] = None,
                 keep_res: bool = False, suppress_warnings: bool = True) -> float:
    """
    NPU profiler主函数

    Args:
        fn: 要profile的函数
        warmup: warmup次数
        active: 有效测量次数
        prof_dir_name: profile结果目录名
        keep_res: 是否保留结果文件
        suppress_warnings: 是否抑制WARNING/INFO输出
    """
    if suppress_warnings:
        with suppress_output():
            exec_time, profile_path = profiler_npu_core(fn, warmup, active, prof_dir_name)
    else:
        exec_time, profile_path = profiler_npu_core(fn, warmup, active, prof_dir_name)

    # 清理结果文件
    if not keep_res and os.path.exists(profile_path):
        shutil.rmtree(profile_path)

    return exec_time


def collect_time(base_dir: str, active: int) -> float:
    """
    从profiling结果中收集时间信息

    Args:
        base_dir: profiling结果目录
        active: 有效测量次数

    Returns:
        float: 平均执行时间(微秒)，失败时返回float('inf')
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
