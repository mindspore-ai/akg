#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""
plog 日志解析脚本（卡死/崩溃版本）
用途：解析 Ascend plog 日志，提取卡死、崩溃、硬件异常信号

使用方法：
    python3 parse_plog.py <plog_file_path>
    python3 parse_plog.py  # 使用最新日志
"""

import os
import sys
import re
import glob
from typing import Dict, Optional


class PlogParser:
    """plog 日志解析器（卡死/崩溃版本）"""

    def __init__(self, log_path: str):
        self.log_path = log_path
        self.timeouts = []
        self.crashes = []
        self.hardware_exceptions = []

    def parse(self) -> Dict:
        if not os.path.exists(self.log_path):
            return {"error": f"日志文件不存在: {self.log_path}"}

        with open(self.log_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        for line_num, line in enumerate(lines, 1):
            self._parse_line(line.strip(), line_num)

        return {
            "log_file": self.log_path,
            "total_lines": len(lines),
            "timeouts": self.timeouts,
            "crashes": self.crashes,
            "hardware_exceptions": self.hardware_exceptions,
            "summary": self._generate_summary()
        }

    @staticmethod
    def _classify_event(line: str) -> str:
        if re.search(r'SIGSEGV|SIGABRT|segmentation fault|segment fault|abort|killed|fatal|core dumped', line, re.IGNORECASE):
            return "信号崩溃"
        elif re.search(r'507035|aic error|vector core exception', line, re.IGNORECASE):
            return "硬件异常"
        elif re.search(r'deadlock', line, re.IGNORECASE):
            return "Buffer死锁"
        else:
            return "超时挂起"

    def _parse_line(self, line: str, line_num: int):
        if re.search(r'timeout|hang|stuck|not respond|no response|deadlock', line, re.IGNORECASE):
            self.timeouts.append({
                "line": line_num,
                "content": line,
                "type": self._classify_event(line)
            })

        if re.search(r'segmentation fault|segment fault|SIGSEGV|SIGABRT|abort|core dumped|killed|fatal', line, re.IGNORECASE):
            self.crashes.append({
                "line": line_num,
                "content": line,
                "type": self._classify_event(line)
            })

        if re.search(r'507035|aic error|vector core exception', line, re.IGNORECASE):
            self.hardware_exceptions.append({
                "line": line_num,
                "content": line,
                "type": self._classify_event(line)
            })

    def _generate_summary(self) -> str:
        summary_lines = []
        summary_lines.append(f"超时/挂起次数: {len(self.timeouts)}")
        summary_lines.append(f"崩溃次数: {len(self.crashes)}")
        summary_lines.append(f"硬件异常次数: {len(self.hardware_exceptions)}")

        all_events = self.timeouts + self.crashes + self.hardware_exceptions
        if all_events:
            event_types = {}
            for ev in all_events:
                event_type = ev["type"]
                event_types[event_type] = event_types.get(event_type, 0) + 1

            summary_lines.append("\n事件类型分布:")
            for etype, count in sorted(event_types.items(), key=lambda x: -x[1]):
                summary_lines.append(f"  - {etype}: {count}")

        return "\n".join(summary_lines)


def find_latest_plog() -> Optional[str]:
    log_dir = os.path.expanduser("~/ascend/log/debug/plog")
    if not os.path.exists(log_dir):
        return None

    log_files = glob.glob(os.path.join(log_dir, "plog-pid_*.log"))
    if not log_files:
        return None

    log_files.sort(key=os.path.getmtime, reverse=True)
    return log_files[0]


def print_report(result: Dict):
    print("=" * 60)
    print("plog 日志解析报告（卡死/崩溃）")
    print("=" * 60)
    print(f"日志文件: {result['log_file']}")
    print(f"总行数: {result['total_lines']}")
    print()
    print(result['summary'])
    print()

    if result['timeouts']:
        print("=" * 60)
        print("超时/挂起信息")
        print("=" * 60)
        for ev in result['timeouts']:
            print(f"[Line {ev['line']}] [{ev['type']}]")
            print(f"  {ev['content'][:200]}")
            print()

    if result['crashes']:
        print("=" * 60)
        print("崩溃信息")
        print("=" * 60)
        for ev in result['crashes']:
            print(f"[Line {ev['line']}] [{ev['type']}]")
            print(f"  {ev['content'][:200]}")
            print()

    if result['hardware_exceptions']:
        print("=" * 60)
        print("硬件异常信息")
        print("=" * 60)
        for ev in result['hardware_exceptions']:
            print(f"[Line {ev['line']}] [{ev['type']}]")
            print(f"  {ev['content'][:200]}")
            print()


def main():
    if len(sys.argv) > 1:
        log_path = sys.argv[1]
    else:
        log_path = find_latest_plog()
        if not log_path:
            print("错误: 未找到 plog 日志文件")
            print("用法: python3 parse_plog.py <plog_file_path>")
            sys.exit(1)
        print(f"使用最新日志: {log_path}")

    parser = PlogParser(log_path)
    result = parser.parse()

    if "error" in result:
        print(f"错误: {result['error']}")
        sys.exit(1)

    print_report(result)

    if result['hardware_exceptions']:
        sys.exit(4)
    elif result['crashes']:
        sys.exit(2)
    elif result['timeouts']:
        sys.exit(3)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
