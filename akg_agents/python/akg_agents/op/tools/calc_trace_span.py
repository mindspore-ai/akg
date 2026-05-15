#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
计算 Chrome Trace 文件的时间跨度。

跨度定义：所有 ph == "X" 的事件中，最早 ts 到最晚 (ts + dur) 的时间差。
"""

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="计算 traceEvents 中 ph==X 的时间跨度"
    )
    parser.add_argument(
        "input",
        type=Path,
        help="trace JSON 文件路径"
    )
    return parser.parse_args()


def calc_span(trace_events: list[dict]) -> float:
    x_events = [e for e in trace_events if e.get("ph") == "X"]
    if not x_events:
        raise ValueError("未找到 ph==X 的事件")

    min_ts = min(float(e.get("ts", 0) or 0) for e in x_events)
    max_end = max(
        float(e.get("ts", 0) or 0) + float(e.get("dur", 0) or 0)
        for e in x_events
    )
    return max_end - min_ts


def main() -> None:
    args = parse_args()
    input_path = args.input

    if not input_path.exists():
        raise FileNotFoundError(f"文件不存在: {input_path}")

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    trace_events = data.get("traceEvents", [])
    span = calc_span(trace_events)
    print(span)


if __name__ == "__main__":
    main()
