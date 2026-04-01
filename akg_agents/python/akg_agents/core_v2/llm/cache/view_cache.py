#!/usr/bin/env python3
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

"""查看 LLM Cache 缓存文件内容的命令行工具（支持本地文件和内存缓存）"""

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

if __package__:
    from . import cache_utils
else:
    current_dir = Path(__file__).parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    import cache_utils


def _replay_hash_from_key(cache_key: str) -> str:
    if ":" in cache_key:
        return cache_key.split(":", 1)[0]
    return ""


def _format_create_time(item: dict) -> str:
    create_time = item.get("create_time", 0)
    if not create_time:
        return "未知"
    time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(create_time))
    expire = item.get("expire_seconds", -1)
    if expire and expire > 0:
        expire_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(create_time + expire))
        return f"{time_str} (过期: {expire_time})"
    return f"{time_str} (永不过期)"


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _truncate_text(text: str, max_len: int = 120) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _extract_task_desc(item: dict) -> str:
    params = item.get("params") if isinstance(item, dict) else {}
    if isinstance(params, dict):
        for key in ("task_desc", "task_description", "task", "prompt"):
            value = params.get(key)
            if isinstance(value, str) and value.strip():
                return _truncate_text(_normalize_text(value))

    result = item.get("result") if isinstance(item, dict) else {}
    if isinstance(result, dict):
        for key in ("task_desc", "task_description"):
            value = result.get(key)
            if isinstance(value, str) and value.strip():
                return _truncate_text(_normalize_text(value))

    messages = item.get("messages") if isinstance(item, dict) else []
    if isinstance(messages, list):
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            if str(msg.get("role", "")).lower() != "user":
                continue
            content = str(msg.get("content", "") or "")
            if not content.strip():
                continue

            # 优先从显式的任务段提取，减少把系统模板当成 task_desc 的概率。
            patterns = [
                r"(?:任务描述|任务需求|task_desc|task description)\s*[:：]\s*(.+)",
                r"##\s*(?:任务描述|任务需求|Task Description)\s*\n+\s*(.+)",
            ]
            for pattern in patterns:
                match = re.search(pattern, content, flags=re.IGNORECASE | re.DOTALL)
                if match:
                    extracted = _normalize_text(match.group(1).split("\n\n", 1)[0])
                    if extracted:
                        return _truncate_text(extracted)

            # 回退：取首个有信息量的文本行。
            for line in content.splitlines():
                line = line.strip()
                if not line:
                    continue
                if line.startswith(("#", "```", "---")):
                    continue
                if "你是" in line and "Agent" in line:
                    continue
                return _truncate_text(_normalize_text(line))

    return "<未提取到 task_desc>"


def _print_cache_entries(data: dict, show_details: bool = False) -> None:
    replay_groups = {}
    replay_order = []
    hidden_non_replay_keys = 0

    for key, item in data.items():
        if not isinstance(item, dict):
            continue

        replay_hash = _replay_hash_from_key(key)
        if replay_hash:
            if replay_hash not in replay_groups:
                replay_groups[replay_hash] = {
                    "first_key": key,
                    "first_item": item,
                    "all_keys": [],
                }
                replay_order.append(replay_hash)
            replay_groups[replay_hash]["all_keys"].append(key)
        else:
            hidden_non_replay_keys += 1

    if replay_order:
        print("\n🧭 按 case 展示（仅显示 replay 回放所需的首 key）")
        for i, replay_hash in enumerate(replay_order, 1):
            group = replay_groups[replay_hash]
            first_key = group["first_key"]
            first_item = group["first_item"]

            print(f"\n{'=' * 60}")
            print(f"  [{i}] Replay Hash: {replay_hash}")
            print(f"      首 Key: {first_key}")
            print(f"      task_desc: {_extract_task_desc(first_item)}")
            print(f"      创建时间: {_format_create_time(first_item)}")
            print(f"      该 case 的 session key 数: {len(group['all_keys'])}")

            if show_details:
                messages = first_item.get("messages", [])
                print(f"      消息数(首 key): {len(messages)}")

                tools = first_item.get("tools", [])
                if tools:
                    print(f"      工具数(首 key): {len(tools)}")

                params = first_item.get("params", {})
                if params:
                    print(f"      参数(首 key): {params}")

                result = first_item.get("result", {})
                result_str = json.dumps(result, ensure_ascii=False, indent=4)
                if len(result_str) > 500:
                    result_str = result_str[:500] + "\n        ... (太长已截断)"
                print(f"      完整结果(首 key):\n        {result_str}")

        if hidden_non_replay_keys:
            print(
                f"\nℹ️ 已隐藏 {hidden_non_replay_keys} 个非 replay 主键（例如内容哈希 key），"
                "避免干扰回放定位。"
            )
        return

    # 回退：如果没有 session/replay key，保持原始行为。
    for i, (key, item) in enumerate(data.items(), 1):
        print(f"\n{'=' * 60}")
        print(f"  [{i}] Key: {key}")
        print(f"      创建时间: {_format_create_time(item)}")

        messages = item.get("messages", []) if isinstance(item, dict) else []
        print(f"      消息数: {len(messages)}")
        if show_details and messages:
            print("      消息列表:")
            for j, msg in enumerate(messages, 1):
                role = msg.get("role", "unknown")
                content = str(msg.get("content", "") or "")
                print(f"        [{j}][{role}]: {_truncate_text(content, 100)}")

        if show_details and isinstance(item, dict):
            result = item.get("result", {})
            result_str = json.dumps(result, ensure_ascii=False, indent=4)
            if len(result_str) > 500:
                result_str = result_str[:500] + "\n        ... (太长已截断)"
            print(f"      完整结果:\n        {result_str}")

            tools = item.get("tools", [])
            if tools:
                print(f"      工具数: {len(tools)}")
            params = item.get("params", {})
            if params:
                print(f"      参数: {params}")


def _is_expired(item: dict, now_ts: float) -> bool:
    expire_seconds = item.get("expire_seconds", -1)
    create_time = item.get("create_time", 0)
    if expire_seconds is None or expire_seconds <= 0:
        return False
    if not create_time:
        return False
    return now_ts - create_time > expire_seconds


def _key_prefix(cache_key: str) -> str:
    if ":" in cache_key:
        return cache_key.split(":", 1)[0]
    return "content_hash"


def _format_bytes(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.2f} KB"
    if num_bytes < 1024 * 1024 * 1024:
        return f"{num_bytes / (1024 * 1024):.2f} MB"
    return f"{num_bytes / (1024 * 1024 * 1024):.2f} GB"


def _print_cache_stats(data: dict, cache_path: Path) -> None:
    now_ts = time.time()
    total_entries = len(data)
    expired_entries = sum(1 for item in data.values() if isinstance(item, dict) and _is_expired(item, now_ts))
    prefix_counter = Counter(_key_prefix(key) for key in data.keys())

    print(f"\n📈 缓存统计 (文件: {cache_path})")
    print(f"  - 缓存条目总数: {total_entries}")
    print(f"  - 本地文件缓存条目数: {total_entries}")
    print("  - 内存缓存条目数: N/A (CLI 无法读取其他进程的实时内存缓存)")
    print(f"  - 过期条目数: {expired_entries}")
    if cache_path.exists():
        file_size = cache_path.stat().st_size
        print(f"  - 缓存文件路径: {cache_path}")
        print(f"  - 缓存文件大小: {_format_bytes(file_size)} ({file_size} B)")
    print("  - 按会话/前缀聚合统计:")
    if prefix_counter:
        for prefix, count in prefix_counter.most_common():
            print(f"    * {prefix}: {count}")
    else:
        print("    * 无")


def view_local_cache_file(cache_path: str, show_details: bool = False, show_stats: bool = False) -> None:
    """查看本地缓存文件内容（静态查看）"""
    cache_path = Path(cache_path).expanduser()

    if not cache_path.exists():
        print(f"❌ 缓存文件不存在: {cache_path}")
        return

    try:
        content = cache_path.read_text(encoding="utf-8")
        if not content.strip():
            data = {}
        else:
            data = json.loads(content)
    except Exception as e:
        print(f"❌ 读取缓存文件失败: {e}")
        return

    if show_stats:
        _print_cache_stats(data, cache_path)

    if not data:
        if not show_stats:
            print("📭 缓存文件为空")
        return

    if show_details:
        print(f"\n📊 缓存文件: {cache_path}")
        print(f"🔢 缓存条目数: {len(data)}")
        _print_cache_entries(data, show_details=show_details)


def view_live_cache(
    cache_file_path: str = "~/.akg/llm_cache/llm_test_cache.json",
    show_details: bool = False,
    show_stats: bool = False,
) -> None:
    """查看当前内存中的缓存（实时查看，读取本地文件+内存同步）"""
    cache_file_path = os.path.expanduser(cache_file_path)

    # 直接读取本地文件（反映最新写入）
    data = cache_utils.read_cache_file(cache_file_path)

    if show_stats:
        _print_cache_stats(data, Path(cache_file_path))

    if not data:
        if not show_stats:
            print("📭 当前无缓存（内存和本地文件都为空）")
        return

    if show_details:
        print(f"\n📊 当前缓存内容 (文件: {cache_file_path})")
        print(f"🔢 缓存条目数: {len(data)}")
        _print_cache_entries(data, show_details=show_details)


def main():
    parser = argparse.ArgumentParser(
        description="查看 LLM Cache 缓存文件内容"
    )
    parser.add_argument(
        "cache_file",
        nargs="?",
        default="~/.akg/llm_cache/llm_test_cache.json",
        help="缓存文件路径 (默认: ~/.akg/llm_cache/llm_test_cache.json)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="显示详细信息（包括完整消息内容、结果等）"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="显示缓存统计摘要（总数、过期数、按会话/前缀聚合、文件大小等）"
    )

    args = parser.parse_args()

    # 检查是否为内存模式（默认文件路径）
    if args.cache_file == "~/.akg/llm_cache/llm_test_cache.json":
        view_live_cache(args.cache_file, show_details=args.verbose, show_stats=args.stats)
    else:
        view_local_cache_file(args.cache_file, show_details=args.verbose, show_stats=args.stats)


if __name__ == "__main__":
    main()
