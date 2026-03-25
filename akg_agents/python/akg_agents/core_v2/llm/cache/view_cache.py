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
import sys
import time
from pathlib import Path

if __package__:
    from . import cache_utils
else:
    current_dir = Path(__file__).parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    import cache_utils


def _print_cache_entries(data: dict, show_details: bool = False) -> None:
    for i, (key, item) in enumerate(data.items(), 1):
        print(f"\n{'='*60}")
        print(f"  [{i}] Key: {key}")

        create_time = item.get('create_time', 0)
        if create_time:
            time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(create_time))
            expire = item.get('expire_seconds', -1)
            if expire > 0:
                expire_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(create_time + expire))
                time_str += f" (过期: {expire_time})"
            else:
                time_str += " (永不过期)"
            print(f"      创建时间: {time_str}")

        messages = item.get('messages', [])
        print(f"      消息数: {len(messages)}")
        if show_details and messages:
            print("      消息列表:")
            for j, msg in enumerate(messages, 1):
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')
                if len(content) > 100:
                    content = content[:100] + "..."
                print(f"        [{j}][{role}]: {content}")

        if show_details:
            result = item.get('result', {})
            result_str = json.dumps(result, ensure_ascii=False, indent=4)
            if len(result_str) > 500:
                result_str = result_str[:500] + "\n        ... (太长已截断)"
            print(f"      完整结果:\n        {result_str}")

            tools = item.get('tools', [])
            if tools:
                print(f"      工具数: {len(tools)}")
            params = item.get('params', {})
            if params:
                print(f"      参数: {params}")


def view_local_cache_file(cache_path: str, show_details: bool = False) -> None:
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

    if not data:
        print("📭 缓存文件为空")
        return

    print(f"\n📊 缓存文件: {cache_path}")
    print(f"🔢 缓存条目数: {len(data)}")
    _print_cache_entries(data, show_details=show_details)


def view_live_cache(cache_file_path: str = "~/.akg/llm_cache/llm_test_cache.json", show_details: bool = False) -> None:
    """查看当前内存中的缓存（实时查看，读取本地文件+内存同步）"""
    cache_file_path = os.path.expanduser(cache_file_path)

    # 直接读取本地文件（反映最新写入）
    data = cache_utils.read_cache_file(cache_file_path)

    if not data:
        print("📭 当前无缓存（内存和本地文件都为空）")
        return

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

    args = parser.parse_args()

    # 检查是否为内存模式（默认文件路径）
    if args.cache_file == "~/.akg/llm_cache/llm_test_cache.json":
        view_live_cache(args.cache_file, show_details=args.verbose)
    else:
        view_local_cache_file(args.cache_file, show_details=args.verbose)


if __name__ == "__main__":
    main()
