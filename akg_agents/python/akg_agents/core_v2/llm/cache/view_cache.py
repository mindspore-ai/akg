#!/usr/bin/env python3
"""查看 LLM Cache 缓存文件内容的命令行工具（支持本地文件和内存缓存）"""

import argparse
import json5
import sys
import os
import time
import tempfile
from pathlib import Path

# 动态加载 cache_utils
CACHE_UTILS_CODE = open(os.path.join(os.path.dirname(__file__), 'cache_utils.py')).read()
exec(CACHE_UTILS_CODE)


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
            data = json5.loads(content)
    except Exception as e:
        print(f"❌ 读取缓存文件失败: {e}")
        return

    if not data:
        print("📭 缓存文件为空")
        return

    print(f"\n📊 缓存文件: {cache_path}")
    print(f"🔢 缓存条目数: {len(data)}")

    for i, (key, item) in enumerate(data.items(), 1):
        print(f"\n{'='*60}")
        print(f"  [{i}] Key: {key}")
        print(f"      创建时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(item.get('create_time', 0)))}")
        print(f"      过期秒数: {item.get('expire_seconds', 'N/A')}")
        messages = item.get('messages', [])
        print(f"      消息数: {len(messages)}")
        if show_details:
            for msg in messages:
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')
                if len(content) > 80:
                    content = content[:80] + "..."
                print(f"        [{role}]: {content}")
            result = item.get('result', {})
            print(f"      结果预览: {str(result)[:200]}")


def view_live_cache(cache_file_path: str = "~/.akg/llm_cache/llm_test_cache.json5", show_details: bool = False) -> None:
    """查看当前内存中的缓存（实时查看，读取本地文件+内存同步）"""
    cache_file_path = os.path.expanduser(cache_file_path)

    # 直接读取本地文件（反映最新写入）
    data = read_cache_file(cache_file_path)

    if not data:
        print("📭 当前无缓存（内存和本地文件都为空）")
        return

    print(f"\n📊 当前缓存内容 (文件: {cache_file_path})")
    print(f"🔢 缓存条目数: {len(data)}")

    for i, (key, item) in enumerate(data.items(), 1):
        print(f"\n{'='*60}")
        print(f"  [{i}] Key: {key}")

        # 创建时间格式化
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

        # 消息内容
        messages = item.get('messages', [])
        print(f"      消息数: {len(messages)}")
        if messages:
            print(f"      消息列表:")
            for j, msg in enumerate(messages, 1):
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')
                if len(content) > 100:
                    content = content[:100] + "..."
                print(f"        [{j}][{role}]: {content}")

        # 结果内容
        if show_details:
            result = item.get('result', {})
            print(f"      完整结果:")
            result_str = json5.dumps(result, ensure_ascii=False, indent=4)
            if len(result_str) > 500:
                result_str = result_str[:500] + "\n        ... (太长已截断)"
            print(f"        {result_str}")

            # 工具信息
            tools = item.get('tools', [])
            if tools:
                print(f"      工具数: {len(tools)}")

            # 参数
            params = item.get('params', {})
            if params:
                print(f"      参数: {params}")


def main():
    parser = argparse.ArgumentParser(
        description="查看 LLM Cache 缓存文件内容"
    )
    parser.add_argument(
        "cache_file",
        nargs="?",
        default="~/.akg/llm_cache/llm_test_cache.json5",
        help="缓存文件路径 (默认: ~/.akg/llm_cache/llm_test_cache.json5)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="显示详细信息（包括完整消息内容、结果等）"
    )

    args = parser.parse_args()

    # 检查是否为内存模式（默认文件路径）
    if args.cache_file == "~/.akg/llm_cache/llm_test_cache.json5":
        view_live_cache(args.cache_file, show_details=args.verbose)
    else:
        view_live_cache(args.cache_file, show_details=args.verbose)


if __name__ == "__main__":
    main()
