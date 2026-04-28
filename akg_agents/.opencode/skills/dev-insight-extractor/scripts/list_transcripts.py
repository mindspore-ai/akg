#!/usr/bin/env python3
"""扫描指定目录下的 Cursor agent-transcripts，列出最近的对话及首/末用户消息预览。

纯 Python 标准库，无外部依赖，Python 3.8+。

用法:
  python list_transcripts.py <transcripts_dir> [选项]

  transcripts_dir  agent-transcripts 目录路径
  --top N          显示最近 N 条（默认 5）
  --json           输出 JSON 格式（供程序调用）
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path


def _extract_user_query(text: str) -> str:
    """从 user 消息文本中提取核心内容。"""
    m = re.search(r"<user_query>\s*(.*?)\s*</user_query>", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    for tag in ("attached_files", "open_and_recently_viewed_files",
                "user_info", "agent_skills", "git_status",
                "agent_transcripts", "system_reminder"):
        text = re.sub(rf"<{tag}>.*?</{tag}>", "", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", "", text)
    return text.strip()


def _scan_jsonl(filepath: Path) -> dict:
    """扫描单个 JSONL 文件，提取首条和末条 user 消息。"""
    first_user = ""
    last_user = ""

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get("role") != "user":
                    continue
                msg = obj.get("message", {})
                content = msg.get("content", [])
                text = ""
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text = item.get("text", "")
                        break
                    elif isinstance(item, str):
                        text = item
                        break
                if not text.strip():
                    continue

                query = _extract_user_query(text)
                if not query:
                    continue

                if not first_user:
                    first_user = query
                last_user = query
    except Exception:
        pass

    return {"first": first_user, "last": last_user}


def scan_transcripts(at_dir: Path, top: int = 5) -> list:
    """扫描 agent-transcripts 目录，返回按时间倒序的文件信息列表。"""
    files = []
    for sub in at_dir.iterdir():
        if not sub.is_dir():
            continue
        for f in sub.iterdir():
            if f.suffix == ".jsonl":
                mtime = os.path.getmtime(f)
                size = os.path.getsize(f)
                files.append((f, mtime, size))

    files.sort(key=lambda x: x[1], reverse=True)

    results = []
    for fp, mt, sz in files[:top]:
        queries = _scan_jsonl(fp)
        results.append({
            "path": str(fp),
            "modified": datetime.fromtimestamp(mt).strftime("%m-%d %H:%M"),
            "size_kb": sz // 1024,
            "first_query": queries["first"][:100],
            "last_query": queries["last"][:100],
        })
    return results


def main():
    parser = argparse.ArgumentParser(
        description="扫描 Cursor agent-transcripts 目录，列出最近对话及预览",
    )
    parser.add_argument("transcripts_dir",
                        help="agent-transcripts 目录路径")
    parser.add_argument("--top", type=int, default=5,
                        help="显示最近 N 条（默认 5）")
    parser.add_argument("--json", action="store_true",
                        help="输出 JSON 格式（供程序调用）")
    args = parser.parse_args()

    at_dir = Path(args.transcripts_dir)
    if not at_dir.is_dir():
        print(f"[ERROR] 目录不存在: {at_dir}", file=sys.stderr)
        sys.exit(1)

    results = scan_transcripts(at_dir, args.top)

    if not results:
        print(f"[WARN] 目录中未找到 .jsonl 对话文件: {at_dir}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        json.dump(results, sys.stdout, ensure_ascii=False, indent=2)
        print()
    else:
        print(f"目录: {at_dir}")
        print(f"共 {len(results)} 条对话（按时间倒序）：")
        print()
        for i, conv in enumerate(results):
            first_preview = conv["first_query"].replace("\n", " ")
            last_preview = conv["last_query"].replace("\n", " ")
            print(f"  [{i+1}] {conv['modified']}  {conv['size_kb']:>4}KB"
                  f"  首: \"{first_preview}\"")
            if first_preview != last_preview:
                print(f"      {' ' * 14}"
                      f"  末: \"{last_preview}\"")
            print(f"      {' ' * 14}"
                  f"  路径: {conv['path']}")
            print()


if __name__ == "__main__":
    main()
