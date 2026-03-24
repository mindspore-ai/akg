#!/usr/bin/env python3
"""对话历史预处理：解析 agent-transcripts JSONL，压缩并切分为滑动窗口。

纯 Python 标准库，无外部依赖，Python 3.8+。

用法:
  python preprocess.py <input_path> [选项]

  input_path    单个 .jsonl 文件，或包含 .jsonl 文件的目录
  --output-dir  输出目录（默认 ~/.akg/dev_insight_workdir/{timestamp}/）
  --max-chars   每窗口字符上限（默认 40000）
  --overlap     相邻窗口重叠 Turn 数（默认 2）

Cursor 的 agent-transcripts 格式特征：
  - 每个 user 消息后跟 N 条 assistant 消息
  - 前 N-1 条 assistant 是中间过程（工具调用、推理）
  - 最后一条 assistant 是给用户的实际回复
  因此 assistant 侧只取最后一条（短于 200 字时额外保留最长的一条）。
"""

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ==================== 数据模型 ====================


@dataclass
class Turn:
    index: int
    user_content: str
    assistant_content: str
    char_count: int = 0

    def __post_init__(self):
        self.char_count = len(self.user_content) + len(self.assistant_content)


@dataclass
class Window:
    window_id: int
    turns: List[Dict]
    turn_range: Tuple[int, int] = (0, 0)
    char_count: int = 0


# ==================== JSONL 解析 ====================


def parse_jsonl(filepath: str) -> List[Turn]:
    """解析 agent-transcripts JSONL 文件为 Turn 列表。"""
    messages: List[Tuple[str, str]] = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            role = obj.get("role", "")
            text = _extract_text(obj)
            if text and role in ("user", "assistant"):
                messages.append((role, text))

    return _pair_turns(messages)


def _extract_text(obj: dict) -> str:
    """从 JSONL 消息对象中提取文本。"""
    msg = obj.get("message", {})
    content = msg.get("content", [])
    parts = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "text":
            parts.append(item.get("text", ""))
        elif isinstance(item, str):
            parts.append(item)
    return "\n".join(parts)


# ==================== Turn 配对 ====================

_FALLBACK_THRESHOLD = 200


def _pair_turns(messages: List[Tuple[str, str]]) -> List[Turn]:
    """将 (role, text) 消息序列配对为 Turn 列表。

    连续 assistant 消息的处理策略：
      - 只保留最后一条（通常是给用户的实际回复）
      - 如果最后一条很短（< 200 chars），额外拼入最长的那条作为补充

    跳过没有 assistant 回复的 user 消息（Cursor 的 summary/replay 产生的重复）。
    用 user 内容前 200 字符作为指纹进行去重。
    """
    turns: List[Turn] = []
    seen_user_fingerprints: set = set()
    i = 0
    turn_idx = 0

    while i < len(messages):
        role, text = messages[i]

        if role == "user":
            user_text = compress_user(text)
            i += 1

            asst_raw: List[str] = []
            while i < len(messages) and messages[i][0] == "assistant":
                asst_raw.append(messages[i][1])
                i += 1

            if not asst_raw:
                continue

            assistant_text = _select_assistant_content(asst_raw)

            fingerprint = user_text.strip()[:200]
            if fingerprint in seen_user_fingerprints:
                continue
            seen_user_fingerprints.add(fingerprint)

            if user_text.strip() or assistant_text.strip():
                turns.append(Turn(
                    index=turn_idx,
                    user_content=user_text.strip(),
                    assistant_content=assistant_text.strip(),
                ))
                turn_idx += 1
        else:
            i += 1

    return turns


def _select_assistant_content(asst_messages: List[str]) -> str:
    """从一组连续 assistant 消息中选出有效内容。

    策略：
      1. 只有 1 条 → 直接用
      2. 多条时取最后一条（给用户的实际回复）
      3. 如果最后一条 < 200 chars，额外拼入最长的一条
    """
    if not asst_messages:
        return ""

    if len(asst_messages) == 1:
        return asst_messages[0]

    last = asst_messages[-1]
    longest_idx = max(range(len(asst_messages)), key=lambda k: len(asst_messages[k]))
    longest = asst_messages[longest_idx]

    if len(last) >= _FALLBACK_THRESHOLD:
        return last

    if longest_idx == len(asst_messages) - 1:
        return last

    return longest + "\n\n---\n\n" + last


# ==================== User 消息压缩 ====================


def compress_user(text: str) -> str:
    """压缩 user 消息：提取 user_query，简化 code_selection，丢弃噪音。"""
    text = _drop_tag_content(text, "system_reminder")

    query_match = re.search(
        r"<user_query>\s*(.*?)\s*</user_query>",
        text,
        re.DOTALL,
    )

    code_selections = re.findall(
        r'<code_selection\s+path="([^"]+)"\s+lines="([^"]+)"[^>]*>',
        text,
    )
    sel_summary = ""
    if code_selections:
        refs = [f"  - {path} (lines {lines})" for path, lines in code_selections]
        sel_summary = "\n[引用代码]\n" + "\n".join(refs)

    if query_match:
        return query_match.group(1).strip() + sel_summary

    cleaned = _drop_tag_content(text, "attached_files")
    cleaned = _drop_tag_content(cleaned, "open_and_recently_viewed_files")
    cleaned = _drop_tag_content(cleaned, "user_info")
    cleaned = _drop_tag_content(cleaned, "agent_skills")
    cleaned = _drop_tag_content(cleaned, "git_status")
    cleaned = _drop_tag_content(cleaned, "agent_transcripts")
    cleaned = re.sub(r"<[^>]+>", "", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()

    return cleaned + sel_summary


def _drop_tag_content(text: str, tag: str) -> str:
    return re.sub(
        rf"<{tag}>.*?</{tag}>",
        "",
        text,
        flags=re.DOTALL,
    )


# ==================== 滑动窗口 ====================

_DEFAULT_MAX_CHARS = 40000


def create_windows(
    turns: List[Turn],
    max_chars: int = _DEFAULT_MAX_CHARS,
    overlap: int = 2,
) -> List[Window]:
    """按字符数上限切分窗口，相邻窗口重叠 overlap 个 turn。

    策略：从当前起始位置开始累加 turn，直到总字符数超过 max_chars 则截断为一个窗口。
    下一个窗口从 (当前窗口末尾 - overlap) 位置开始。
    每个窗口至少包含 1 个 turn（即使单个 turn 超过 max_chars）。
    """
    if not turns:
        return []

    windows = []
    start = 0

    while start < len(turns):
        char_sum = 0
        end = start

        while end < len(turns):
            next_chars = turns[end].char_count
            if char_sum + next_chars > max_chars and end > start:
                break
            char_sum += next_chars
            end += 1

        window_turns = turns[start:end]
        windows.append(Window(
            window_id=len(windows),
            turns=[asdict(t) for t in window_turns],
            turn_range=(window_turns[0].index, window_turns[-1].index),
            char_count=char_sum,
        ))

        if end >= len(turns):
            break

        next_start = end - overlap
        if next_start <= start:
            next_start = end
        start = next_start

    return windows


# ==================== 输出 ====================


def write_output(
    windows: List[Window],
    turns: List[Turn],
    source_file: str,
    output_dir: str,
    original_size: int,
    config: dict,
):
    os.makedirs(output_dir, exist_ok=True)

    for w in windows:
        window_path = os.path.join(output_dir, f"window_{w.window_id}.json")
        with open(window_path, "w", encoding="utf-8") as f:
            json.dump({
                "window_id": w.window_id,
                "turns": w.turns,
            }, f, ensure_ascii=False, indent=2)

    compressed_size = sum(t.char_count for t in turns)
    ratio = f"{original_size / compressed_size:.1f}x" if compressed_size > 0 else "N/A"

    title = Path(source_file).stem
    if title and len(title) > 36:
        title = title[37:] if "-" in title[30:40] else title

    manifest = {
        "source_file": os.path.basename(source_file),
        "source_format": "jsonl",
        "conversation_title": title,
        "total_turns": len(turns),
        "total_windows": len(windows),
        "config": config,
        "compression_stats": {
            "original_bytes": original_size,
            "compressed_chars": compressed_size,
            "compression_ratio": ratio,
        },
        "windows": [
            {
                "id": w.window_id,
                "file": f"window_{w.window_id}.json",
                "turn_range": list(w.turn_range),
                "char_count": w.char_count,
            }
            for w in windows
        ],
    }

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return manifest_path


# ==================== 主函数 ====================


def process_single_file(
    filepath: str,
    output_dir: str,
    max_chars: int,
    overlap: int,
) -> Optional[str]:
    """处理单个 JSONL 文件，返回 manifest.json 路径。"""
    if not filepath.endswith(".jsonl"):
        print(f"[WARN] 不是 .jsonl 文件，跳过: {filepath}", file=sys.stderr)
        return None

    original_size = os.path.getsize(filepath)
    turns = parse_jsonl(filepath)

    if not turns:
        print(f"[WARN] 未解析到有效 Turn: {filepath}", file=sys.stderr)
        return None

    if len(turns) < 3:
        print(f"[WARN] Turn 数过少 ({len(turns)})，跳过: {filepath}", file=sys.stderr)
        return None

    windows = create_windows(turns, max_chars, overlap)
    if not windows:
        print(f"[WARN] 未生成有效窗口: {filepath}", file=sys.stderr)
        return None

    config = {"max_chars": max_chars, "overlap": overlap}
    manifest_path = write_output(
        windows, turns, filepath, output_dir, original_size, config,
    )

    compressed_chars = sum(t.char_count for t in turns)
    print(
        f"[OK] {os.path.basename(filepath)}: "
        f"{len(turns)} turns, {len(windows)} windows, "
        f"{original_size} -> {compressed_chars} chars "
        f"({original_size / compressed_chars:.1f}x compression)"
        f"\nOutput directory: {output_dir}"
    )
    return manifest_path


def main():
    parser = argparse.ArgumentParser(
        description="对话历史预处理：解析 agent-transcripts JSONL，压缩并切分为滑动窗口",
    )
    parser.add_argument("input_path", help="单个 .jsonl 文件或包含 .jsonl 的目录")
    parser.add_argument("--output-dir", default="", help="输出目录")
    parser.add_argument("--max-chars", type=int, default=_DEFAULT_MAX_CHARS,
                        help=f"每窗口字符上限（默认 {_DEFAULT_MAX_CHARS}）")
    parser.add_argument("--overlap", type=int, default=2,
                        help="相邻窗口重叠 Turn 数（默认 2）")
    args = parser.parse_args()

    if not args.output_dir:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = str(
            Path.home() / ".akg" / "dev_insight_workdir" / ts
        )

    input_path = args.input_path

    if os.path.isfile(input_path):
        process_single_file(
            input_path, args.output_dir,
            args.max_chars, args.overlap,
        )
    elif os.path.isdir(input_path):
        files = []
        for root, _, fnames in os.walk(input_path):
            for fn in fnames:
                if fn.endswith(".jsonl"):
                    files.append(os.path.join(root, fn))

        if not files:
            print(f"[ERROR] 目录中未找到 .jsonl 文件: {input_path}",
                  file=sys.stderr)
            sys.exit(1)

        manifests = []
        for fp in sorted(files):
            stem = Path(fp).stem
            sub_dir = os.path.join(args.output_dir, stem)
            result = process_single_file(
                fp, sub_dir, args.max_chars, args.overlap,
            )
            if result:
                manifests.append({"file": os.path.basename(fp), "manifest": result})

        batch_path = os.path.join(args.output_dir, "batch_manifest.json")
        with open(batch_path, "w", encoding="utf-8") as f:
            json.dump({"files": manifests, "total": len(manifests)},
                      f, ensure_ascii=False, indent=2)
        print(f"\n[DONE] 共处理 {len(manifests)}/{len(files)} 个文件")
        print(f"       输出目录: {args.output_dir}")
    else:
        print(f"[ERROR] 路径不存在: {input_path}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
