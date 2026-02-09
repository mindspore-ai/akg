#!/usr/bin/env python3
"""
查看 KernelAgent 会话日志

使用方法:
    # 查看最新会话的可读日志
    python view_test_logs.py --latest
    
    # 查看指定会话目录
    python view_test_logs.py /path/to/session_dir
    
    # 查看指定会话的 events.jsonl（紧凑索引）
    python view_test_logs.py /path/to/session_dir --events
    
    # 查看某一轮的 prompt / response
    python view_test_logs.py /path/to/session_dir --round 1 --prompts
    
    # 查看某一轮的工具调用详情
    python view_test_logs.py /path/to/session_dir --round 1 --tools
    
    # 兼容旧格式: 直接查看 .jsonl 文件
    python view_test_logs.py /path/to/file.jsonl
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime


# ==================== 新格式：结构化会话目录 ====================

def view_session_log(session_dir: Path):
    """直接打印 session.log（人类可读时间线）"""
    log_file = session_dir / "session.log"
    if not log_file.exists():
        print(f"未找到 session.log: {log_file}")
        return
    print(log_file.read_text(encoding="utf-8"))


def view_events(session_dir: Path, filter_round: int = None, filter_type: str = None):
    """查看 events.jsonl（紧凑事件索引）"""
    events_file = session_dir / "events.jsonl"
    if not events_file.exists():
        print(f"未找到 events.jsonl: {events_file}")
        return
    
    print(f"\n[事件索引] {events_file}\n")
    
    with open(events_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            if filter_round is not None and event.get("round") != filter_round:
                continue
            if filter_type and event.get("type") != filter_type:
                continue
            
            # 格式化输出
            ts = event.get("ts", "")
            etype = event.get("type", "?")
            rnd = event.get("round", "")
            rnd_str = f"R{rnd:02d}" if isinstance(rnd, int) else ""
            
            if etype == "init":
                tools = event.get("tools", {})
                print(f"[{ts}]       INIT  tools={tools.get('total', 0)} task={event.get('task_id', '')}")
            elif etype == "user_input":
                print(f"[{ts}] {rnd_str}  INPUT {event.get('input', '')[:80]}")
            elif etype == "llm_request":
                print(f"[{ts}] {rnd_str}  LLM-> {event.get('prompt_len', 0)} chars "
                      f"model={event.get('model', '')} file={event.get('prompt_file', '')}")
            elif etype == "llm_response":
                print(f"[{ts}] {rnd_str}  <-LLM {event.get('response_len', 0)} chars "
                      f"file={event.get('response_file', '')}")
            elif etype == "tool_call":
                print(f"[{ts}] {rnd_str}  TOOL  {event.get('tool', '')} -> "
                      f"{event.get('status', '?')} "
                      f"({event.get('duration_ms', 0):.0f}ms) "
                      f"file={event.get('detail_file', '')}")
            elif etype == "result":
                print(f"[{ts}] {rnd_str}  RESULT {event.get('status', '?')} "
                      f"output_len={event.get('output_len', 0)}")
            else:
                print(f"[{ts}] {rnd_str}  {etype}  {json.dumps(event, ensure_ascii=False)[:100]}")
    
    print()


def view_prompts(session_dir: Path, filter_round: int = None):
    """查看 prompt 文件"""
    prompts_dir = session_dir / "prompts"
    if not prompts_dir.exists():
        print("未找到 prompts 目录")
        return
    
    files = sorted(prompts_dir.glob("*.txt"))
    if filter_round is not None:
        prefix = f"R{filter_round:02d}_"
        files = [f for f in files if f.name.startswith(prefix)]
    
    for f in files:
        content = f.read_text(encoding="utf-8")
        print(f"\n{'=' * 60}")
        print(f"  Prompt: {f.name} ({len(content)} chars)")
        print(f"{'=' * 60}")
        print(content[:5000])
        if len(content) > 5000:
            print(f"\n... (截断，共 {len(content)} 字符)")
        print()


def view_responses(session_dir: Path, filter_round: int = None):
    """查看 response 文件"""
    resp_dir = session_dir / "responses"
    if not resp_dir.exists():
        print("未找到 responses 目录")
        return
    
    files = sorted(resp_dir.glob("*.txt"))
    if filter_round is not None:
        prefix = f"R{filter_round:02d}_"
        files = [f for f in files if f.name.startswith(prefix)]
    
    for f in files:
        content = f.read_text(encoding="utf-8")
        print(f"\n{'=' * 60}")
        print(f"  Response: {f.name} ({len(content)} chars)")
        print(f"{'=' * 60}")
        print(content[:5000])
        if len(content) > 5000:
            print(f"\n... (截断，共 {len(content)} 字符)")
        print()


def view_tool_calls(session_dir: Path, filter_round: int = None):
    """查看工具调用详情"""
    tools_dir = session_dir / "tool_calls"
    if not tools_dir.exists():
        print("未找到 tool_calls 目录")
        return
    
    files = sorted(tools_dir.glob("*.json"))
    if filter_round is not None:
        prefix = f"R{filter_round:02d}_"
        files = [f for f in files if f.name.startswith(prefix)]
    
    for f in files:
        content = f.read_text(encoding="utf-8")
        print(f"\n{'─' * 40}")
        print(f"  Tool Call: {f.name}")
        print(f"{'─' * 40}")
        # 已经是 pretty-printed JSON，直接输出
        print(content)


# ==================== 兼容旧格式：单个 .jsonl ====================

def view_legacy_jsonl(log_file: Path, show_full: bool = False,
                      filter_type: str = None, filter_round: int = None):
    """查看旧格式 JSONL"""
    print(f"\n[查看旧格式日志] {log_file}\n")
    
    entries = []
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    
    if filter_type:
        entries = [e for e in entries if e.get("type") == filter_type]
    if filter_round is not None:
        entries = [e for e in entries if e.get("round") == filter_round]
    
    print(f"共 {len(entries)} 条记录\n")
    
    for entry in entries:
        ts = entry.get("timestamp", "")
        try:
            ts = datetime.fromisoformat(ts).strftime("%H:%M:%S")
        except Exception:
            pass
        
        rnd = entry.get("round", 0)
        etype = entry.get("type", "?")
        data = entry.get("data", {})
        
        if etype == "user_input":
            print(f"[{ts}] R{rnd:02d} INPUT: {data.get('input', '')[:80]}")
        elif etype == "llm_request":
            print(f"[{ts}] R{rnd:02d} LLM-> {data.get('prompt_length', 0)} chars")
        elif etype == "llm_response":
            print(f"[{ts}] R{rnd:02d} <-LLM {data.get('response_length', 0)} chars")
        elif etype == "tool_call":
            print(f"[{ts}] R{rnd:02d} TOOL: {data.get('tool_name', '?')} -> "
                  f"{data.get('result', {}).get('status', '?')}")
        elif etype == "result":
            print(f"[{ts}] R{rnd:02d} RESULT: {data.get('status', '?')}")
        else:
            print(f"[{ts}] R{rnd:02d} {etype}")


# ==================== 查找最新会话 ====================

def find_latest_session() -> Path:
    """查找最新的会话目录"""
    log_dir = Path.home() / ".akg" / "logs"
    
    # 先尝试 latest 指针
    latest_file = log_dir / "latest_session.txt"
    if latest_file.exists():
        target = Path(latest_file.read_text(encoding="utf-8").strip())
        if target.exists():
            return target
    
    # 搜索最新的会话目录
    sessions_dir = log_dir / "sessions"
    if sessions_dir.exists():
        # 按修改时间倒序查找
        session_dirs = sorted(
            [d for d in sessions_dir.rglob("session.log")],
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        if session_dirs:
            return session_dirs[0].parent
    
    # 兼容旧格式：查找 .jsonl 文件
    jsonl_files = sorted(
        log_dir.rglob("*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    if jsonl_files:
        return jsonl_files[0]
    
    return None


# ==================== 主入口 ====================

def main():
    parser = argparse.ArgumentParser(
        description="查看 KernelAgent 会话日志",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s --latest              查看最新会话的可读日志
  %(prog)s --latest --events     查看最新会话的事件索引
  %(prog)s --latest -r 1 -p      查看第 1 轮的 prompt
  %(prog)s --latest -r 1 -t      查看第 1 轮的工具调用详情
  %(prog)s /path/to/session      查看指定会话
  %(prog)s /path/to/file.jsonl   兼容查看旧格式日志
        """
    )
    parser.add_argument("path", nargs="?", help="会话目录或 .jsonl 文件路径")
    parser.add_argument("--latest", action="store_true", help="使用最新会话")
    parser.add_argument("--events", "-e", action="store_true", help="显示事件索引")
    parser.add_argument("--prompts", "-p", action="store_true", help="显示 prompt 文件内容")
    parser.add_argument("--responses", action="store_true", help="显示 response 文件内容")
    parser.add_argument("--tools", "-t", action="store_true", help="显示工具调用详情")
    parser.add_argument("--round", "-r", type=int, help="过滤特定轮次")
    parser.add_argument("--full", "-f", action="store_true", help="显示完整信息（旧格式）")
    parser.add_argument("--type", help="过滤事件类型")
    
    args = parser.parse_args()
    
    # 确定目标路径
    target = None
    if args.path:
        target = Path(args.path)
    elif args.latest:
        target = find_latest_session()
        if not target:
            print("未找到任何会话日志")
            print(f"日志目录: {Path.home() / '.akg' / 'logs'}")
            return
        print(f"[最新会话] {target}\n")
    else:
        # 默认查看最新
        target = find_latest_session()
        if not target:
            print("未找到任何会话日志")
            print(f"日志目录: {Path.home() / '.akg' / 'logs'}")
            parser.print_help()
            return
        print(f"[最新会话] {target}\n")
    
    if not target.exists():
        print(f"路径不存在: {target}")
        return
    
    # 判断是旧格式 .jsonl 还是新格式目录
    if target.is_file() and target.suffix == ".jsonl":
        view_legacy_jsonl(target, show_full=args.full,
                         filter_type=args.type, filter_round=args.round)
        return
    
    # 新格式：会话目录
    if not target.is_dir():
        print(f"路径不是目录: {target}")
        return
    
    # 根据参数选择显示内容
    showed_something = False
    
    if args.events:
        view_events(target, filter_round=args.round, filter_type=args.type)
        showed_something = True
    
    if args.prompts:
        view_prompts(target, filter_round=args.round)
        showed_something = True
    
    if args.responses:
        view_responses(target, filter_round=args.round)
        showed_something = True
    
    if args.tools:
        view_tool_calls(target, filter_round=args.round)
        showed_something = True
    
    if not showed_something:
        # 默认显示 session.log
        view_session_log(target)


if __name__ == "__main__":
    main()
