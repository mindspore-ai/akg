#!/usr/bin/env python3
"""
查看 KernelAgent 测试日志的辅助工具

使用方法:
    python view_test_logs.py [log_file]
    
如果不指定 log_file，将显示最新的日志文件
"""

import sys
import json
from pathlib import Path
from datetime import datetime


def format_timestamp(ts_str):
    """格式化时间戳"""
    try:
        dt = datetime.fromisoformat(ts_str)
        return dt.strftime("%H:%M:%S")
    except:
        return ts_str


def print_entry(entry, show_full=False):
    """打印单条日志"""
    timestamp = format_timestamp(entry.get("timestamp", ""))
    round_num = entry.get("round", 0)
    entry_type = entry.get("type", "unknown")
    data = entry.get("data", {})
    
    # 根据类型选择显示内容
    if entry_type == "user_input":
        user_input = data.get("input", "")
        print(f"[{timestamp}] Round {round_num} | 👤 用户输入: {user_input[:80]}")
        if show_full and len(user_input) > 80:
            print(f"   完整输入: {user_input}")
    
    elif entry_type == "llm_request":
        prompt_len = data.get("prompt_length", 0)
        model = data.get("model_level", "")
        print(f"[{timestamp}] Round {round_num} | 🤖 LLM 请求: model={model}, prompt_len={prompt_len}")
        if show_full:
            preview = data.get("prompt_preview", "")
            print(f"   预览: {preview}")
    
    elif entry_type == "llm_response":
        resp_len = data.get("response_length", 0)
        print(f"[{timestamp}] Round {round_num} | 💬 LLM 响应: length={resp_len}")
        if show_full:
            preview = data.get("response_preview", "")
            print(f"   预览: {preview}")
    
    elif entry_type == "tool_call":
        tool_name = data.get("tool_name", "")
        result_status = data.get("result", {}).get("status", "")
        duration = data.get("duration_ms", 0)
        print(f"[{timestamp}] Round {round_num} | 🔧 工具调用: {tool_name} → {result_status} ({duration}ms)")
        if show_full:
            print(f"   参数: {json.dumps(data.get('arguments', {}), ensure_ascii=False, indent=2)}")
            print(f"   结果: {json.dumps(data.get('result', {}), ensure_ascii=False, indent=2)}")
    
    elif entry_type == "result":
        status = data.get("status", "")
        has_error = data.get("has_error", False)
        print(f"[{timestamp}] Round {round_num} | 📊 执行结果: {status} {'❌' if has_error else '✅'}")
    
    elif entry_type == "initialization":
        total = data.get("total_tools", 0)
        agent_count = len(data.get("agent_tools", []))
        print(f"[{timestamp}] 🔧 初始化: {total} 个工具 (Agent: {agent_count})")
    
    else:
        print(f"[{timestamp}] Round {round_num} | {entry_type}: {data}")


def view_log(log_file, show_full=False, filter_type=None, filter_round=None):
    """查看日志文件"""
    log_path = Path(log_file)
    
    if not log_path.exists():
        print(f"❌ 日志文件不存在: {log_file}")
        return
    
    print(f"\n📝 查看日志: {log_file}")
    print("="*80)
    
    entries = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
                entries.append(entry)
            except Exception as e:
                print(f"⚠️  解析失败: {e}")
    
    # 应用过滤
    if filter_type:
        entries = [e for e in entries if e.get("type") == filter_type]
    if filter_round is not None:
        entries = [e for e in entries if e.get("round") == filter_round]
    
    print(f"共 {len(entries)} 条记录\n")
    
    for entry in entries:
        print_entry(entry, show_full=show_full)
    
    print("\n" + "="*80)


def find_latest_log():
    """查找最新的日志文件"""
    log_dir = Path.home() / ".aikg" / "logs"
    if not log_dir.exists():
        return None
    
    log_files = sorted(log_dir.glob("kernel_agent_test_*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    return log_files[0] if log_files else None


def main():
    import argparse
    parser = argparse.ArgumentParser(description="查看 KernelAgent 测试日志")
    parser.add_argument("log_file", nargs="?", help="日志文件路径（可选）")
    parser.add_argument("-f", "--full", action="store_true", help="显示完整信息")
    parser.add_argument("-t", "--type", help="过滤特定类型 (user_input, llm_request, llm_response, tool_call, result)")
    parser.add_argument("-r", "--round", type=int, help="过滤特定轮次")
    
    args = parser.parse_args()
    
    log_file = args.log_file
    if not log_file:
        log_file = find_latest_log()
        if not log_file:
            print("❌ 未找到日志文件")
            print(f"日志目录: {Path.home() / '.aikg' / 'logs'}")
            return
        print(f"📝 使用最新日志文件")
    
    view_log(log_file, show_full=args.full, filter_type=args.type, filter_round=args.round)


if __name__ == "__main__":
    main()
