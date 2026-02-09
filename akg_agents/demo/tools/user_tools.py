"""
用户交互工具 - ask_user
"""
from typing import Dict, Any
from .registry import ToolRegistry


def ask_user(args: Dict[str, Any]) -> Dict[str, Any]:
    """向用户提问并获取回复"""
    question = args.get("question", "")
    try:
        print(f"\n{'='*60}")
        print(f"  Agent 提问: {question}")
        print(f"{'='*60}")
        reply = input("  你的回复 > ").strip()
        return {"status": "success", "output": reply, "error": ""}
    except (KeyboardInterrupt, EOFError):
        return {"status": "error", "output": "", "error": "用户取消输入"}


ToolRegistry.register(
    "ask_user",
    "向用户提问以获取确认或补充信息。当你不确定用户意图、缺少必要信息时使用。",
    {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "要向用户提出的问题"},
        },
        "required": ["question"],
    },
    ask_user,
)
