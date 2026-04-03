"""
Context compression helpers for the agent conversation.

Three layers:
  1. microcompact — clears old tool_result content (cheap, every turn)
  2. auto_compact — LLM summarization when token threshold exceeded
  3. compact tool — agent manually triggers auto_compact
"""

import json
import os
import time


def estimate_tokens(messages: list, chars_per_token: int = 4) -> int:
    """Rough token estimate based on JSON serialization length."""
    return len(json.dumps(messages, default=str)) // chars_per_token


def microcompact(messages: list, min_chars: int = 200, keep_recent: int = 3):
    """
    Replace old tool_result content with "[cleared]" to save tokens.
    Keeps the most recent *keep_recent* tool_result blocks intact.
    """
    tool_results = []
    for msg in messages:
        if msg["role"] == "user" and isinstance(msg.get("content"), list):
            for part in msg["content"]:
                if isinstance(part, dict) and part.get("type") == "tool_result":
                    tool_results.append(part)
    if len(tool_results) <= keep_recent:
        return
    for part in tool_results[:-keep_recent]:
        if isinstance(part.get("content"), str) and len(part["content"]) > min_chars:
            part["content"] = "[cleared]"


def _save_transcript(messages: list, task_dir: str, session_dir: str = "agent_session",
                     filename: str = "transcript_latest.jsonl",
                     mode: str = "w"):
    """Save messages to a transcript file.

    mode="w": overwrite (for transcript_latest.jsonl — current messages state).
    mode="a": append (for transcript_full.jsonl — cumulative history).
    """
    transcript_dir = os.path.join(task_dir, session_dir, "transcripts")
    os.makedirs(transcript_dir, exist_ok=True)
    path = os.path.join(transcript_dir, filename)
    with open(path, mode, encoding="utf-8") as f:
        for msg in messages:
            f.write(json.dumps(msg, default=str) + "\n")


async def _llm_summarize(client, model: str, provider: str,
                         prompt: str, max_tokens: int) -> str:
    """调用 LLM 生成摘要。"""
    if provider == "openai":
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or "(no summary)"
    else:
        resp = await client.messages.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        return resp.content[0].text


_COMPACT_PROMPT = (
    "Summarize this optimization session for continuity. "
    "You MUST preserve:\n"
    "1. Performance ranking: ALL attempted approaches sorted by metric, "
    "including their round number, metric value, and KEEP/DISCARD status. "
    "Do NOT only keep the best — the full ranking shows which directions "
    "are close to optimal and worth combining.\n"
    "2. Discarded attempts: distinguish between "
    "(a) crashes/correctness errors (must avoid) and "
    "(b) no improvement (may revisit in combination with other changes). "
    "Include the specific error or reason for each.\n"
    "3. Current plan status and active item.\n"
    "4. Any diagnostic reports or forced direction changes.\n"
    "Format as: Performance Ranking (sorted table), "
    "Errors (crashes/correctness — must avoid), "
    "Discarded (no improvement — context for future), "
    "Current State.\n\n"
)


async def auto_compact(messages: list, client, model: str, provider: str,
                       task_dir: str, *, text_limit: int,
                       summary_max_tokens: int,
                       session_dir: str = "agent_session") -> list:
    """
    Save transcript to disk, LLM-summarize, replace messages with summary.
    Returns the new compressed messages list.
    """
    # Mark compact boundary in full history (messages already appended by loop)
    _save_transcript(
        [{"role": "system", "content": "[COMPACT] Context compressed at this point"}],
        task_dir, session_dir=session_dir,
        filename="transcript_full.jsonl", mode="a")

    conv_text = json.dumps(messages, default=str)[:text_limit]
    prompt = _COMPACT_PROMPT + conv_text

    summary = await _llm_summarize(client, model, provider, prompt, summary_max_tokens)

    return [
        {"role": "user", "content": f"[Compressed. Transcript saved.]\n{summary}"},
        {"role": "assistant", "content": "Understood. Continuing with the summary context."},
    ]
