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

"""
Shared utilities for Claude Code hook scripts.
"""
import json
import re
import sys


def block_decision(reason: str):
    """Emit a PreToolUse block decision and exit.

    Wire format `{"decision": "block", "reason": ...}` is what Claude Code's
    hook framework expects to abort the in-flight tool call. Exit code 2
    means "hook ran successfully and reached a block verdict"; 0 means
    proceed, non-zero non-2 means the hook itself errored. Single helper
    so hooks/guard_bash and hooks/guard_edit can't drift on the protocol.
    """
    print(json.dumps({"decision": "block", "reason": reason}))
    sys.exit(2)


def block_with_guidance(task_dir: str, reason: str):
    """Block with a `[AR] {reason}. <fresh phase guidance>` message.

    Used by every guard that wants the LLM to read both the specific
    rejection reason AND the current phase's recovery instructions in one
    message. Imported lazily because phase_machine -> hooks/utils would
    create a cycle on package init.
    """
    from phase_machine import get_guidance
    block_decision(f"[AR] {reason}. {get_guidance(task_dir)}")


# Path field names by tool. Edit / Write / MultiEdit all use `file_path`;
# NotebookEdit uses `notebook_path`. Reading only `file_path` (the
# original implementation) silently let NotebookEdit hits skip the hook
# entirely — the matcher caught the tool name but the empty path made
# main() fall to `sys.exit(0)`. Centralized here so both guard and post
# hooks agree on the extraction.
_TOOL_PATH_FIELDS = ("file_path", "notebook_path")


def extract_target_path(hook_input: dict) -> str:
    """Pull the target file path out of a hook payload regardless of
    whether the tool is Edit / Write / MultiEdit (file_path) or
    NotebookEdit (notebook_path). Returns '' if neither is set —
    callers should treat that as 'nothing to gate' and exit 0.
    """
    tool_input = hook_input.get("tool_input", {}) or {}
    for field in _TOOL_PATH_FIELDS:
        v = tool_input.get(field)
        if v:
            return v
    return ""


def read_hook_input() -> dict:
    """Read and parse hook input from stdin.

    Handles Windows paths with unescaped backslashes in JSON
    (e.g., C:\\Users becomes C:\\\\Users before JSON parsing).
    """
    raw = sys.stdin.read()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Fix unescaped backslashes in Windows paths
        fixed = re.sub(r'(?<!\\)\\(?![\\"/bfnrtu])', r'\\\\', raw)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            return {}


def emit_status(msg: str):
    """Print a human-readable hook status line to stderr."""
    print(msg, file=sys.stderr)


def emit_todowrite_context(task_dir: str, header: str):
    """Print PostToolUse `hookSpecificOutput.additionalContext` JSON that
    instructs Claude to mirror plan.md into TodoWrite on the next turn.

    Only pending + in_progress items from the CURRENT plan are projected.
    Settled items (KEEP / DISCARD / FAIL) live in plan.md's Settled History
    table and history.jsonl — they are the durable audit trail, not part of
    the live TodoWrite queue. This caps the TodoWrite list at
    `items_per_plan` entries (typically 3-5) regardless of how many REPLAN
    cycles have happened.

    plan.md is the source of truth; TodoWrite is a UI mirror of current work.

    Emits even when no live items remain — an empty `{"todos": []}` payload
    explicitly clears the UI. Without the explicit clear, the model's
    TodoWrite UI keeps the prior plan's last in_progress entry through the
    non-strict REPLACE at create_plan time.
    """
    from phase_machine import get_plan_items
    live = [it for it in get_plan_items(task_dir) if not it["done"]]

    todos = []
    for it in live:
        status = "in_progress" if it["active"] else "pending"
        todos.append({
            "content": f"[{it['id']}] {it['description'][:80]}",
            "activeForm": f"Working on {it['id']}: {it['description'][:60]}",
            "status": status,
        })
    context = (
        f"{header}\n"
        f"Required action: call TodoWrite NOW with the exact list below. "
        f"This REPLACES any existing todos — do NOT merge, append, or "
        f"preserve older entries. plan.md is the source of truth; TodoWrite "
        f"is a UI mirror of current live work only (completed items live in "
        f"plan.md's Settled History). Pass this payload verbatim.\n"
        f"TodoWrite payload:\n{json.dumps({'todos': todos}, ensure_ascii=False)}"
    )
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PostToolUse",
            "additionalContext": context,
        }
    }))
