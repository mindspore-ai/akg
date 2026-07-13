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

"""Claude Code adapter — the ENTIRE Claude-specific surface, one file,
living in Claude's own mount point (.claude/). A second agent (opencode)
gets its own adapter under .opencode/ that calls the SAME neutral
``decide`` layer in scripts/.

Claude Code's settings.json must spawn a command per hook registration, so
exactly one entry script is unavoidable; this is it. The hook *kind* is
passed as argv (``pre_tool`` | ``post_tool`` | ``stop``); the tool itself
arrives in the payload.

Responsibilities — translation only, no phase/dispatch logic:
  * read Claude's stdin payload
  * build a neutral ``AgentEvent``
  * call the agent-neutral ``decide`` layer (scripts/decide.py)
  * render the returned ``Decision`` into Claude wire bytes + exit code

settings.json wiring:
    PreToolUse  (Bash|Edit|Write|MultiEdit|NotebookEdit|Task) → cc_hook.py pre_tool
    PostToolUse (Bash|Edit|Write|MultiEdit|NotebookEdit|Task) → cc_hook.py post_tool
    Stop                                                      → cc_hook.py stop
"""
import json
import os
import re
import sys

# This adapter lives at <project>/.claude/hooks/cc_hook.py; the neutral
# decision layer lives at <project>/scripts/. Put scripts/ on sys.path.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "scripts"))
from decide import AgentEvent, Decision, decide  # noqa: E402


# --- Claude stdin payload parsing ------------------------------------------
# Edit/Write/MultiEdit use file_path; NotebookEdit uses notebook_path.
_TOOL_PATH_FIELDS = ("file_path", "notebook_path")

# Claude's native tool names -> decide()'s neutral tool taxonomy. This map is
# the Claude adapter's own business; decide() only ever sees the neutral kind.
_CC_TOOL_KIND = {
    "Bash": "shell",
    "Edit": "edit", "Write": "edit", "MultiEdit": "edit",
    "NotebookEdit": "edit",
    "Task": "subagent",
}


def _read_payload() -> dict:
    """Parse the hook payload from stdin. Tolerates Windows paths whose
    backslashes arrive unescaped in the JSON (C:\\Users → C:\\\\Users)."""
    raw = sys.stdin.read()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        fixed = re.sub(r'(?<!\\)\\(?![\\"/bfnrtu])', r'\\\\', raw)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            return {}


def _target_path(payload: dict) -> str:
    tool_input = payload.get("tool_input", {}) or {}
    for field in _TOOL_PATH_FIELDS:
        v = tool_input.get(field)
        if v:
            return v
    return ""


def _event_from_payload(kind: str, payload: dict) -> AgentEvent:
    tool_input = payload.get("tool_input", {}) or {}
    tool_name = payload.get("tool_name", "")
    return AgentEvent(
        kind=kind,
        tool_kind=_CC_TOOL_KIND.get(tool_name, ""),
        tool=tool_name,
        command=tool_input.get("command", ""),
        file_path=_target_path(payload),
        subagent_type=tool_input.get("subagent_type", "") or "",
        output=str(payload.get("tool_output", "")),
        stop_reason=payload.get("stop_reason", "unknown"),
        session_id=(os.environ.get("AR_SESSION_ID")
                    or os.environ.get("CLAUDE_CODE_SESSION_ID", "")),
    )


# --- Decision -> Claude wire ------------------------------------------------
def _render_todowrite_envelope(header: str, todos: list) -> str:
    """The PostToolUse additionalContext for mirroring the live plan into a
    todo tool. WHAT to mirror (`todos`) was decided neutrally in decide();
    this envelope is best-effort — if the model has a todo/task-list tool
    (e.g. TodoWrite) it mirrors the list, otherwise it ignores this. The
    mirror is cosmetic; plan.md is the source of truth, so a model without
    such a tool (e.g. a non-Claude model behind Claude Code) loses nothing."""
    context = (
        f"{header}\n"
        f"If you have a todo/task-list tool (e.g. TodoWrite), mirror the exact "
        f"list below with it — replace any existing entries, do not merge or "
        f"append. If you have no such tool, ignore this. plan.md is the source "
        f"of truth; the mirror reflects current live work only (completed items "
        f"live in plan.md's Settled History). Pass the payload verbatim.\n"
        f"Plan-mirror payload:\n{json.dumps({'todos': todos}, ensure_ascii=False)}"
    )
    return json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PostToolUse",
            "additionalContext": context,
        }
    })


def _emit(kind: str, d: Decision) -> None:
    """Render a Decision into Claude wire bytes, in the channel order the
    former hooks used:
      status  -> stderr (one line per print)
      block   -> stdout {"decision":"block"}; exit 2 for PreToolUse, 0 for Stop
      todos   -> stdout TodoWrite-mirror envelope (post_tool only)
      context -> stdout plain additionalContext   (post_tool only)
    """
    for line in d.status:
        print(line, file=sys.stderr)

    if d.block:
        print(json.dumps({"decision": "block", "reason": d.block_reason}))
        sys.exit(2 if kind == "pre_tool" else 0)

    if d.todos_header is not None:
        print(_render_todowrite_envelope(d.todos_header, d.todos or []))
    elif d.context is not None:
        print(json.dumps({
            "hookSpecificOutput": {
                "hookEventName": "PostToolUse",
                "additionalContext": d.context,
            }
        }))
    sys.exit(0)


def main() -> None:
    kind = sys.argv[1] if len(sys.argv) > 1 else ""
    if kind not in ("pre_tool", "post_tool", "stop"):
        # Unknown invocation → behave as a no-op allow, never crash a hook.
        sys.exit(0)
    _emit(kind, decide(_event_from_payload(kind, _read_payload())))


if __name__ == "__main__":
    main()
