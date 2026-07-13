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

"""opencode adapter — the JS↔Python door.

opencode plugins are JavaScript; the autoresearch brain (``decide``) is
Python. This 15-line process is the only crossing point: the JS plugin
spawns it, hands over a normalised event, and reads back a normalised
``Decision``. It contains NO logic of its own — it is the exact opencode-
side analogue of ``.claude/hooks/cc_hook.py`` (which is the Claude-side
crossing), and both call the SAME ``scripts/decide.py``.

Invocation (argv carries a base64-encoded JSON event, so no stdin piping
or shell-escaping is needed from Bun's $):

    python door.py event <base64(json AgentEvent)>

Output (stdout): a JSON Decision:
    {"block": bool, "block_reason": str, "status": [str],
     "context": str|null, "todos_header": str|null, "todos": [..]|null}

The session id travels inside the event; the door exports it as
AR_SESSION_ID so ``decide`` -> phase_machine resolves task ownership via
the same per-session index Claude Code uses. The door never raises into
the plugin: on any internal error it returns a safe allow-Decision so a
broken hook can never wedge the agent.
"""
import base64
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)        # <project>/.opencode -> <project>
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "scripts"))


def _safe_allow() -> str:
    return json.dumps({
        "block": False, "block_reason": "", "status": [],
        "context": None, "todos_header": None, "todos": None,
    })


def _handle_event(payload: dict) -> str:
    # Ownership resolves through the same seam as Claude Code: set the
    # neutral AR_SESSION_ID before decide() reads get_task_dir().
    os.environ["AR_SESSION_ID"] = str(payload.get("session_id", "") or "")

    from decide import AgentEvent, decide
    ev = AgentEvent(
        kind=str(payload.get("kind", "")),
        tool_kind=str(payload.get("tool_kind", "") or ""),
        tool=str(payload.get("tool", "")),
        command=str(payload.get("command", "") or ""),
        file_path=str(payload.get("file_path", "") or ""),
        subagent_type=str(payload.get("subagent_type", "") or ""),
        output=str(payload.get("output", "") or ""),
        stop_reason=str(payload.get("stop_reason", "unknown") or "unknown"),
        session_id=str(payload.get("session_id", "") or ""),
    )
    d = decide(ev)
    return json.dumps({
        "block": d.block,
        "block_reason": d.block_reason,
        "status": list(d.status or []),
        "context": d.context,
        "todos_header": d.todos_header,
        "todos": d.todos,
    })


def main() -> None:
    try:
        mode = sys.argv[1] if len(sys.argv) > 1 else ""
        if mode != "event":
            print(_safe_allow())
            return
        raw = base64.b64decode(sys.argv[2]).decode("utf-8") if len(sys.argv) > 2 else "{}"
        print(_handle_event(json.loads(raw)))
    except Exception as e:  # never propagate into the plugin
        sys.stderr.write(f"[ar-door] internal error (allowing): {e}\n")
        print(_safe_allow())


if __name__ == "__main__":
    main()
