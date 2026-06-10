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

"""Deterministic argument parser for /autoresearch.

This script is the single source of truth for the slash command's args.
It eats the user's raw `$ARGUMENTS`, decides which mode applies (resume /
scaffold / ask), validates required fields, and emits a JSON dispatch
record. The slash command tells the LLM to run only the `command` field
verbatim and to read flag values only from the `values` field — no
inventing values, no pulling defaults from docstrings, no paraphrasing.

The previous architecture handed `$ARGUMENTS` straight into the LLM as
prose context and asked it to construct the bash itself, which let the
LLM rewrite or substitute flag values on retries (e.g. quietly turning
`--devices 6` into `--devices 0` after a hook block). Putting an argparse
between the user and the LLM closes that drift.

Modes:
  resume     — `--resume [task_dir]` or a bare existing task path
  scaffold   — init flags (--ref + --kernel + --op-name + --devices)
  ask        — empty args, or scaffold flags incomplete

Output (single JSON line on stdout):
  {"mode": "scaffold|resume|ask",
   "command": "python ... (verbatim, ready to exec)" | null,
   "values":  {parsed flag dict — ground truth for the LLM},
   "missing": [human-readable required fields, ask-mode only]}

Exit code is always 0; errors surface inside the JSON. Failing on the
shell side would force the LLM to guess what happened, which is exactly
what this script exists to prevent.
"""
import json
import os
import shlex
import sys

# scaffold.py lives at scripts/ root (one level up from engine/) — make it
# importable so the lazy `from scaffold import _make_arg_parser` resolves.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.settings import default_max_rounds  # noqa: E402


def _emit(payload: dict) -> None:
    print(json.dumps(payload, ensure_ascii=False))
    sys.exit(0)


def _build_scaffold_command(args) -> str:
    """Reconstruct an exec-ready scaffold invocation from parsed args.

    Every value comes from the argparse Namespace, never from the raw
    input — that's the whole point: once argparse has accepted the args,
    the canonical form is what scaffold sees, regardless of any quoting
    or whitespace quirks in the user's typed string.
    """
    parts = ["python", "scripts/scaffold.py"]
    parts += ["--ref", shlex.quote(args.ref)]
    parts += ["--kernel", shlex.quote(args.kernel)]
    if args.op_name:
        parts += ["--op-name", shlex.quote(args.op_name)]
    if args.devices:
        parts += ["--devices", str(args.devices)]
    parts += ["--max-rounds", str(args.max_rounds)]
    parts += ["--eval-timeout", str(args.eval_timeout)]
    # shlex.quote here too: --output-dir "C:\tmp\my tasks" would
    # otherwise be re-split into "C:\tmp\my" + "tasks" when the
    # generated command line is re-parsed by bash.
    parts += ["--output-dir", shlex.quote(args.output_dir or "ar_tasks")]
    parts.append("--run-baseline")
    # scaffold's CLI uses dest='code_checker' with store_const:
    # None (no flag) -> omit, let scaffold resolve from config;
    # False (--no-code-checker) -> forward the flag.
    if args.code_checker is False:
        parts.append("--no-code-checker")
    if getattr(args, "worker_url", ""):
        parts += ["--worker-url", shlex.quote(args.worker_url)]
    return " ".join(parts)


def main():
    tokens = sys.argv[1:]

    # --- empty: ask mode ---
    if not tokens:
        _emit({
            "mode": "ask",
            "command": None,
            "values": {},
            "missing": [
                "--ref <file>",
                "--kernel <file|catlass_op_dir>",
                "--op-name <name>",
                "--devices <N>",
                f"--max-rounds (optional, default {default_max_rounds()})",
            ],
            "note": ("no arguments — ask the user for the missing fields, "
                     "then re-invoke /autoresearch with the full flag set."),
        })

    # --- resume forms ---
    if tokens[0] == "--resume":
        task_dir = tokens[1] if len(tokens) > 1 else ""
        cmd_parts = ["python", "scripts/resume.py"]
        if task_dir:
            cmd_parts.append(shlex.quote(task_dir))
        _emit({
            "mode": "resume",
            "command": " ".join(cmd_parts),
            "values": {"task_dir": task_dir or None},
            "missing": [],
        })

    # bare path → resume that task
    if not tokens[0].startswith("--"):
        path = tokens[0]
        if not os.path.isdir(path):
            _emit({
                "mode": "ask",
                "command": None,
                "values": {"first_token": path},
                "missing": [f"first token {path!r} is neither a flag nor an "
                            f"existing directory — clarify with the user "
                            f"before re-invoking /autoresearch."],
            })
        _emit({
            "mode": "resume",
            "command": f"python scripts/resume.py {shlex.quote(path)}",
            "values": {"task_dir": path},
            "missing": [],
        })

    # --- scaffold form ---
    # Reuse scaffold's parser so flag spec stays in lockstep.
    from scaffold import _make_arg_parser
    parser = _make_arg_parser()

    # Capture argparse's own error path (it normally prints to stderr and
    # sys.exit(2)) and convert into a structured ask-mode payload so the LLM
    # gets a JSON to react to instead of a stderr message.
    class _CapturedExit(Exception):
        def __init__(self, msg):
            self.msg = msg

    def _err(msg):
        raise _CapturedExit(msg)

    parser.error = _err  # type: ignore[assignment]

    try:
        args = parser.parse_args(tokens)
    except _CapturedExit as e:
        _emit({
            "mode": "ask",
            "command": None,
            "values": {"raw_tokens": tokens},
            "missing": [f"argparse rejected the args: {e.msg}"],
        })

    # Workflow-level required fields. argparse already enforces --ref and
    # --kernel as required positionals; the rest we check here so the LLM
    # gets a single error list.
    missing = []
    if not args.op_name:
        missing.append("--op-name <name>")
    # scaffold accepts --worker-url as the remote-only escape from local
    # --devices. Keep parse_args in lockstep so a dev box without an NPU
    # can fire `/autoresearch --ref ... --kernel ... --op-name ...
    # --worker-url ...` without being kicked into ask mode for a missing
    # --devices it shouldn't need.
    if not args.devices and not getattr(args, "worker_url", ""):
        missing.append("--devices <N> (or --worker-url <host:port>)")

    values = {
        "ref": args.ref,
        "kernel": args.kernel,
        "op_name": args.op_name,
        "devices": args.devices,
        "max_rounds": args.max_rounds,
        "eval_timeout": args.eval_timeout,
        "output_dir": args.output_dir or "ar_tasks",
        "run_baseline": True,
        "code_checker": getattr(args, "code_checker", None),
        "worker_url": getattr(args, "worker_url", ""),
    }

    if missing:
        _emit({
            "mode": "ask",
            "command": None,
            "values": values,
            "missing": missing,
        })

    _emit({
        "mode": "scaffold",
        "command": _build_scaffold_command(args),
        "values": values,
        "missing": [],
    })


if __name__ == "__main__":
    main()
