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
  scaffold   — init flags (--ref + --kernel + --op-name + --dsl + devices/worker)
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

# pylint: disable=import-outside-toplevel,missing-function-docstring,wrong-import-position
import json
import os
import shlex
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.hw_detect import list_supported_dsls

_SUPPORTED_DSLS_DOC = "|".join(list_supported_dsls())


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
    parts = ["python", ".autoresearch/scripts/scaffold.py"]
    parts += ["--ref", shlex.quote(args.ref)]
    parts += ["--kernel", shlex.quote(args.kernel)]
    if args.op_name:
        parts += ["--op-name", shlex.quote(args.op_name)]
    if args.dsl:
        parts += ["--dsl", args.dsl]
    if args.framework and args.framework != "torch":
        parts += ["--framework", args.framework]
    if args.devices:
        parts += ["--devices", str(args.devices)]
    if args.worker_url:
        parts += ["--worker-url", args.worker_url]
    parts += ["--max-rounds", str(args.max_rounds)]
    parts += ["--eval-timeout", str(args.eval_timeout)]
    parts += ["--output-dir", args.output_dir or "ar_tasks"]
    parts.append("--run-baseline")
    if args.no_code_checker:
        parts.append("--no-code-checker")
    return " ".join(parts)


def _emit_empty_ask() -> None:
    _emit({
        "mode": "ask",
        "command": None,
        "values": {},
        "missing": [
            "--ref <file>",
            "--kernel <file>",
            "--op-name <name>",
            f"--dsl <{_SUPPORTED_DSLS_DOC}>",
            "--devices <N> or --worker-url <host:port>",
            "--max-rounds (optional, default 20)",
        ],
        "note": ("no arguments — ask the user for the missing fields, "
                 "then re-invoke /autoresearch with the full flag set."),
    })


def _handle_resume_forms(tokens: list) -> None:
    """Detect `--resume [path]` or bare `<path>` and emit. Returns
    without emitting if the tokens aren't a resume form."""
    if tokens[0] == "--resume":
        task_dir = tokens[1] if len(tokens) > 1 else ""
        cmd_parts = ["python", ".autoresearch/scripts/resume.py"]
        if task_dir:
            cmd_parts.append(shlex.quote(task_dir))
        _emit({
            "mode": "resume",
            "command": " ".join(cmd_parts),
            "values": {"task_dir": task_dir or None},
            "missing": [],
        })
    if tokens[0].startswith("--"):
        return
    path = tokens[0]
    if not os.path.isdir(path):
        _emit({
            "mode": "ask",
            "command": None,
            "values": {"first_token": path},
            "missing": [f"first token {path!r} is neither a flag nor an "
                        "existing directory — clarify with the user "
                        "before re-invoking /autoresearch."],
        })
    _emit({
        "mode": "resume",
        "command": "python .autoresearch/scripts/resume.py "
                   f"{shlex.quote(path)}",
        "values": {"task_dir": path},
        "missing": [],
    })


def _parse_scaffold_args(tokens: list):
    """Run scaffold's argparse, converting its sys.exit-on-error into an
    ask-mode emission so the LLM sees JSON instead of stderr noise."""
    from scaffold import _make_arg_parser
    parser = _make_arg_parser()

    class _CapturedExit(Exception):
        def __init__(self, msg):
            self.msg = msg

    def _err(msg):
        raise _CapturedExit(msg)

    parser.error = _err  # type: ignore[assignment]
    try:
        return parser.parse_args(tokens)
    except _CapturedExit as e:
        _emit({
            "mode": "ask",
            "command": None,
            "values": {"raw_tokens": tokens},
            "missing": [f"argparse rejected the args: {e.msg}"],
        })


def _collect_missing(args) -> list:
    """Workflow-level required fields. argparse enforces --ref / --kernel
    as required positionals; the rest is checked here so the LLM gets a
    single error list."""
    missing = []
    if not args.op_name:
        missing.append("--op-name <name>")
    if not args.dsl:
        # scaffold has default_dsl fallback from config.yaml, but at the
        # slash-command level we want explicit DSL so the LLM never
        # silently picks one.
        missing.append(f"--dsl <{_SUPPORTED_DSLS_DOC}>")
    if not args.devices and not args.worker_url:
        missing.append("--devices <N> or --worker-url <host:port>")
    if args.devices and args.worker_url:
        missing.append("--devices and --worker-url are mutually exclusive — "
                       "pass exactly one")
    return missing


def _args_to_values(args) -> dict:
    return {
        "ref": args.ref,
        "kernel": args.kernel,
        "op_name": args.op_name,
        "dsl": args.dsl,
        "framework": args.framework,
        "devices": args.devices,
        "worker_url": args.worker_url,
        "max_rounds": args.max_rounds,
        "eval_timeout": args.eval_timeout,
        "output_dir": args.output_dir or "ar_tasks",
        "run_baseline": True,
        "no_code_checker": args.no_code_checker,
    }


def main():
    tokens = sys.argv[1:]
    if not tokens:
        _emit_empty_ask()
    _handle_resume_forms(tokens)

    args = _parse_scaffold_args(tokens)
    missing = _collect_missing(args)
    values = _args_to_values(args)
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
