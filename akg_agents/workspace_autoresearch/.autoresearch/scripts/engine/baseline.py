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

"""Run baseline eval (seed kernel + ref) and initialize .ar_state.

Usage:
    python .autoresearch/scripts/engine/baseline.py <task_dir> [--device-id N] [--worker-url URL]
"""

# pylint: disable=missing-function-docstring,wrong-import-position
import argparse
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPTS_ROOT)
from task_config import load_task_config
from utils.akg_eval import eval_kernel
from utils.failure_extractor import format_for_stdout
from workflow import run_baseline_init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task_dir")
    parser.add_argument("--device-id", type=int, default=None)
    parser.add_argument("--worker-url", default=None)
    args = parser.parse_args()

    task_dir = os.path.abspath(args.task_dir)
    os.makedirs(os.path.join(task_dir, ".ar_state"), exist_ok=True)

    config = load_task_config(task_dir)
    if config is None:
        print("[baseline] ERROR: task.yaml not found", file=sys.stderr)
        sys.exit(1)

    worker_url = args.worker_url or (config.worker_urls[0] if config.worker_urls else None)
    if worker_url:
        device_id = 0  # remote worker manages its own devices
    else:
        device_id = (args.device_id
                     if args.device_id is not None
                     else (config.devices[0] if config.devices else 0))

    print("[baseline] Running baseline eval...", flush=True)
    eval_data = eval_kernel(task_dir, config, device_id=device_id,
                            worker_url=worker_url, current_step=0)

    if not eval_data.get("correctness", False) or eval_data.get("error"):
        if eval_data.get("error"):
            print(f"[baseline] Error: {eval_data['error']}", flush=True)
        pretty = format_for_stdout(eval_data.get("failure_signals") or {})
        if pretty:
            print(pretty, flush=True)
        elif eval_data.get("raw_output_tail"):
            print("[baseline] Worker log tail:", flush=True)
            print(eval_data["raw_output_tail"], flush=True)

    sys.exit(run_baseline_init(task_dir, json.dumps(eval_data)))


if __name__ == "__main__":
    main()
