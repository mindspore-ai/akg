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

"""
Mechanical plan.md settlement — no LLM needed.

After keep_or_discard.py runs, this script:
1. Reads the decision (KEEP/DISCARD/FAIL) from keep_or_discard output
2. Updates plan.md: mark active item [x] with result, advance (ACTIVE)

Usage:
    python settle.py <task_dir> <decision_json>

Output (stdout, last line):
    {"settled_item": "p1", "decision": "KEEP", "metric": 1294.8}

All plan.md mutation goes through workflow.PlanStore so the parse / render
formats can't drift across files.

Scope note: settle.py does NOT advance .ar_state/.phase. The phase
transition after a settled round is owned by pipeline.py's _post_settle
(via PhaseController.on_round_settled). When settle.py was its own
subprocess AND advanced phase, the parent orchestrator also re-ran the
transition — two owners writing the same state file. compute_next_phase
happened to be idempotent so it didn't surface as a bug, but the
ownership story stayed split. Keep the rule here: settle.py owns
plan.md only.
"""

# pylint: disable=missing-function-docstring,wrong-import-position
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from workflow import PlanStore


def main():
    if len(sys.argv) != 3:
        print(json.dumps({
            "error": "invalid arguments",
            "usage": "python settle.py <task_dir> <decision_json>",
            "received_args": sys.argv[1:],
        }))
        sys.exit(1)

    task_dir = sys.argv[1]
    decision_json = sys.argv[2]

    try:
        decision_data = json.loads(decision_json)
    except json.JSONDecodeError as exc:
        print(json.dumps({
            "error": "invalid decision_json",
            "details": str(exc),
        }))
        sys.exit(1)
    decision = decision_data.get("decision", "FAIL")
    best_metric = decision_data.get("best_metric")
    # For KEEP, best_metric is this round's value. For DISCARD we have no metric.
    metric_val = best_metric if decision == "KEEP" else None

    store = PlanStore(task_dir)
    if not store.exists():
        print(json.dumps({"error": "plan.md not found"}))
        sys.exit(1)

    try:
        settled_item_id, _ = store.settle_active(decision, metric_val)
    except RuntimeError as exc:
        print(json.dumps({"error": str(exc)}))
        sys.exit(1)

    print(json.dumps({
        "settled_item": settled_item_id,
        "decision": decision,
        "metric": metric_val,
    }))


if __name__ == "__main__":
    main()
