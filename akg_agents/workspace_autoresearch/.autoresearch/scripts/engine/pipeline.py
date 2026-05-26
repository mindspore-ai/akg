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
Post-edit pipeline — runs ALL mechanical steps after Claude Code edits code.

Claude Code does the LLM work (plan, edit, diagnose). Then calls this:
    python .autoresearch/scripts/engine/pipeline.py <task_dir>

This script does:
    1. quick_check → fail? rollback, report
    2. eval → get metrics
    3. keep_or_discard → KEEP/DISCARD/FAIL
    4. settle → update plan.md, advance (ACTIVE)
    5. compute next phase → write .phase
    6. print status + next guidance

Output: human-readable status to stdout. Claude Code sees it and acts accordingly.
"""

# pylint: disable=broad-exception-caught,import-outside-toplevel,missing-function-docstring,wrong-import-position
import json
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPTS_ROOT)
sys.path.insert(0, SCRIPT_DIR)
from task_config import load_task_config
from phase_machine import (
    get_active_item,
    get_guidance, auto_rollback, load_progress, edit_marker_path,
    pending_settle_path, FINISH,
)
from utils.akg_eval import eval_kernel
from utils.failure_extractor import format_for_stdout
from utils.json_io import parse_last_json_line
from workflow import PhaseController, record_round


def _run_settle(task_dir: str, kd_json: dict) -> tuple:
    """Invoke settle.py with the given kd_json.

    Returns (rc, stdout_tail, stderr_tail, settle_json):
      - rc:           settle.py exit code
      - stdout_tail:  last 400 chars of stdout (for error reports)
      - stderr_tail:  last 400 chars of stderr
      - settle_json:  parsed last-JSON-line from stdout, or None on
                      parse failure / non-zero rc. Carries the
                      `settled_item` id, which the caller needs for the
                      status report — `get_active_item()` AFTER settle
                      points at the NEXT ACTIVE item, not the one we
                      just settled.
    """
    settle = subprocess.run(
        [sys.executable, os.path.join(SCRIPT_DIR, "settle.py"),
         task_dir, json.dumps(kd_json)],
        capture_output=True, text=True, timeout=10, check=False,
    )
    settle_json = parse_last_json_line(settle.stdout) if settle.returncode == 0 else None
    return (
        settle.returncode,
        (settle.stdout or "").strip()[-400:],
        (settle.stderr or "").strip()[-400:],
        settle_json,
    )


def _persist_pending_settle(task_dir: str, kd_json: dict) -> None:
    path = pending_settle_path(task_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(kd_json, f)


def _clear_pending_settle(task_dir: str) -> None:
    path = pending_settle_path(task_dir)
    if os.path.exists(path):
        os.remove(path)


def _emit_settle_failure(task_dir: str, rc: int, tail_out: str,
                         tail_err: str) -> None:
    print(f"[PIPELINE] SETTLE FAILED (rc={rc}). plan.md was NOT updated. "
          "progress.json + history.jsonl already moved during this round; "
          "re-running this script will RETRY SETTLE ONLY (kd_json was "
          "persisted to .ar_state/.pending_settle.json) — it will NOT "
          "re-run quick_check/eval/keep_or_discard.\n"
          "\n"
          "Recovery options (do NOT hand-edit plan.md):\n"
          "  1. Fix the underlying cause from the stderr tail below, "
          "then re-run pipeline.py — the replay-only path will retry "
          "settle on the same kd_json.\n"
          "  2. If the failure is structural (plan.md malformed, no "
          "(ACTIVE) item, etc.) and settle cannot recover, run "
          "create_plan.py to write a fresh plan.md. While "
          "pending_settle.json exists, hooks/guard_bash.py allows "
          "create_plan.py in EDIT phase as a recovery path; on "
          "successful create_plan validation hooks/post_bash.py clears "
          "pending_settle.json. The orphan history.jsonl row stays "
          "(audit trail) but no longer corresponds to any plan item.\n"
          "\n"
          f"stdout tail: {tail_out}\n"
          f"stderr tail: {tail_err}", file=sys.stderr)


def _post_settle(task_dir: str, decision: str, settled_id: str) -> None:
    """Common path after a successful settle: advance phase, clear edit
    marker, print status. Runs whether settle succeeded the first time or
    on the replay-only retry."""
    next_phase = PhaseController(task_dir).on_round_settled()
    marker = edit_marker_path(task_dir)
    if os.path.exists(marker):
        os.remove(marker)

    # FINISH is a one-way terminal transition — generate the deterministic
    # report.md (summary tables + inline SVG curve) here so it's on disk
    # before the FINISH guidance announces its path.
    if next_phase == FINISH:
        try:
            from report import write_report
            rp = write_report(task_dir)
            if rp:
                print("[PIPELINE] Report written: "
                      f"{os.path.relpath(rp, task_dir)}")
        except Exception as e:
            print(f"[PIPELINE] Report generation failed: {e}",
                  file=sys.stderr)

    progress = load_progress(task_dir) or {}
    rounds = progress.get("eval_rounds", 0)
    max_rounds = progress.get("max_rounds", "?")
    best = progress.get("best_metric")
    baseline = progress.get("baseline_metric")
    failures = progress.get("consecutive_failures", 0)

    improv = ""
    if (
        best is not None and baseline is not None
        and isinstance(best, (int, float))
        and isinstance(baseline, (int, float))
        and baseline != 0 and best != 0
    ):
        pct = (baseline - best) / abs(baseline) * 100
        speedup = baseline / best
        improv = f" ({speedup:.2f}x vs ref, {pct:+.1f}%)"

    print(f"\n{'=' * 50}")
    print(f"[{decision}] {settled_id} | Round {rounds}/{max_rounds} | "
          f"Best: {best}{improv} | Failures: {failures}")
    print(f"Phase -> {next_phase}")
    print(f"{'=' * 50}")
    print(get_guidance(task_dir))


def _handle_replay_settle(task_dir: str) -> None:
    """If .pending_settle.json exists, retry settle.py with the persisted
    kd_json and exit. Reaching here means a previous pipeline.py invocation
    got past keep_or_discard but settle.py failed; re-running from scratch
    would re-eval and double-write progress/history.

    Lives BEFORE task.yaml load so retry works even if task config has
    drifted (settle only touches .ar_state)."""
    pending_path = pending_settle_path(task_dir)
    if not os.path.exists(pending_path):
        return
    try:
        with open(pending_path, "r", encoding="utf-8") as f:
            kd_json = json.load(f)
    except Exception as e:
        print(f"[PIPELINE] pending settle file unreadable ({e}). "
              "Removing it and bailing — please re-run pipeline.py "
              "to start a fresh round.", file=sys.stderr)
        _clear_pending_settle(task_dir)
        sys.exit(1)
    print("[PIPELINE] Retrying settle from "
          f"{os.path.basename(pending_path)} "
          "(skipping quick_check/eval/keep_or_discard).", flush=True)
    rc, tail_out, tail_err, settle_json = _run_settle(task_dir, kd_json)
    if rc != 0:
        _emit_settle_failure(task_dir, rc, tail_out, tail_err)
        sys.exit(1)
    _clear_pending_settle(task_dir)
    # Use settle.py's reported settled_item — by this point ACTIVE has
    # already advanced to the NEXT pending item.
    settled_id = (settle_json or {}).get("settled_item") or "?"
    _post_settle(task_dir, kd_json.get("decision", "?"), settled_id)
    sys.exit(0)


def _run_quick_check(task_dir: str) -> None:
    """Run quick_check.py; on failure roll back, print guidance, exit 0
    (the agent re-edits)."""
    print("[PIPELINE] Running quick_check...", flush=True)
    qc = subprocess.run(
        [sys.executable, os.path.join(SCRIPT_DIR, "quick_check.py"),
         task_dir],
        capture_output=True, text=True, timeout=60, check=False,
    )
    if qc.returncode == 0 and "OK" in qc.stdout:
        print("[PIPELINE] Quick check PASS", flush=True)
        return
    auto_rollback(task_dir)
    marker = edit_marker_path(task_dir)
    if os.path.exists(marker):
        os.remove(marker)
    print(f"[PIPELINE] QUICK CHECK FAIL: {qc.stdout[:200]}")
    print("[PIPELINE] Auto-rolled back. Fix and re-edit.")
    print(get_guidance(task_dir))
    sys.exit(0)


def _do_eval(task_dir: str, config, device_id: int,
             worker_url) -> dict:
    print("[PIPELINE] Running eval...", flush=True)
    progress_for_count = load_progress(task_dir) or {}
    current_step = int(progress_for_count.get("eval_rounds", 0)) + 1
    try:
        return eval_kernel(task_dir, config, device_id=device_id,
                           worker_url=worker_url,
                           current_step=current_step)
    except Exception as exc:
        auto_rollback(task_dir)
        print(f"[PIPELINE] EVAL CRASH: {exc}", file=sys.stderr)
        sys.exit(1)


def _surface_eval_failure(eval_json: dict) -> None:
    """Print structured failure signals (UB overflow, aivec trap, OOM, ...)
    extracted from the worker's raw log. Falls back through increasingly
    coarse sources so something always reaches the user on failure."""
    if eval_json.get("error"):
        print(f"[PIPELINE] Error: {eval_json['error']}", flush=True)
    pretty = format_for_stdout(eval_json.get("failure_signals") or {})
    if pretty:
        print(pretty, flush=True)
    elif eval_json.get("raw_output_tail"):
        print("[PIPELINE] Worker log tail (no structured signals matched):",
              flush=True)
        print(eval_json["raw_output_tail"], flush=True)


def main():
    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <task_dir>")
        sys.exit(1)

    task_dir = os.path.abspath(sys.argv[1])
    _handle_replay_settle(task_dir)

    config = load_task_config(task_dir)
    if config is None:
        print("[PIPELINE] ERROR: task.yaml not found")
        sys.exit(1)

    active = get_active_item(task_dir)
    # Persist the full description — dashboards/logs do their own
    # display-time truncation.
    desc = active["description"] if active else "optimization round"
    plan_item = active["id"] if active else None

    worker_url = config.worker_urls[0] if config.worker_urls else None
    device_id = (0 if worker_url
                 else (config.devices[0] if config.devices else 0))

    _run_quick_check(task_dir)
    eval_json = _do_eval(task_dir, config, device_id, worker_url)

    correctness = eval_json.get("correctness", False)
    metrics = eval_json.get("metrics", {})
    print(f"[PIPELINE] Eval: correctness={correctness}, "
          f"metrics={metrics}", flush=True)

    # infra_fail: eval pipeline broke before kernel was meaningfully
    # exercised. Roll back and skip the round — recording a FAIL here
    # would mislead later DIAGNOSE / KEEP / DISCARD.
    if eval_json.get("outcome") == "infra_fail":
        auto_rollback(task_dir)
        print("[PIPELINE] INFRA_FAIL: "
              f"{eval_json.get('error', 'no data')}. "
              "Rolled back, not recording round.", flush=True)
        sys.exit(0)

    if not correctness or eval_json.get("error"):
        _surface_eval_failure(eval_json)

    # In-process keep_or_discard (no subprocess + stdout JSON round-trip).
    # Stray stdout from an imported module would otherwise corrupt the
    # decision protocol.
    try:
        kd_json = record_round(task_dir, eval_json,
                               description=desc, plan_item=plan_item)
    except Exception as exc:
        print(f"[PIPELINE] KEEP/DISCARD ERROR: {exc}")
        sys.exit(1)

    decision = kd_json.get("decision", "FAIL")

    # plan.md is the only state piece settle.py owns. If settle fails,
    # the kd_json is persisted to .pending_settle.json so the NEXT
    # invocation of pipeline.py retries settle alone (no second eval,
    # no duplicate history row).
    rc, tail_out, tail_err, _ = _run_settle(task_dir, kd_json)
    if rc != 0:
        _persist_pending_settle(task_dir, kd_json)
        _emit_settle_failure(task_dir, rc, tail_out, tail_err)
        sys.exit(1)
    _clear_pending_settle(task_dir)

    settled_id = active["id"] if active else "?"
    _post_settle(task_dir, decision, settled_id)


if __name__ == "__main__":
    main()
