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

"""Print a human-readable report of <batch_dir>/batch_progress.json.

Designed for the "after the batch is done, what happened?" view — distinct
from monitor.py which reads ar_tasks/ live state. Static, fast, copy-pasteable.

Usage:
    python .autoresearch/scripts/batch/summarize.py <batch_dir>
"""

# pylint: disable=missing-function-docstring,wrong-import-position
from __future__ import annotations

import argparse
import statistics
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import manifest as mf


def _bucket_by_status(cases: dict) -> dict:
    by_status: dict[str, list[tuple[str, dict]]] = {
        "done": [], "error": [], "skip": [], "pending": [], "running": [],
    }
    for k, v in cases.items():
        by_status.setdefault(v.get("status", "pending"), []).append((k, v))
    return by_status


def _print_header(batch_dir: Path, progress: dict, total: int,
                  by_status: dict) -> None:
    print(f"batch summary  ({datetime.now().isoformat(timespec='seconds')})")
    print(f"batch_dir  {batch_dir}")
    print(f"mode={progress.get('mode', '?')}  "
          f"dsl={progress.get('dsl', '?')}")
    print("─" * 60)
    print(f"  total:    {total}")
    for s in ("done", "error", "skip", "pending", "running"):
        bucket = by_status.get(s, [])
        if bucket or s in ("done", "error"):
            print(f"  {s:8s}: {len(bucket)}")
    print()


def _compute_speedups(done_bucket: list) -> tuple:
    """Return (speedups, no_metric_keys). Each speedup tuple is
    (op_name, ratio, baseline, best)."""
    speedups: list[tuple[str, float, float, float]] = []
    no_metric: list[str] = []
    for k, v in done_bucket:
        r = v.get("result") or {}
        bm, best = r.get("baseline_metric"), r.get("best_metric")
        if (isinstance(bm, (int, float))
                and isinstance(best, (int, float)) and best > 0):
            speedups.append((k, bm / best, bm, best))
        else:
            no_metric.append(k)
    return speedups, no_metric


def _print_speedup_section(speedups: list) -> None:
    if not speedups:
        return
    vals = [s for _, s, _, _ in speedups]
    improved = sum(1 for v in vals if v > 1.05)
    onpar = sum(1 for v in vals if 0.95 <= v <= 1.05)
    regr = sum(1 for v in vals if v < 0.95)
    print("speedup (baseline / best, higher better):")
    print(f"  ops with metric: {len(speedups)}")
    print(f"  median:          {statistics.median(vals):.2f}x")
    print(f"  best:            {max(vals):.2f}x")
    print(f"  worst:           {min(vals):.2f}x")
    print(f"  improved:        {improved}  (>1.05x)")
    print(f"  on-par:          {onpar}    (0.95-1.05x)")
    print(f"  regress:         {regr}     (<0.95x)")
    print()


def _print_regressions(speedups: list) -> None:
    regressions = [(k, sp, base, bm) for k, sp, base, bm in speedups
                   if sp < 0.95]
    if not regressions:
        return
    print(f"regressions ({len(regressions)} ops slower than baseline):")
    for k, sp, base, bm in sorted(regressions, key=lambda r: r[1]):
        print(f"  - {k}: baseline {base:.3f} -> best {bm:.3f}  ({sp:.2f}x)")
    print()


def _print_no_metric(no_metric: list) -> None:
    if not no_metric:
        return
    print(f"done but no metric extracted ({len(no_metric)}):")
    for k in no_metric[:8]:
        print(f"  - {k}")
    if len(no_metric) > 8:
        print(f"  ... and {len(no_metric) - 8} more")
    print()


def _print_status_buckets(by_status: dict) -> None:
    for bucket_name, label in (("error", "errored ops"),
                               ("skip", "skipped ops")):
        bucket = by_status.get(bucket_name)
        if not bucket:
            continue
        print(f"{label} ({len(bucket)}):")
        for k, v in bucket:
            note = (v.get("note") or "(no note)")[:80]
            if bucket_name == "error":
                phase = v.get("final_phase", "?")
                print(f"  - {k}: phase={phase}  {note}")
            else:
                print(f"  - {k}: {note}")
        print()

    running = by_status.get("running")
    if running:
        print(f"running (likely stale; batch died mid-op): {len(running)}")
        for k, v in running:
            print(f"  - {k}: started_at={v.get('started_at', '?')}")
        print()

    pending = by_status.get("pending", [])
    if pending:
        print(f"still pending: {len(pending)}")
        for k, _ in pending[:10]:
            print(f"  - {k}")
        if len(pending) > 10:
            print(f"  ... and {len(pending) - 10} more")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("batch_dir")
    args = ap.parse_args()

    batch_dir = Path(args.batch_dir).resolve()
    if not batch_dir.is_dir():
        sys.exit(f"batch dir not found: {batch_dir}")

    progress = mf.load_progress(batch_dir)
    cases = progress.get("cases", {})
    if not cases:
        print(f"no cases recorded in {batch_dir / mf.PROGRESS_FILENAME}")
        return 1

    by_status = _bucket_by_status(cases)
    _print_header(batch_dir, progress, len(cases), by_status)

    speedups, no_metric = _compute_speedups(by_status.get("done", []))
    _print_speedup_section(speedups)
    _print_regressions(speedups)
    _print_no_metric(no_metric)
    _print_status_buckets(by_status)
    return 0


if __name__ == "__main__":
    sys.exit(main())
