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
    python scripts/batch/summarize.py <batch_dir>
"""
from __future__ import annotations

import argparse
import statistics
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import manifest as mf
# Reach up one level (scripts/) for the shared settings accessors.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.settings import (  # noqa: E402
    classify_speedup, speedup_improved_above, speedup_regress_below,
)


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

    by_status: dict[str, list[tuple[str, dict]]] = {
        "done": [], "error": [], "skip": [], "pending": [], "running": [],
    }
    for k, v in cases.items():
        by_status.setdefault(v.get("status", "pending"), []).append((k, v))

    total = len(cases)
    print(f"batch summary  ({datetime.now().isoformat(timespec='seconds')})")
    print(f"batch_dir  {batch_dir}")
    print(f"mode={progress.get('mode', '?')}")
    print("─" * 60)
    print(f"  total:    {total}")
    for s in ("done", "error", "skip", "pending", "running"):
        bucket = by_status.get(s, [])
        if bucket or s in ("done", "error"):
            print(f"  {s:8s}: {len(bucket)}")
    print()

    # Speedup over done ops
    speedups: list[tuple[str, float, float, float]] = []
    no_metric: list[str] = []
    for k, v in by_status.get("done", []):
        r = v.get("result") or {}
        bm, best = r.get("baseline_metric"), r.get("best_metric")
        if (isinstance(bm, (int, float)) and isinstance(best, (int, float))
                and best > 0):
            speedups.append((k, bm / best, bm, best))
        else:
            no_metric.append(k)

    if speedups:
        vals = [s for _, s, _, _ in speedups]
        hi, lo = speedup_improved_above(), speedup_regress_below()
        labels = [classify_speedup(v) for v in vals]
        improved = labels.count("improved")
        onpar = labels.count("on-par")
        regr = labels.count("regress")
        print("speedup (baseline / best, higher better):")
        print(f"  ops with metric: {len(speedups)}")
        print(f"  median:          {statistics.median(vals):.2f}x")
        print(f"  best:            {max(vals):.2f}x")
        print(f"  worst:           {min(vals):.2f}x")
        print(f"  improved:        {improved}  (>{hi}x)")
        print(f"  on-par:          {onpar}    ({lo}-{hi}x)")
        print(f"  regress:         {regr}     (<{lo}x)")
        print()

    regressions = [(k, sp, base, bm) for k, sp, base, bm in speedups
                   if sp < speedup_regress_below()]
    if regressions:
        print(f"regressions ({len(regressions)} ops slower than baseline):")
        for k, sp, base, bm in sorted(regressions, key=lambda r: r[1]):
            print(f"  - {k}: baseline {base:.3f} -> best {bm:.3f}  ({sp:.2f}x)")
        print()

    if no_metric:
        print(f"done but no metric extracted ({len(no_metric)}):")
        for k in no_metric[:8]:
            print(f"  - {k}")
        if len(no_metric) > 8:
            print(f"  ... and {len(no_metric) - 8} more")
        print()

    if by_status.get("error"):
        print(f"errored ops ({len(by_status['error'])}):")
        for k, v in by_status["error"]:
            note = (v.get("note") or "(no note)")[:80]
            phase = v.get("final_phase", "?")
            print(f"  - {k}: phase={phase}  {note}")
        print()

    if by_status.get("skip"):
        print(f"skipped ops ({len(by_status['skip'])}):")
        for k, v in by_status["skip"]:
            note = (v.get("note") or "(no note)")[:80]
            print(f"  - {k}: {note}")
        print()

    if by_status.get("running"):
        print(f"running (likely stale; batch died mid-op): "
              f"{len(by_status['running'])}")
        for k, v in by_status["running"]:
            started = v.get("started_at", "?")
            print(f"  - {k}: started_at={started}")
        print()

    pending = by_status.get("pending", [])
    if pending:
        print(f"still pending: {len(pending)}")
        for k, _ in pending[:10]:
            print(f"  - {k}")
        if len(pending) > 10:
            print(f"  ... and {len(pending) - 10} more")

    return 0


if __name__ == "__main__":
    sys.exit(main())
