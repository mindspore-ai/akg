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

"""One-shot preparation step: discover ops + static verify.

Combines two mechanical pre-flight steps that always run together when
seeding a batch dir:

  1. Scan refs/ + kernels/ for the <op>_ref.py / <op>_kernel.py naming
     convention; write/update manifest.yaml's ops list.
  2. For every discovered op, compile the file, import the module, and
     check the required exports (Model / get_inputs / get_init_inputs in
     ref; ModelNew in kernel) are present. Per-op subprocess isolation —
     a missing dependency in one op doesn't poison the others.

This is the only step where merging makes sense. Everything else (worker
start, run, monitor, summarize) involves user decisions and stays as
separate commands.

Usage:
    python .autoresearch/scripts/batch/prepare.py <batch_dir> --dsl triton_ascend
    python .autoresearch/scripts/batch/prepare.py <batch_dir>
        # re-run after adding/removing files; inherits dsl from manifest

Flags mirror discover.py (filter / exclude / dirs) and verify.py (only).
Exits 0 only if both steps pass; on discover failure verify is skipped.
"""

# pylint: disable=broad-exception-caught,import-outside-toplevel,missing-function-docstring,wrong-import-position
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import discover
import manifest as mf
import verify


def _preflight_check_hook_paths() -> int:
    """Verify every `.autoresearch/scripts/...py` referenced by
    `.claude/settings.json` hook commands actually exists. Returns 0 on
    pass, prints a re-sync hint and returns 1 on stale paths.

    A common breakage is settings.json was activated before a refactor
    that renamed hook scripts — the activated settings.json then
    references stale paths, all hooks silently fail, and Claude has no
    AR guidance when the batch later runs.
    """
    import json
    import re
    repo_root = Path(__file__).resolve().parents[3]
    settings_path = repo_root / ".claude" / "settings.json"
    if not settings_path.is_file():
        return 0
    try:
        settings = json.loads(settings_path.read_text())
    except Exception as e:
        print(f"[preflight] cannot parse {settings_path}: {e}",
              file=sys.stderr)
        return 1

    pattern = re.compile(r"\.autoresearch/scripts/[\w/]+\.py")
    missing: list[tuple[str, str]] = []
    for phase, entries in (settings.get("hooks") or {}).items():
        for entry in entries:
            for hook in entry.get("hooks") or []:
                for rel in pattern.findall(hook.get("command", "")):
                    if not (repo_root / rel).is_file():
                        missing.append((phase, rel))

    if missing:
        print(f"[preflight] {settings_path} references missing hook scripts:",
              file=sys.stderr)
        for phase, rel in missing:
            print(f"    {phase}: {rel}", file=sys.stderr)
        print("\nLikely cause: settings.json activated before a refactor "
              "renamed hook scripts. Re-sync from the canonical source.",
              file=sys.stderr)
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Prepare a batch dir: discover ops + verify static check.",
    )
    ap.add_argument("batch_dir")
    ap.add_argument("--dsl", default="",
                    help="DSL written into manifest, e.g. triton_ascend "
                         "(inherits from existing manifest if present)")
    ap.add_argument("--ref-dir", default="",
                    help="ref subdirectory (default: from manifest, else 'refs')")
    ap.add_argument("--kernel-dir", default="",
                    help="kernel subdirectory (default: from manifest, else 'kernels')")
    ap.add_argument("--filter", default="",
                    help="glob to KEEP only matching op names (e.g. '*norm')")
    ap.add_argument("--exclude", action="append", default=[],
                    help="glob(s) to drop matching op names; repeatable")
    ap.add_argument("--only", default="",
                    help="restrict the verify step to comma-separated op names "
                         "(does not affect what gets written to the manifest)")
    ap.add_argument("--skip-verify", action="store_true",
                    help="run discover only; don't invoke static verify")
    args = ap.parse_args()

    batch_dir = Path(args.batch_dir).resolve()
    if not batch_dir.is_dir():
        sys.exit(f"batch dir not found: {batch_dir}")

    # ---- Step 0: preflight (hook paths) ----------------------------------
    if _preflight_check_hook_paths() != 0:
        return 1

    # ---- Step 1: discover -------------------------------------------------
    print(f"[prepare 1/2] discover  batch_dir={batch_dir}")

    existing: dict = {}
    try:
        manifest_path = mf.find_manifest(batch_dir)
        existing = mf.load_manifest(manifest_path)
    except mf.ManifestError:
        pass

    dsl = args.dsl or existing.get("dsl") or ""
    if not dsl:
        sys.exit("--dsl required (no existing manifest to inherit from)")

    ref_dir = args.ref_dir or existing.get("ref_dir") or "refs"
    kernel_dir = args.kernel_dir or existing.get("kernel_dir") or "kernels"

    ops = discover.discover(
        batch_dir, ref_dir, kernel_dir,
        include_glob=args.filter or None,
        exclude_globs=list(args.exclude),
    )
    if not ops:
        sys.exit("no ops discovered. Expected files matching "
                 "<op_name>_ref.py / <op_name>_kernel.py in the configured "
                 "ref_dir / kernel_dir.")

    target = discover.write_manifest(batch_dir, dsl, ref_dir, kernel_dir, ops)
    print(f"  wrote {len(ops)} ops to {target.name}")
    for op in ops:
        print(f"  - {op}")

    if args.skip_verify:
        print("\n[prepare 2/2] verify static check: skipped (--skip-verify)")
        return 0

    # ---- Step 2: verify static check -------------------------------------------
    print("\n[prepare 2/2] verify static check")
    rc = verify.run_verification(batch_dir, only=args.only)
    if rc == 0:
        print("\n[prepare] all checks passed; batch dir is ready to run.")
    else:
        print("\n[prepare] verify static check reported failures; "
              "fix the offending files and re-run prepare.py before "
              "starting the batch.")
    return rc


if __name__ == "__main__":
    sys.exit(main())
