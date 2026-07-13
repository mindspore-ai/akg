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

"""One-shot preparation step: discover ops + verify Tier 1.

Combines two mechanical pre-flight steps that always run together when
seeding a batch dir:

  1. Scan refs/ + kernels/ for the <op>_ref.py / <op>_kernel.py naming
     convention; write/update manifest.yaml's ops list.
  2. For every discovered op, compile the file, import the module, and
     check the required exports (Model / get_inputs / get_init_inputs in
     ref; ModelNew in kernel) are present. Per-op subprocess isolation —
     a missing dependency in one op doesn't poison the others.

This is the only step where merging makes sense. Everything else (run,
monitor, summarize) involves user decisions and stays as separate
commands.

Usage:
    python scripts/batch/prepare.py <batch_dir>

Flags mirror discover.py (filter / exclude / dirs) and verify.py (only).
Exits 0 only if both steps pass; on discover failure verify is skipped.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import discover
import verify


def _preflight_check_hook_paths() -> int:
    """Verify every `autoresearch/scripts/...py` referenced by
    `.claude/settings.json` hook commands actually exists. Returns 0 on
    pass, prints a re-sync hint and returns 1 on stale paths.

    A common breakage is `mv setups/autoresearch/* .claude/` was done
    before a refactor that renamed hook scripts — the activated
    settings.json then references stale paths, all hooks silently fail,
    and Claude has no AR guidance when the batch later runs.
    """
    import json
    import re
    repo_root = Path(__file__).resolve().parents[3]
    settings_path = repo_root / ".claude" / "settings.json"
    if not settings_path.is_file():
        return 0  # no activated settings → nothing to check
    try:
        settings = json.loads(settings_path.read_text())
    except Exception as e:
        print(f"[preflight] cannot parse {settings_path}: {e}",
              file=sys.stderr)
        return 1

    # `r"\autoresearch/..."` was treated as `\a` (bell) + `utoresearch/`
    # — the regex never matched, so missing hook scripts went undetected
    # until the batch hit them at runtime. Use a raw-bytes-safe form.
    pattern = re.compile(r"autoresearch/scripts/[\w/]+\.py")
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
        print("\nLikely cause: setups/autoresearch/ was activated (mv'd) "
              "before a refactor renamed hook scripts. Re-sync:",
              file=sys.stderr)
        print("  cp setups/autoresearch/settings.json .claude/settings.json",
              file=sys.stderr)
        return 1
    return 0


def main() -> int:
    ap = discover.make_parser(
        "Prepare a batch dir: discover ops + verify Tier 1.")
    ap.add_argument("--only", default="",
                    help="restrict the verify step to comma-separated op names "
                         "(does not affect what gets written to the manifest)")
    ap.add_argument("--skip-verify", action="store_true",
                    help="run discover only; don't invoke Tier 1 verify")
    args = ap.parse_args()

    # ---- Step 0: preflight (hook paths) ----------------------------------
    # Catch stale `.claude/settings.json` hook references before spending
    # minutes on discover + Tier 1 verify. Cheap (just file existence).
    if _preflight_check_hook_paths() != 0:
        return 1

    # ---- Step 1: discover -------------------------------------------------
    batch_dir, ref_dir, kernel_dir, ops = discover.resolve_request(args)
    print(f"[prepare 1/2] discover  batch_dir={batch_dir}")

    target = discover.write_manifest(batch_dir, ref_dir, kernel_dir, ops)
    print(f"  wrote {len(ops)} ops to {target.name}")
    for op in ops:
        print(f"  - {op}")

    if args.skip_verify:
        print("\n[prepare 2/2] verify Tier 1: skipped (--skip-verify)")
        return 0

    # ---- Step 2: verify Tier 1 -------------------------------------------
    print("\n[prepare 2/2] verify Tier 1")
    rc = verify.run_verification(
        batch_dir, full=False, only=args.only,
    )
    if rc == 0:
        print("\n[prepare] all checks passed; batch dir is ready to run.")
    else:
        print("\n[prepare] verify Tier 1 reported failures; "
              "fix the offending files and re-run prepare.py before "
              "starting the batch.")
    return rc


if __name__ == "__main__":
    sys.exit(main())
