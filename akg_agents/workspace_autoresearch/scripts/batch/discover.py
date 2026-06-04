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

"""Auto-discover op names in a batch dir by the <op>_ref.py / <op>_kernel.py
naming convention, and optionally write/update the manifest's ops list.

Usage:
    # Just print the discovered ops (one per line, sorted):
    python scripts/batch/discover.py <batch_dir>

    # Bootstrap a fresh manifest from filesystem state (when no manifest exists):
    python scripts/batch/discover.py <batch_dir> --write-manifest

    # Refresh the ops list of an existing manifest after adding/removing files:
    python scripts/batch/discover.py <batch_dir> --write-manifest

    # Filter:
    python scripts/batch/discover.py <batch_dir> \\
        --filter "*norm" --exclude "groupnorm"

Pairing rule: an op is included only if BOTH <ref_dir>/<op>_ref.py and
<kernel_dir>/<op>_kernel.py exist; unpaired files are printed as
warnings on stderr.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import manifest as mf


def _scan_dir(dir_path: Path, suffix: str) -> set[str]:
    """Return op names whose files match <op_name><suffix>.py in dir_path."""
    if not dir_path.is_dir():
        sys.exit(f"directory not found: {dir_path}")
    out: set[str] = set()
    pattern = f"*{suffix}.py"
    for p in dir_path.glob(pattern):
        op = p.stem[: -len(suffix)]
        if op:
            out.add(op)
    return out


def discover(batch_dir: Path, ref_dir: str, kernel_dir: str,
             include_glob: str | None, exclude_globs: list[str]) -> list[str]:
    ref_path = (batch_dir / ref_dir).resolve()
    ref_ops = _scan_dir(ref_path, "_ref")

    kern_path = (batch_dir / kernel_dir).resolve()
    kern_ops = _scan_dir(kern_path, "_kernel")

    only_ref = ref_ops - kern_ops
    only_kern = kern_ops - ref_ops
    if only_ref:
        print(f"warning: {len(only_ref)} ops have ref but no kernel: "
              f"{', '.join(sorted(only_ref)[:5])}"
              f"{', ...' if len(only_ref) > 5 else ''}",
              file=sys.stderr)
    if only_kern:
        print(f"warning: {len(only_kern)} ops have kernel but no ref: "
              f"{', '.join(sorted(only_kern)[:5])}"
              f"{', ...' if len(only_kern) > 5 else ''}",
              file=sys.stderr)
    ops = ref_ops & kern_ops

    if include_glob:
        ops = {op for op in ops if fnmatch.fnmatch(op, include_glob)}
    for excl in exclude_globs:
        ops = {op for op in ops if not fnmatch.fnmatch(op, excl)}

    return sorted(ops)


def _has_pyyaml() -> bool:
    try:
        import yaml  # noqa: F401
        return True
    except ImportError:
        return False


def write_manifest(batch_dir: Path, ref_dir: str,
                   kernel_dir: str, ops: list[str]) -> Path:
    """Create or update <batch_dir>/manifest.{yaml,json}.

    Preserves any extra fields already present in the file. Only
    mode/ref_dir/kernel_dir/ops are written from this call.
    """
    yaml_path = batch_dir / "manifest.yaml"
    json_path = batch_dir / "manifest.json"

    if yaml_path.exists():
        target = yaml_path
    elif json_path.exists():
        target = json_path
    else:
        target = yaml_path if _has_pyyaml() else json_path

    existing: dict = {}
    if target.exists():
        try:
            existing = mf.load_manifest(target) or {}
        except mf.ManifestError as e:
            sys.exit(f"existing {target.name} unreadable: {e}")

    existing["mode"] = "ref-kernel"
    existing["ref_dir"] = ref_dir
    existing["kernel_dir"] = kernel_dir
    existing["ops"] = ops

    if target.suffix in (".yaml", ".yml"):
        if not _has_pyyaml():
            sys.exit("manifest.yaml requested but pyyaml not installed; "
                     "remove the .yaml file to fall back to JSON, or "
                     "`pip install pyyaml`")
        import yaml
        target.write_text(
            yaml.safe_dump(existing, sort_keys=False, default_flow_style=False,
                           allow_unicode=True),
            encoding="utf-8",
        )
    else:
        target.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    return target


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Auto-discover ops by <op>_ref.py / <op>_kernel.py convention."
    )
    ap.add_argument("batch_dir")
    ap.add_argument("--ref-dir", default="",
                    help="ref subdirectory (default: from manifest, else 'refs')")
    ap.add_argument("--kernel-dir", default="",
                    help="kernel subdirectory (default: from manifest, else 'kernels')")
    ap.add_argument("--filter", default="",
                    help="glob to KEEP only matching op names (e.g. '*norm')")
    ap.add_argument("--exclude", action="append", default=[],
                    help="glob(s) to drop matching op names; repeatable")
    ap.add_argument("--write-manifest", action="store_true",
                    help="write/update the manifest's ops list (and other "
                         "fields if given via flags); without this, just "
                         "print the discovered ops")
    ap.add_argument("--json", action="store_true",
                    help="when not writing manifest: print as JSON array")
    args = ap.parse_args()

    batch_dir = Path(args.batch_dir).resolve()
    if not batch_dir.is_dir():
        sys.exit(f"batch dir not found: {batch_dir}")

    existing: dict = {}
    try:
        manifest_path = mf.find_manifest(batch_dir)
        existing = mf.load_manifest(manifest_path)
    except mf.ManifestError:
        pass

    ref_dir = args.ref_dir or existing.get("ref_dir") or "refs"
    kernel_dir = args.kernel_dir or existing.get("kernel_dir") or "kernels"

    ops = discover(
        batch_dir, ref_dir, kernel_dir,
        include_glob=args.filter or None,
        exclude_globs=list(args.exclude),
    )
    if not ops:
        sys.exit("no ops discovered. expected files matching "
                 "<op_name>_ref.py / <op_name>_kernel.py in the configured "
                 "ref_dir / kernel_dir.")

    if args.write_manifest:
        target = write_manifest(batch_dir, ref_dir, kernel_dir, ops)
        print(f"wrote {len(ops)} ops to {target.name}")
        for op in ops:
            print(f"  - {op}")
        return 0

    if args.json:
        print(json.dumps(ops, indent=2))
    else:
        for op in ops:
            print(op)
    return 0


if __name__ == "__main__":
    sys.exit(main())
