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

"""Auto-discover op names in a batch dir by the per-DSL kernel layout
(see ``manifest.resolve_kernel_paths_for_op``) paired with the universal
``<op>_ref.py`` naming, and optionally write/update the manifest's ops list.

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

Pairing rule: an op is included only if BOTH ``<ref_dir>/<op>_ref.py``
exists AND ``manifest.resolve_kernel_paths_for_op(<kernel_dir>, <op>)``
accepts the kernel-side layout for the configured DSL (flat
``<op>_kernel.py`` or multi-file ``<op>/{kernel.py,catlass_op/}``).
Unpaired sides are printed as warnings on stderr.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import manifest as mf


def _scan_ref_ops(ref_path: Path) -> set[str]:
    """Universal across DSLs: every op exposes ``<op>_ref.py``."""
    if not ref_path.is_dir():
        sys.exit(f"directory not found: {ref_path}")
    out: set[str] = set()
    for p in ref_path.glob("*_ref.py"):
        op = p.stem[: -len("_ref")]
        if op:
            out.add(op)
    return out


def _scan_kernel_ops(kern_path: Path) -> set[str]:
    """Enumerate kernel-side ops by delegating the per-DSL layout rule to
    ``manifest.resolve_kernel_paths_for_op``. discover only collects
    *candidate names* from the filesystem (flat-file stems + subdir
    names); the resolver decides which ones actually satisfy the DSL's
    convention. One owner for the rule — manifest — instead of forking
    flat-vs-multi-file logic here."""
    if not kern_path.is_dir():
        sys.exit(f"directory not found: {kern_path}")
    candidates: set[str] = set()
    for p in kern_path.glob("*_kernel.py"):
        op = p.stem[: -len("_kernel")]
        if op:
            candidates.add(op)
    for child in kern_path.iterdir():
        if child.is_dir():
            candidates.add(child.name)
    out: set[str] = set()
    for op in candidates:
        try:
            mf.resolve_kernel_paths_for_op(kern_path, op)
            out.add(op)
        except mf.ManifestError:
            continue
    return out


def discover(batch_dir: Path, ref_dir: str, kernel_dir: str,
             include_glob: str | None, exclude_globs: list[str]) -> list[str]:
    ref_path = (batch_dir / ref_dir).resolve()
    ref_ops = _scan_ref_ops(ref_path)

    kern_path = (batch_dir / kernel_dir).resolve()
    kern_ops = _scan_kernel_ops(kern_path)

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


def make_parser(description: str) -> argparse.ArgumentParser:
    """Shared discovery CLI surface used by discover and prepare."""
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("batch_dir")
    ap.add_argument("--ref-dir", default="",
                    help="ref subdirectory (default: manifest or 'refs')")
    ap.add_argument("--kernel-dir", default="",
                    help="kernel subdirectory (default: manifest or 'kernels')")
    ap.add_argument("--filter", default="", help="glob of op names to keep")
    ap.add_argument("--exclude", action="append", default=[],
                    help="glob(s) to drop; repeatable")
    return ap


def resolve_request(args) -> tuple[Path, str, str, list[str]]:
    """Resolve directories and discover ops once for both entry points."""
    batch_dir = Path(args.batch_dir).resolve()
    if not batch_dir.is_dir():
        sys.exit(f"batch dir not found: {batch_dir}")
    try:
        existing = mf.load_manifest(mf.find_manifest(batch_dir))
    except mf.ManifestError:
        existing = {}
    ref_dir = args.ref_dir or existing.get("ref_dir") or "refs"
    kernel_dir = args.kernel_dir or existing.get("kernel_dir") or "kernels"
    ops = discover(batch_dir, ref_dir, kernel_dir, args.filter or None,
                   list(args.exclude))
    if not ops:
        sys.exit("no ops discovered; expected paired <op>_ref.py and "
                 "<op>_kernel.py/DSL project in ref_dir and kernel_dir")
    return batch_dir, ref_dir, kernel_dir, ops


def main() -> int:
    ap = make_parser("Auto-discover ops by ref/kernel naming and DSL layout.")
    ap.add_argument("--write-manifest", action="store_true",
                    help="write/update the manifest's ops list (and other "
                         "fields if given via flags); without this, just "
                         "print the discovered ops")
    ap.add_argument("--json", action="store_true",
                    help="when not writing manifest: print as JSON array")
    args = ap.parse_args()
    batch_dir, ref_dir, kernel_dir, ops = resolve_request(args)

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
