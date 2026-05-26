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

"""Manifest loading + progress JSON I/O for the batch runner.

Workspace convention:
    <batch_dir>/
        manifest.yaml | manifest.json    # user-authored
        batch_progress.json              # written here
        batch.log                        # written here
        <ref_dir>/<op_name>_ref.py
        <kernel_dir>/<op_name>_kernel.py

YAML support is optional (requires pyyaml). JSON works with stdlib only.
"""

# pylint: disable=import-outside-toplevel,missing-class-docstring,missing-function-docstring
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

PROGRESS_FILENAME = "batch_progress.json"
LOG_FILENAME = "batch.log"
VALID_MODES = ("ref-kernel",)
VALID_STATUSES = ("pending", "running", "done", "error", "skip")


class ManifestError(Exception):
    pass


def _load_yaml(path: Path) -> Any:
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise ManifestError(
            f"{path.name} is YAML but pyyaml is not installed. "
            "Either `pip install pyyaml` or rename to manifest.json "
            "(JSON format)."
        ) from exc
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def find_manifest(batch_dir: Path) -> Path:
    """Return path to manifest.yaml or manifest.json, preferring YAML."""
    yaml_path = batch_dir / "manifest.yaml"
    if yaml_path.exists():
        return yaml_path
    json_path = batch_dir / "manifest.json"
    if json_path.exists():
        return json_path
    raise ManifestError(
        f"no manifest.yaml or manifest.json in {batch_dir}"
    )


def load_manifest(manifest_path: Path) -> dict:
    if manifest_path.suffix in (".yaml", ".yml"):
        data = _load_yaml(manifest_path)
    elif manifest_path.suffix == ".json":
        data = _load_json(manifest_path)
    else:
        raise ManifestError(f"unknown manifest extension: {manifest_path}")
    if not isinstance(data, dict):
        raise ManifestError(f"manifest root must be a mapping, got {type(data).__name__}")
    return data


def resolve_cases(batch_dir: Path, manifest: dict, mode: str) -> list[dict]:
    """Apply the <op_name>_{ref,kernel}.py naming convention and return resolved
    case dicts. Pre-flight check that every referenced file exists.

    Returns a list of dicts with keys: op_name, ref (abs path), kernel
    (abs path).
    """
    if mode not in VALID_MODES:
        raise ManifestError(f"mode must be one of {VALID_MODES}, got {mode!r}")

    ops = manifest.get("ops")
    if not ops or not isinstance(ops, list):
        raise ManifestError("manifest.ops must be a non-empty list")

    ref_dir_raw = manifest.get("ref_dir")
    if not ref_dir_raw:
        raise ManifestError("manifest.ref_dir is required")
    ref_dir = (batch_dir / ref_dir_raw).resolve()
    if not ref_dir.is_dir():
        raise ManifestError(f"ref_dir not found: {ref_dir}")

    kernel_dir_raw = manifest.get("kernel_dir")
    if not kernel_dir_raw:
        raise ManifestError("kernel_dir is required")
    kernel_dir = (batch_dir / kernel_dir_raw).resolve()
    if not kernel_dir.is_dir():
        raise ManifestError(f"kernel_dir not found: {kernel_dir}")

    cases: list[dict] = []
    seen: set[str] = set()
    for entry in ops:
        if not isinstance(entry, str):
            raise ManifestError(
                f"manifest.ops entries must be strings (op names); got {entry!r}"
            )
        op_name = entry.strip()
        if not op_name:
            raise ManifestError("empty op_name in manifest.ops")
        if op_name in seen:
            raise ManifestError(f"duplicate op_name: {op_name}")
        seen.add(op_name)

        ref_path = ref_dir / f"{op_name}_ref.py"
        if not ref_path.is_file():
            raise ManifestError(f"{ref_path.relative_to(batch_dir)} not found")

        kernel_path = kernel_dir / f"{op_name}_kernel.py"
        if not kernel_path.is_file():
            raise ManifestError(
                f"{kernel_path.relative_to(batch_dir)} not found"
            )

        cases.append({
            "op_name": op_name,
            "ref": str(ref_path),
            "kernel": str(kernel_path),
        })

    return cases


def load_progress(batch_dir: Path) -> dict:
    path = batch_dir / PROGRESS_FILENAME
    if not path.exists():
        return {"batch_dir": str(batch_dir.resolve()), "cases": {}}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ManifestError(
            f"corrupt progress file at {path}: {e}"
        ) from e


def save_progress(batch_dir: Path, progress: dict) -> None:
    path = batch_dir / PROGRESS_FILENAME
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(progress, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def merge_cases(progress: dict, resolved_cases: list[dict],
                mode: str, dsl: str) -> tuple[dict, list[str]]:
    """Merge freshly-resolved cases into the progress dict.

    The manifest is the source of truth: ops that exist in the old progress
    file but no longer in `resolved_cases` are dropped (so a user filtering
    or deleting ops via discover.py / manual manifest edits actually shrinks
    the queue, matching the docs' "ops list fully replaced" promise).

    New cases are inserted as pending; surviving cases keep their status but
    their ref/kernel paths refresh.

    Returns (progress, dropped_op_names).
    """
    progress["mode"] = mode
    progress["dsl"] = dsl
    old_cases = progress.get("cases", {})
    resolved_ops = {c["op_name"] for c in resolved_cases}
    dropped = sorted(op for op in old_cases if op not in resolved_ops)

    new_cases: dict = {}
    for c in resolved_cases:
        op = c["op_name"]
        existing = old_cases.get(op)
        if existing is None:
            new_cases[op] = {
                "op_name": op,
                "ref": c["ref"],
                "kernel": c["kernel"],
                "status": "pending",
                "task_dir": None,
                "started_at": None,
                "finished_at": None,
                "final_phase": None,
                "rc": None,
                "result": {
                    "baseline_metric": None,
                    "best_metric": None,
                    "rounds": None,
                    "consecutive_failures": None,
                },
                "note": "",
            }
        else:
            existing["ref"] = c["ref"]
            existing["kernel"] = c["kernel"]
            new_cases[op] = existing
    progress["cases"] = new_cases
    return progress, dropped


def update_case(batch_dir: Path, op_name: str, **fields: Any) -> None:
    """Atomic update of one case's fields. Reloads progress file on each call
    so concurrent edits (e.g. by hand) aren't clobbered."""
    progress = load_progress(batch_dir)
    case = progress.get("cases", {}).get(op_name)
    if case is None:
        raise ManifestError(f"unknown op_name: {op_name}")
    if "status" in fields and fields["status"] not in VALID_STATUSES:
        raise ManifestError(f"invalid status: {fields['status']}")
    case.update(fields)
    save_progress(batch_dir, progress)


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def read_task_state(task_dir: Path) -> dict:
    """Pull the result block from <task_dir>/.ar_state/progress.json. Returns
    a dict with whichever fields could be read."""
    out: dict = {
        "baseline_metric": None,
        "best_metric": None,
        "rounds": None,
        "consecutive_failures": None,
    }
    for name in ("progress.json", ".progress.json"):
        pf = task_dir / ".ar_state" / name
        if not pf.exists():
            continue
        try:
            data = json.loads(pf.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        out["baseline_metric"] = data.get("baseline_metric")
        out["best_metric"] = data.get("best_metric")
        out["rounds"] = data.get("eval_rounds")
        out["consecutive_failures"] = data.get("consecutive_failures")
        break
    return out


def read_phase(task_dir: Path) -> str:
    pf = task_dir / ".ar_state" / ".phase"
    if pf.exists():
        try:
            return pf.read_text(encoding="utf-8").strip() or "UNKNOWN"
        except OSError:
            pass
    return "UNKNOWN"


def repo_root() -> Path:
    """The claude-autoresearch repo root, derived from this file's location.

    Layout: <repo>/.autoresearch/scripts/batch/manifest.py
    """
    return Path(__file__).resolve().parent.parent.parent.parent


_SCAFFOLD_RESULT_STATUSES = frozenset({"ok", "error"})
_SCAFFOLD_CREATED_MARKER = "[scaffold] Task directory created: "
_HEX6 = frozenset("0123456789abcdef")


def task_dir_belongs_to_op(name: str, op: str) -> bool:
    """Exact match for scaffold's `<op>_<int(time.time())>_<uuid.hex[:6]>`
    layout. Prefix-only matching would let op="avg" claim
    `avg_pool2d_*`; this splits off the last two `_` fields and
    compares the head verbatim."""
    parts = name.rsplit("_", 2)
    if len(parts) != 3:
        return False
    head, ts, rand = parts
    return (head == op
            and ts.isdigit() and ts
            and len(rand) == 6 and all(c in _HEX6 for c in rand))


def parse_scaffold_created_line(line: str) -> Path | None:
    """Early identity bind: scaffold prints
    `[scaffold] Task directory created: <abs>` on stderr right after
    mkdir, BEFORE baseline runs (which can stay silent >5s and would
    otherwise let the mid-run mtime fallback race a sibling batch)."""
    idx = line.find(_SCAFFOLD_CREATED_MARKER)
    if idx < 0:
        return None
    path = line[idx + len(_SCAFFOLD_CREATED_MARKER):].strip()
    if not path:
        return None
    p = Path(path)
    return p if p.is_dir() else None


def parse_scaffold_result_line(line: str) -> Path | None:
    """Identity-bound task_dir: a scaffold result JSON in THIS claude
    subprocess's stdout names the dir scaffold created for THIS run.
    Accepts both ok and error shapes — ok is OK / KERNEL_FAIL (both
    activatable, rc=0); error is INFRA_FAIL (rc=4) which still carries
    task_dir for inspection. Done/error verdict is decided downstream
    from .phase + rc."""
    s = line.strip()
    if not (s.startswith("{") and s.endswith("}")):
        return None
    try:
        d = json.loads(s)
    except json.JSONDecodeError:
        return None
    if (not isinstance(d, dict)
            or d.get("status") not in _SCAFFOLD_RESULT_STATUSES):
        return None
    td = d.get("task_dir")
    if not isinstance(td, str):
        return None
    p = Path(td)
    return p if p.is_dir() else None


def snapshot_task_dirs() -> set[Path]:
    """Current `ar_tasks/<dir>` set. Diff against a later snapshot to
    find dirs created since."""
    tasks_root = repo_root() / "ar_tasks"
    if not tasks_root.is_dir():
        return set()
    return {d for d in tasks_root.iterdir() if d.is_dir()}


def pick_new_task_dir(pre_snapshot: set[Path], op_name: str) -> Path | None:
    """Post-process fallback when no in-loop identity bind landed.
    Among `current - pre_snapshot`, pick the most-recently-mtime dir
    whose name passes `task_dir_belongs_to_op` (exact `<op>_<ts>_<hex6>`
    match, not a prefix). Races with sibling batches on the mtime
    tiebreak; the in-loop scaffold parsers are the primary path."""
    tasks_root = repo_root() / "ar_tasks"
    if not tasks_root.is_dir():
        return None
    try:
        current = {d for d in tasks_root.iterdir() if d.is_dir()}
    except OSError:
        return None
    matches = [d for d in (current - pre_snapshot)
               if task_dir_belongs_to_op(d.name, op_name)]
    if not matches:
        return None
    matches.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return matches[0]


def _parse_iso_ts(s: Any) -> float:
    if not isinstance(s, str):
        return 0.0
    try:
        return datetime.fromisoformat(s).timestamp()
    except ValueError:
        return 0.0


def _running_case(progress: dict) -> tuple:
    """Most-recently-started case with status=running, or (None, None)."""
    running = [(op, v) for op, v in (progress.get("cases") or {}).items()
               if isinstance(v, dict) and v.get("status") == "running"]
    if not running:
        return None, None
    running.sort(key=lambda kv: kv[1].get("started_at", ""), reverse=True)
    return running[0]


def _resolve_via_pointer(op: str) -> Path | None:
    pointer = repo_root() / ".autoresearch" / ".active_task"
    if not pointer.is_file():
        return None
    try:
        td = Path(pointer.read_text(encoding="utf-8").strip())
    except OSError:
        return None
    if td.is_dir() and task_dir_belongs_to_op(td.name, op):
        return td
    return None


def _resolve_via_recorded(case: dict, op: str) -> Path | None:
    recorded = case.get("task_dir")
    if not isinstance(recorded, str):
        return None
    p = Path(recorded)
    if p.is_dir() and task_dir_belongs_to_op(p.name, op):
        return p
    return None


def _resolve_via_mtime_scan(case: dict, op: str) -> Path | None:
    started_ts = _parse_iso_ts(case.get("started_at"))
    tasks_root = repo_root() / "ar_tasks"
    if not tasks_root.is_dir():
        return None
    try:
        cands = [d for d in tasks_root.iterdir()
                 if d.is_dir()
                 and task_dir_belongs_to_op(d.name, op)
                 and d.stat().st_mtime >= started_ts]
    except OSError:
        return None
    if not cands:
        return None
    return max(cands, key=lambda d: d.stat().st_mtime)


def find_running_case_task_dir(batch_dir: Path) -> Path | None:
    """task_dir of THIS batch's currently-running case, sourced from
    filesystem state — not from `batch_progress.task_dir`, which depends
    on `claude --print` flushing the scaffold line to stdout and can lag
    by tens of minutes.

    Primary source: `<repo>/.autoresearch/.active_task`, written by the
    post_bash hook the instant Claude runs `export AKG_AGENTS_AR_TASK_DIR=...`.
    Scoped to this batch by verifying the pointed dir's name matches
    the running case's op (so a sibling batch / manual session sharing
    `ar_tasks/` can't bleed into this view)."""
    progress = load_progress(batch_dir)
    if not progress:
        return None
    op, case = _running_case(progress)
    if op is None:
        return None
    return (_resolve_via_pointer(op)
            or _resolve_via_recorded(case, op)
            or _resolve_via_mtime_scan(case, op))
