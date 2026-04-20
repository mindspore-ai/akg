# Copyright 2025-2026 Huawei Technologies Co., Ltd
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

"""Replay Guard: LangGraph node-level state snapshot recording & verification.

In ``record`` mode every node's output is captured as a snapshot keyed by
``(node_name, step_count)``.  In ``replay`` mode the same snapshot is loaded
and compared against the actual node output after execution.  A mismatch
raises ``ReplayGuardError`` so that framework regressions are caught early
instead of silently diverging into un-cached workflow branches.

The snapshot store is persisted inside the **same** LLM-cache JSON file
under the reserved top-level key ``_node_snapshots``.  This keeps the
recording / replay artefact self-contained – one file per scenario.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from akg_agents.core_v2.llm.cache.cache_utils import read_cache_file, write_cache_file

logger = logging.getLogger(__name__)

_NODE_SNAPSHOTS_KEY = "_node_snapshots"

GUARD_FIELDS: Dict[str, List[str]] = {
    "coder": ["coder_code", "codegen_invalid", "codegen_invalid_reason"],
    "kernel_gen": ["coder_code", "codegen_invalid", "codegen_invalid_reason"],
    "verifier": ["verifier_result", "verifier_error"],
    "conductor": ["conductor_decision", "conductor_suggestion"],
    "kernel_conductor": ["conductor_decision", "conductor_suggestion"],
    "designer": ["designer_code"],
    "kernel_designer": ["designer_code"],
    "code_checker": ["code_check_passed", "code_check_errors"],
    "fix_code_gen": ["coder_code", "fix_code_gen_success"],
}

HASH_FIELDS: Set[str] = {
    "coder_code", "designer_code", "verifier_error",
    "conductor_suggestion", "code_check_errors",
}


class ReplayGuardError(RuntimeError):
    """Raised when a node's actual output diverges from the recorded snapshot."""

    def __init__(self, node_name: str, step: int, field: str,
                 expected: Any, actual: Any):
        self.node_name = node_name
        self.step = step
        self.field = field
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"[ReplayGuard] {node_name}@step{step} field '{field}' mismatch: "
            f"expected={_preview(expected)}, actual={_preview(actual)}"
        )


def _preview(value: Any, limit: int = 120) -> str:
    s = str(value)
    if len(s) > limit:
        return s[:limit] + "..."
    return s


def _fingerprint(value: Any) -> str:
    """Stable hash for long string fields so snapshots stay small."""
    text = str(value) if value is not None else ""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _snap_value(field: str, value: Any) -> Any:
    if field in HASH_FIELDS and isinstance(value, str) and len(value) > 256:
        return {"__hash": _fingerprint(value), "__len": len(value)}
    if value is None:
        return None
    if isinstance(value, (bool, int, float, str)):
        return value
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return str(value)


def _is_empty(value: Any) -> bool:
    """Treat None, empty string, and empty list as semantically equivalent."""
    return value is None or value == "" or value == []


def _match(field: str, expected: Any, actual: Any) -> bool:
    """Check whether *actual* matches the recorded *expected* snapshot value."""
    if _is_empty(expected) and _is_empty(actual):
        return True
    if isinstance(expected, dict) and "__hash" in expected:
        return _fingerprint(actual) == expected["__hash"]
    if isinstance(expected, bool) or isinstance(actual, bool):
        return expected == actual
    return expected == actual


class ReplaySnapshotStore:
    """Record / replay node-level state snapshots.

    Parameters
    ----------
    cache_file_path : str
        Path to the LLM-cache JSON (the same file used by ``LLMCache``).
    mode : str
        ``"record"`` or ``"replay"``.
    guard_fields : dict, optional
        Override per-node field lists.  Falls back to module-level
        ``GUARD_FIELDS`` for nodes not listed.
    """

    def __init__(
        self,
        cache_file_path: str,
        mode: str,
        guard_fields: Optional[Dict[str, List[str]]] = None,
    ):
        if mode not in ("record", "replay"):
            raise ValueError(f"ReplaySnapshotStore mode must be record/replay, got {mode}")
        self.cache_file_path = cache_file_path
        self.mode = mode
        self._field_map = {**GUARD_FIELDS, **(guard_fields or {})}
        self._snapshots: Dict[str, Dict[str, Any]] = {}
        if mode == "replay":
            self._load()

    def _snap_key(self, node_name: str, step: int) -> str:
        return f"{node_name}@{step}"

    def _load(self) -> None:
        data = read_cache_file(self.cache_file_path)
        self._snapshots = data.get(_NODE_SNAPSHOTS_KEY, {})
        if not self._snapshots:
            logger.warning(
                f"[ReplayGuard] No node snapshots found in {self.cache_file_path}. "
                "Node-level replay guard is disabled for this run."
            )

    def _save(self) -> None:
        data = read_cache_file(self.cache_file_path)
        data[_NODE_SNAPSHOTS_KEY] = self._snapshots
        write_cache_file(self.cache_file_path, data)

    def record(self, node_name: str, step: int, result: Dict[str, Any]) -> None:
        """Capture a snapshot of *result* for the given node invocation."""
        fields = self._field_map.get(node_name)
        if not fields:
            return
        snap: Dict[str, Any] = {}
        for f in fields:
            if f in result:
                snap[f] = _snap_value(f, result[f])
        key = self._snap_key(node_name, step)
        self._snapshots[key] = snap
        self._save()
        logger.debug(f"[ReplayGuard] Recorded snapshot {key}: {list(snap.keys())}")

    def verify(self, node_name: str, step: int, result: Dict[str, Any]) -> None:
        """Verify *result* against the recorded snapshot.  Raises on mismatch."""
        fields = self._field_map.get(node_name)
        if not fields:
            return
        key = self._snap_key(node_name, step)
        expected = self._snapshots.get(key)
        if expected is None:
            logger.warning(
                f"[ReplayGuard] No snapshot for {key}, skipping verification "
                "(the recorded run may not have reached this node)"
            )
            return
        for f in fields:
            exp_val = expected.get(f)
            act_val = result.get(f)
            if not _match(f, exp_val, act_val):
                raise ReplayGuardError(node_name, step, f, exp_val, act_val)
        logger.info(f"[ReplayGuard] {key} passed ({len(fields)} fields checked)")

    @property
    def has_snapshots(self) -> bool:
        return bool(self._snapshots)
