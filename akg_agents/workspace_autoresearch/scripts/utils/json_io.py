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

"""JSON helpers shared by every disk / HTTP JSON boundary.

`sanitize_floats` strips `inf` / `-inf` / `nan` to `None`. Python's
default `json.dump` writes those as `Infinity` / `-Infinity` / `NaN`,
which is NOT valid strict JSON: FastAPI's encoder rejects them with
HTTP 500, and consumers that load with the default `parse_constant`
get back `float('inf')` / `float('nan')` (then break on arithmetic
or comparison). Run every metrics-bearing dict through this before
serialising — `eval_assemble` filters most non-finite values, but
per-shape arrays / pass-through scalars / artifact JSON blobs still
slip through, so the sanitiser is the canonical safety net.

Every JSON write boundary that can carry profiler output goes
through here:

  - phase_machine.state_store: state.json (single source of truth)
                                + history.jsonl (append-only round log)
  - engine.eval_kernel: .eval_result*.json sidecar
  - utils.eval_runner: profile-block artifact JSONs
  - worker.server: /api/v1/run response (FastAPI 500 path)
"""
from __future__ import annotations

import json
import math
import os
from typing import Any, List, Optional


def _read_whole_file(path: str) -> str:
    """Loop os.read until EOF — `open().read()` short-reads on large
    history.jsonl, silently dropping the tail."""
    fd = os.open(path, os.O_RDONLY)
    try:
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks).decode("utf-8", errors="replace")
    finally:
        os.close(fd)


def load_jsonl(path: str) -> List[dict]:
    """Every JSON object in a JSONL file. Missing file → []. Blank
    and malformed lines are skipped."""
    if not os.path.exists(path):
        return []
    out: list[dict] = []
    for line in _read_whole_file(path).split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def sanitize_floats(obj: Any) -> Any:
    """Recursively replace inf / -inf / nan with None.

    Returns a new object — does not mutate the input. Scalars, dicts,
    lists, and tuples are walked. Other types (str, int, bool, None,
    custom objects) pass through unchanged.
    """
    if isinstance(obj, float):
        return None if not math.isfinite(obj) else obj
    if isinstance(obj, dict):
        return {k: sanitize_floats(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_floats(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(sanitize_floats(v) for v in obj)
    return obj


def parse_last_json_line(text: str) -> Optional[dict]:
    """Last `{...}` line in `text`, parsed. None if no line is valid
    JSON. Non-JSON lines after the result don't cause false negatives."""
    if not text:
        return None
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None
