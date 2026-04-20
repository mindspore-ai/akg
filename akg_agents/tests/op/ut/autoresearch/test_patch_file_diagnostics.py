# Copyright 2025-2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for ``edit`` tool diagnostics — the behaviors the autoresearch
agent relies on to recover from failed edits without burning an extra
turn on re-reading the file.

Historical note: these used to test the standalone ``patch_file`` tool
which has been folded into ``edit``. File name kept for continuity with
the original test ledger.
"""
import os
from types import SimpleNamespace

from akg_agents.op.autoresearch.agent.tools import execute_edit


def _cfg(editable):
    return SimpleNamespace(
        editable_files=list(editable),
        forbidden_patterns={},
        banned_args={},
        dsl=None,
        backend=None,
        framework=None,
        max_patch_size=10_000_000,
    )


def _write(tmp_path, name, content):
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return str(p)


def _edit(tmp_path, editable, **kw):
    """Invoke execute_edit with single-edit shorthand. Matches the
    argument shape the LLM passes through build_tool_handlers."""
    return execute_edit(
        path=kw.pop("path", "f.py"),
        task_dir=str(tmp_path),
        config=_cfg(editable),
        **kw,
    )


def test_whitespace_mismatch_reports_actual_line(tmp_path):
    _write(tmp_path, "f.py", "def foo():\n    return 1\n")
    # agent-side old_str uses a tab where the file uses 4 spaces
    r = _edit(tmp_path, ["f.py"], mode="exact",
              old_str="\treturn 1", new_str="\treturn 2")
    assert not r.ok
    assert "whitespace/indent mismatch" in r.message
    assert "line 2" in r.message
    # Must not trigger stale-marking upstream
    assert "old_str not found" not in r.message


def test_multi_match_enumerates_line_numbers(tmp_path):
    src = "a = 1\nreturn None\nb = 2\nreturn None\nc = 3\n"
    _write(tmp_path, "f.py", src)
    r = _edit(tmp_path, ["f.py"], mode="exact",
              old_str="return None", new_str="return 0")
    assert not r.ok
    assert "appears 2 times" in r.message
    assert "L2" in r.message and "L4" in r.message
    assert "anchor_line" in r.message
    assert "old_str not found" not in r.message


def test_anchor_line_disambiguates(tmp_path):
    # Two occurrences >10 lines apart so anchor can resolve uniquely within ±5.
    src = "return None\n" + "x\n" * 15 + "return None\n"
    path = _write(tmp_path, "f.py", src)
    r = _edit(tmp_path, ["f.py"], mode="exact",
              old_str="return None", new_str="return 0",
              anchor_line=17)  # close to the second occurrence (L17), far from L1
    assert r.ok, r.message
    after = open(path, encoding="utf-8").read()
    assert after == "return None\n" + "x\n" * 15 + "return 0\n"


def test_anchor_line_ambiguous_when_both_in_window(tmp_path):
    # Both occurrences within ±5 of the anchor → must refuse rather than
    # silently pick one.
    src = "a = 1\nreturn None\nb = 2\nreturn None\nc = 3\n"
    _write(tmp_path, "f.py", src)
    r = _edit(tmp_path, ["f.py"], mode="exact",
              old_str="return None", new_str="return 0",
              anchor_line=3)
    assert not r.ok
    assert "ambiguous within" in r.message


def test_anchor_line_too_far_retries_widened(tmp_path):
    # Anchor far from both matches — exact lookup rejects (not within
    # ±5), then the dispatcher widens to ±15 which is still ambiguous
    # (both matches fall inside), so the final error calls out the
    # widened attempt.
    src = "a = 1\nreturn None\n" + "x\n" * 20 + "return None\n"
    _write(tmp_path, "f.py", src)
    r = _edit(tmp_path, ["f.py"], mode="exact",
              old_str="return None", new_str="return 0",
              anchor_line=12)  # ±5 misses both; ±15 still ambiguous
    assert not r.ok
    # Either the ±5 rejection or the widened rejection is visible.
    assert ("not within ±5" in r.message
            or "widened anchor" in r.message)


def test_genuine_missing_still_marks_stale(tmp_path):
    _write(tmp_path, "f.py", "def foo():\n    return 1\n")
    r = _edit(tmp_path, ["f.py"], mode="exact",
              old_str="totally_unrelated_symbol_xyz", new_str="bar")
    assert not r.ok
    # Literal phrase is the stale-marking token consumed by turn.py
    assert "old_str not found" in r.message


def test_exact_unique_match_still_succeeds(tmp_path):
    path = _write(tmp_path, "f.py", "def foo():\n    return 1\n")
    r = _edit(tmp_path, ["f.py"], mode="exact",
              old_str="    return 1", new_str="    return 42")
    assert r.ok, r.message
    assert open(path, encoding="utf-8").read() == "def foo():\n    return 42\n"
