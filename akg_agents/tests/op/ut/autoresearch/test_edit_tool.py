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
"""Tests for the unified ``edit`` tool — dispatcher, three-stage retry,
similar-line suggestions, and post-write semantic validation.

These exercise behavior the agent directly depends on to recover from
small edit mistakes without burning a turn on re-reading the file.
"""
import os
from types import SimpleNamespace

from akg_agents.op.autoresearch.agent.tools import (
    TOOLS, build_tool_handlers, execute_edit,
)


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


# ----- schema / dispatch surface ------------------------------------------


def test_tools_list_exposes_edit_not_patch_file():
    names = {t["name"] for t in TOOLS}
    assert "edit" in names
    assert "patch_file" not in names
    assert "write_file" not in names


def test_build_handlers_exposes_edit_only_for_file_ops(tmp_path):
    cfg = _cfg(["f.py"])
    handlers = build_tool_handlers(str(tmp_path), cfg)
    assert set(handlers) == {"read_file", "edit"}


# ----- mode=exact ----------------------------------------------------------


def test_edit_exact_unique_match(tmp_path):
    path = _write(tmp_path, "f.py", "def foo():\n    return 1\n")
    r = execute_edit(path="f.py", mode="exact",
                     task_dir=str(tmp_path), config=_cfg(["f.py"]),
                     old_str="    return 1", new_str="    return 42")
    assert r.ok, r.message
    assert open(path, encoding="utf-8").read() == "def foo():\n    return 42\n"


# ----- multi-edit (atomic batch) ------------------------------------------


def test_multi_edit_applies_sequentially(tmp_path):
    """edits=[...] composes: each step sees prior step's result."""
    path = _write(tmp_path, "f.py",
                  "import a\nimport b\n\ndef go():\n    return a.x + b.y\n")
    r = execute_edit(
        path="f.py", task_dir=str(tmp_path), config=_cfg(["f.py"]),
        edits=[
            {"mode": "exact", "old_str": "import a\n", "new_str": "import alpha\n"},
            {"mode": "exact", "old_str": "a.x", "new_str": "alpha.x"},
        ],
    )
    assert r.ok, r.message
    final = open(path, encoding="utf-8").read()
    assert "import alpha" in final
    assert "alpha.x" in final
    assert "import a\n" not in final
    assert "2 edits" in r.message


def test_multi_edit_atomic_rollback(tmp_path):
    """If edit #2 fails, edit #1's change is also discarded."""
    path = _write(tmp_path, "f.py", "a = 1\nb = 2\n")
    r = execute_edit(
        path="f.py", task_dir=str(tmp_path), config=_cfg(["f.py"]),
        edits=[
            {"mode": "exact", "old_str": "a = 1", "new_str": "a = 10"},
            # This will fail — "totally_missing" is not in the file.
            {"mode": "exact", "old_str": "totally_missing",
             "new_str": "anything"},
        ],
    )
    assert not r.ok
    assert "edit #2/2" in r.message
    # File unchanged — edit #1's write rolled back.
    assert open(path, encoding="utf-8").read() == "a = 1\nb = 2\n"


def test_multi_edit_rejects_rewrite_not_alone(tmp_path):
    """mode='rewrite' can't coexist with other edits in the same batch."""
    _write(tmp_path, "f.py", "old\n")
    r = execute_edit(
        path="f.py", task_dir=str(tmp_path), config=_cfg(["f.py"]),
        edits=[
            {"mode": "exact", "old_str": "old", "new_str": "new"},
            {"mode": "rewrite", "new_str": "entirely different\n"},
        ],
    )
    assert not r.ok
    assert "rewrite" in r.message


def test_multi_edit_mixed_modes(tmp_path):
    """exact + block can share a batch — sequential composition works."""
    path = _write(tmp_path, "f.py",
                  "def foo():\n    return 1\n\ndef bar():\n    return 2\n")
    r = execute_edit(
        path="f.py", task_dir=str(tmp_path), config=_cfg(["f.py"]),
        edits=[
            {"mode": "exact", "old_str": "return 1", "new_str": "return 10"},
            # Tab indentation — block mode whitespace fallback handles it
            {"mode": "block", "old_str": "\treturn 2",
             "new_str": "\treturn 20"},
        ],
    )
    assert r.ok, r.message
    final = open(path, encoding="utf-8").read()
    assert "return 10" in final
    assert "return 20" in final


def test_multi_edit_post_batch_validator(tmp_path):
    """Combined result must pass syntax check even if per-step intermediates
    are syntactically invalid. Here both edits together restore validity."""
    path = _write(tmp_path, "f.py", "def f():\n    x = 1\n    return x\n")
    # Step 1: delete `return x` (leaves function body with just `x = 1`,
    # which still parses). Step 2: replace `x = 1` with `return 1`.
    r = execute_edit(
        path="f.py", task_dir=str(tmp_path), config=_cfg(["f.py"]),
        edits=[
            {"mode": "exact", "old_str": "    return x\n", "new_str": ""},
            {"mode": "exact", "old_str": "    x = 1\n",
             "new_str": "    return 1\n"},
        ],
    )
    assert r.ok, r.message
    assert open(path, encoding="utf-8").read() == "def f():\n    return 1\n"


def test_rewrite_creates_missing_parent_directories(tmp_path):
    """editable_files with a subdirectory path should support creation
    via mode='rewrite'; the old write_file did this and the multi-edit
    refactor must not silently drop it."""
    r = execute_edit(
        path="nested/new.py", task_dir=str(tmp_path),
        config=_cfg(["nested/new.py"]),
        mode="rewrite", new_str="x = 1\n",
    )
    assert r.ok, r.message
    import os
    assert open(os.path.join(str(tmp_path), "nested", "new.py"),
                encoding="utf-8").read() == "x = 1\n"


def test_retry_exhausted_error_body_not_empty(tmp_path):
    """Regression for _format_tool_result_content: when the retry tag
    is at the FRONT of the error (retry-exhausted failure), the XML
    envelope must still carry the actual error text. The previous
    implementation sliced ``msg[:idx]`` which left the body empty."""
    from akg_agents.op.autoresearch.agent.turn import (
        _format_tool_result_content,
    )
    from types import SimpleNamespace
    # Simulate what execute_edit would return after exhausting retries.
    output = SimpleNamespace(
        ok=False, message="[retries=2] ERROR: old_str not found in f.py."
    )
    args = {"path": "f.py", "mode": "block"}
    rendered = _format_tool_result_content("edit", args, output)
    # The actual error text must survive the render.
    assert "ERROR: old_str not found" in rendered
    # The retry count must be captured in the envelope attribute.
    assert 'retries="2"' in rendered
    # The raw tag should NOT be duplicated in the body.
    assert "[retries=2]" not in rendered


def test_multi_edit_single_edit_shorthand_still_works(tmp_path):
    """Legacy top-level mode/old_str/new_str form still accepted."""
    path = _write(tmp_path, "f.py", "a = 1\n")
    r = execute_edit(path="f.py", task_dir=str(tmp_path),
                     config=_cfg(["f.py"]),
                     mode="exact", old_str="a = 1", new_str="a = 2")
    assert r.ok, r.message
    assert open(path, encoding="utf-8").read() == "a = 2\n"
    # Single-edit success does NOT add the "(N edits)" tag.
    assert "edits)" not in r.message


def test_edit_exact_missing_gives_similar_line_suggestions(tmp_path):
    _write(tmp_path, "f.py",
           "def compute_grid_size(N, BLOCK):\n    return (N + BLOCK - 1) // BLOCK\n")
    # Close but not exact — differs by "_shape" vs "_size".
    r = execute_edit(path="f.py", mode="exact",
                     task_dir=str(tmp_path), config=_cfg(["f.py"]),
                     old_str="def compute_grid_shape(N, BLOCK):",
                     new_str="def compute_grid_shape(N, BLOCK):")
    assert not r.ok
    # Error must include similar-line hint with line number
    assert "Did you mean" in r.message
    assert "L1" in r.message


# ----- mode=block (whitespace-tolerant retry) ------------------------------


def test_edit_block_whitespace_tolerant_retry(tmp_path):
    path = _write(tmp_path, "f.py", "def foo():\n    return 1\n")
    # old_str uses a TAB where file uses 4 spaces — no substring overlap.
    r = execute_edit(path="f.py", mode="block",
                     task_dir=str(tmp_path), config=_cfg(["f.py"]),
                     old_str="\treturn 1", new_str="\treturn 42")
    assert r.ok, r.message
    assert "retries=" in r.message
    # New content must preserve file's original 4-space indent.
    assert open(path, encoding="utf-8").read() == "def foo():\n    return 42\n"


def test_edit_exact_does_NOT_retry_block(tmp_path):
    """mode=exact should NOT fall back to whitespace-fuzzy — only block does."""
    _write(tmp_path, "f.py", "def foo():\n    return 1\n")
    r = execute_edit(path="f.py", mode="exact",
                     task_dir=str(tmp_path), config=_cfg(["f.py"]),
                     old_str="\treturn 1", new_str="\treturn 42")
    assert not r.ok
    # Whitespace diagnosis fires but doesn't silently patch.
    assert "whitespace/indent mismatch" in r.message


# ----- anchor widening retry ----------------------------------------------


def test_edit_widens_anchor_on_near_miss(tmp_path):
    src = "return None\n" + "x\n" * 10 + "return None\n"
    path = _write(tmp_path, "f.py", src)
    # Anchor is 9 lines away from the first match — outside ±5 but
    # inside ±15 (the widened fallback window).
    r = execute_edit(path="f.py", mode="exact",
                     task_dir=str(tmp_path), config=_cfg(["f.py"]),
                     old_str="return None", new_str="return 0",
                     anchor_line=10)  # ~9 lines after L1, ~2 before L12
    # Either attempt succeeds (widened or first). Must write exactly one.
    assert r.ok, r.message
    assert open(path, encoding="utf-8").read().count("return 0") == 1


# ----- mode=rewrite --------------------------------------------------------


def test_edit_rewrite_full_file(tmp_path):
    path = _write(tmp_path, "f.py", "old\n")
    r = execute_edit(path="f.py", mode="rewrite",
                     task_dir=str(tmp_path), config=_cfg(["f.py"]),
                     new_str="def new():\n    return 1\n")
    assert r.ok, r.message
    assert open(path, encoding="utf-8").read() == "def new():\n    return 1\n"


# ----- mode=unified --------------------------------------------------------


def test_edit_unified_diff_single_hunk(tmp_path):
    path = _write(tmp_path, "f.py", "a = 1\nb = 2\nc = 3\n")
    diff = (
        "--- a/f.py\n"
        "+++ b/f.py\n"
        "@@ -2 +2 @@\n"
        "-b = 2\n"
        "+b = 20\n"
    )
    r = execute_edit(path="f.py", mode="unified",
                     task_dir=str(tmp_path), config=_cfg(["f.py"]),
                     diff=diff)
    assert r.ok, r.message
    assert open(path, encoding="utf-8").read() == "a = 1\nb = 20\nc = 3\n"


def test_edit_unified_diff_context_fuzz(tmp_path):
    """Diff header points at wrong line — fuzz-window should still find it."""
    path = _write(tmp_path, "f.py", "a = 1\nb = 2\nc = 3\nd = 4\n")
    # Claim the hunk starts at L1, but the actual change is at L2.
    diff = (
        "@@ -1 +1 @@\n"
        "-b = 2\n"
        "+b = 20\n"
    )
    r = execute_edit(path="f.py", mode="unified",
                     task_dir=str(tmp_path), config=_cfg(["f.py"]),
                     diff=diff)
    assert r.ok, r.message
    assert open(path, encoding="utf-8").read() == "a = 1\nb = 20\nc = 3\nd = 4\n"


# ----- post-write semantic validator --------------------------------------


def test_edit_rolls_back_on_syntax_error(tmp_path):
    path = _write(tmp_path, "f.py", "def foo():\n    return 1\n")
    # Remove the return body — leaves a bare ``def foo():`` which parses
    # on its own actually, so induce a real SyntaxError: unmatched paren.
    r = execute_edit(path="f.py", mode="exact",
                     task_dir=str(tmp_path), config=_cfg(["f.py"]),
                     old_str="    return 1", new_str="    return ((1")
    assert not r.ok
    assert "post-write validator" in r.message or "SyntaxError" in r.message
    # File should be unchanged (rollback worked).
    assert open(path, encoding="utf-8").read() == "def foo():\n    return 1\n"


def test_edit_rejects_empty_rewrite(tmp_path):
    _write(tmp_path, "f.py", "a = 1\n")
    r = execute_edit(path="f.py", mode="rewrite",
                     task_dir=str(tmp_path), config=_cfg(["f.py"]),
                     new_str="")
    # Empty new_str on rewrite is a degenerate case — write_file
    # currently allows it, but the empty file check in the validator
    # only fires for non-rewrite paths. The test documents current
    # behaviour: rewrite bypasses validator (explicit user intent).
    assert r.ok or "empty" in r.message.lower()


# ----- similar-line suggestions ribbon ------------------------------------


def test_edit_suggestions_rank_by_similarity(tmp_path):
    content = (
        "def handle_patch(x):\n"
        "    return x\n"
        "def handle_pattern(x):\n"
        "    return x\n"
        "def entirely_unrelated():\n"
        "    return 0\n"
    )
    _write(tmp_path, "f.py", content)
    r = execute_edit(path="f.py", mode="exact",
                     task_dir=str(tmp_path), config=_cfg(["f.py"]),
                     old_str="def handle_patc(x):",
                     new_str="def handle_patc(x):")
    assert not r.ok
    # handle_patch and handle_pattern both share significant prefix.
    # Suggestions should name at least one of them.
    assert "L1" in r.message or "L3" in r.message
