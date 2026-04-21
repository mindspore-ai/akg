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
"""Post-write semantic checks (``validate_patch_result``)."""

from akg_agents.op.autoresearch.agent.patch_validator import (
    validate_patch_result,
)


def test_valid_python_edit_passes():
    old = "def f():\n    return 1\n"
    new = "def f():\n    return 2\n"
    assert validate_patch_result(old, new, "f.py") is None


def test_syntax_error_caught():
    old = "def f():\n    return 1\n"
    new = "def f():\n    return ((1\n"
    err = validate_patch_result(old, new, "f.py")
    assert err is not None
    assert "SyntaxError" in err


def test_empty_content_rejected():
    old = "x = 1\n"
    new = ""
    err = validate_patch_result(old, new, "f.py")
    assert err is not None
    assert "empty" in err.lower()


def test_noop_rejected():
    old = "x = 1\n"
    assert validate_patch_result(old, old, "f.py") is not None


def test_indent_explosion_caught():
    old = "def f():\n    return 1\n"
    # Artificially deep nested indent that still parses (inside a
    # function body). Old depth = 1, new depth = 5 → jump > 2.
    new = (
        "def f():\n"
        "    if True:\n"
        "        if True:\n"
        "            if True:\n"
        "                if True:\n"
        "                    return 1\n"
    )
    err = validate_patch_result(old, new, "f.py")
    assert err is not None
    assert "Indentation jumped" in err


def test_delta_drift_caught():
    old = "a = 1\n"
    # Added 9 lines when we claimed +1 — outside tolerance.
    new = old + "\n".join(f"x{i} = {i}" for i in range(10)) + "\n"
    err = validate_patch_result(old, new, "f.py", expected_delta_lines=1)
    assert err is not None
    assert "Line-count drift" in err


def test_delta_drift_within_tolerance_passes():
    old = "a = 1\n"
    # +2 lines claimed, actual +3 → within tolerance.
    new = "a = 1\nb = 2\nc = 3\nd = 4\n"
    assert validate_patch_result(old, new, "f.py", expected_delta_lines=2) is None


def test_non_python_skips_ast_check():
    # Invalid Python but a .c file — AST not run, should pass the syntax stage.
    old = "int main() { return 0; }\n"
    new = "int main() { return ((0; }\n"
    # Validator skips ast.parse for non-.py files, but the no-op /
    # empty checks still run. This new content is non-empty and
    # different, and indent depth is flat, so validate returns None.
    assert validate_patch_result(old, new, "main.c") is None
