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

"""AST-level reference.py validation — pure library module.

A reference module must define `class Model` + `get_init_inputs()` plus
*one of*:
  - `get_inputs()`           — single-shape (legacy)
  - `get_input_groups()`     — multi-shape (NPUKernelBench style)

Both forms are accepted; downstream `input_groups.resolve` collapses them
to a `List[Tuple]`. Used both at scaffold time (to reject obviously
invalid pasted reference code before writing the task dir) and at runtime
(by validators.validate_reference, as the static-symbol stage that runs
before the subprocess import-and-run check).

This module is INTENTIONALLY dependency-free and CLI-free: it imports
only `ast` from stdlib, exposes one function, and never grows. Both
scaffold.py and phase_machine.validators consume it; nothing else
should.
"""
from __future__ import annotations

import ast


REQUIRED_REF_SYMBOLS = (
    ("Model", "class Model"),
    ("get_init_inputs", "get_init_inputs()"),
)
INPUT_PROVIDERS = ("get_inputs", "get_input_groups")


def validate_ref(code: str, source: str = "reference") -> None:
    """Raise ValueError if `code` is missing any required reference symbol.

    Returns None on pass — keeping the (no return value, raises on
    failure) shape that scaffold and validators both already use.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise ValueError(
            f"Reference from {source} has syntax error: {e}"
        ) from e

    names = {
        node.name for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
    }
    missing = [label for name, label in REQUIRED_REF_SYMBOLS if name not in names]
    if missing:
        raise ValueError(
            f"Reference from {source} missing: {', '.join(missing)}"
        )
    if not any(p in names for p in INPUT_PROVIDERS):
        raise ValueError(
            f"Reference from {source} missing input provider: define one of "
            f"{', '.join(p + '()' for p in INPUT_PROVIDERS)}"
        )
