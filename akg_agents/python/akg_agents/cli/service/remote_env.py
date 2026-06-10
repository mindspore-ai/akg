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

"""Shared remote shell environment bootstrap helpers."""

from __future__ import annotations

import shlex
from typing import Optional


_CONDA_HOOK_BASH = r"""
if command -v conda >/dev/null 2>&1; then
  __akg_conda_base="$(conda info --base 2>/dev/null || true)"
  if [ -n "$__akg_conda_base" ] && [ -f "$__akg_conda_base/etc/profile.d/conda.sh" ]; then
    . "$__akg_conda_base/etc/profile.d/conda.sh" >/dev/null 2>&1 || true
  else
    eval "$(conda shell.bash hook 2>/dev/null)" >/dev/null 2>&1 || true
  fi
  unset __akg_conda_base
fi
""".strip()


def conda_hook_bash() -> str:
    """Return a bash snippet that enables ``conda activate`` in SSH shells."""
    return _CONDA_HOOK_BASH


def source_env_script_bash(env_script: Optional[str]) -> str:
    """Return bash that prepares conda and sources a literal env script path."""
    parts = [conda_hook_bash()]
    if env_script:
        parts.append(f"source {shlex.quote(env_script)}")
    return "\n".join(parts)


def source_env_var_bash(var_name: str) -> str:
    """Return bash that prepares conda and sources ``$var_name`` if present."""
    return "\n".join([
        conda_hook_bash(),
        f'if [ -n "${var_name}" ] && [ -f "${var_name}" ]; then',
        f'  source "${var_name}" >/dev/null 2>&1',
        "fi",
    ])
