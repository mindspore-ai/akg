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

"""Resolve CATLASS_ROOT and catlass_op source directory for ascendc_catlass verify."""

from __future__ import annotations

import os
import re
from typing import Any, Dict, Optional

_AKG_CATLASS_CMAKE_MARKER = "_AKG_CATLASS_ROOT"
# Resolution order matches Python-side ``resolve_catlass_root``:
#   1. -DCATLASS_ROOT=... (forwarded by adapter from task.yaml catlass.root)
#   2. ENV{CATLASS_ROOT}
# Python pre-resolves to ``<akg-root>/thirdparty/catlass`` as a default
# and forwards via -D; cmake never reaches FATAL_ERROR in the standard
# deployment. The fatal-error branch stays as a clear failure message
# for fully-custom (no env, no -D, no thirdparty/) deployments.
_AKG_CATLASS_ROOT_BLOCK = f"""# Resolved by akg_agents (task.yaml catlass.root / -DCATLASS_ROOT / ENV{{CATLASS_ROOT}} / <akg-root>/thirdparty/catlass)
if(DEFINED CATLASS_ROOT AND NOT "${{CATLASS_ROOT}}" STREQUAL "")
  set({_AKG_CATLASS_CMAKE_MARKER} "${{CATLASS_ROOT}}")
elseif(DEFINED ENV{{CATLASS_ROOT}} AND NOT "$ENV{{CATLASS_ROOT}}" STREQUAL "")
  set({_AKG_CATLASS_CMAKE_MARKER} "$ENV{{CATLASS_ROOT}}")
else()
  message(FATAL_ERROR "CATLASS_ROOT not set. Pass -DCATLASS_ROOT, export CATLASS_ROOT, or install catlass at <akg-root>/thirdparty/catlass via `bash download.sh --with_catlass`.")
endif()
set(CATLASS_ROOT "${{{_AKG_CATLASS_CMAKE_MARKER}}}")
"""

_RELATIVE_CATLASS_ROOT_RE = re.compile(
    r"^\s*set\s*\(\s*CATLASS_ROOT\s+\$\{CMAKE_CURRENT_SOURCE_DIR\}/.*\)\s*$",
    re.MULTILINE,
)


def resolve_catlass_op_src(
    *,
    task_dir: Optional[str] = None,
    catlass_op_src: Optional[str] = None,
    catlass_op_dir: Optional[str] = None,
) -> Optional[str]:
    """Return absolute path to catlass_op project directory (folder).

    Priority: explicit ``catlass_op_src`` > ``task_dir`` + ``catlass_op_dir``.
    ``catlass_op_dir`` may be absolute or relative to ``task_dir`` (default name
    ``catlass_op``).
    """
    if catlass_op_src:
        path = os.path.abspath(catlass_op_src)
        return path if os.path.isdir(path) else None

    rel = catlass_op_dir or "catlass_op"
    if os.path.isabs(rel):
        return rel if os.path.isdir(rel) else None

    if not task_dir:
        return None
    path = os.path.abspath(os.path.join(task_dir, rel))
    return path if os.path.isdir(path) else None


def _bundled_catlass_root() -> Optional[str]:
    """Standard install location: ``<akg-root>/thirdparty/catlass`` (cloned
    by ``download.sh --with_catlass``). Lets operators skip CATLASS_ROOT
    config entirely when catlass is installed at this canonical path."""
    # __file__ = .../akg_agents/python/akg_agents/op/utils/catlass_paths.py
    # akg root  = .../akg_agents/
    akg_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), *([os.pardir] * 4))
    )
    cand = os.path.join(akg_root, "thirdparty", "catlass")
    return cand if os.path.isdir(os.path.join(cand, "include", "catlass")) else None


def resolve_catlass_root(
    *,
    catlass_root: Optional[str] = None,
) -> Optional[str]:
    """Return absolute CATLASS repo root (directory containing
    ``include/catlass``). Resolution order:

      1. Explicit ``catlass_root`` arg (task.yaml ``catlass.root``)
      2. ``CATLASS_ROOT`` env var (worker daemon env)
      3. ``<akg-root>/thirdparty/catlass`` (the standard install location
         via ``download.sh --with_catlass``) — kicks in when neither 1
         nor 2 is set, so the common deployment needs zero config.
    """
    root = catlass_root or os.environ.get("CATLASS_ROOT")
    if root:
        path = os.path.abspath(root)
        include_catlass = os.path.join(path, "include", "catlass")
        if os.path.isdir(include_catlass):
            return path
        return None
    return _bundled_catlass_root()


def merge_catlass_config(
    config: Dict[str, Any],
    *,
    task_dir: Optional[str] = None,
    task_info: Optional[Dict[str, Any]] = None,
) -> None:
    """Fill ``config`` in-place with ``catlass_op_src`` / ``catlass_root`` when missing."""
    task_info = task_info or {}
    td = task_dir or task_info.get("task_dir") or config.get("task_dir")

    if not config.get("catlass_op_src"):
        op_src = resolve_catlass_op_src(
            task_dir=td,
            catlass_op_src=config.get("catlass_op_src")
            or task_info.get("catlass_op_src"),
            catlass_op_dir=config.get("catlass_op_dir")
            or task_info.get("catlass_op_dir"),
        )
        if op_src:
            config["catlass_op_src"] = op_src

    if not config.get("catlass_root"):
        root = resolve_catlass_root(
            catlass_root=config.get("catlass_root")
            or task_info.get("catlass_root"),
        )
        if root:
            config["catlass_root"] = root


def patch_catlass_op_cmake(catlass_op_dir: str) -> bool:
    """Replace ``CATLASS_ROOT = CMAKE_CURRENT_SOURCE_DIR/..`` with config/env resolution.

    Idempotent: skips CMakeLists that already use the akg block or ``-DCATLASS_ROOT`` priority.
    """
    cmake_path = os.path.join(os.path.abspath(catlass_op_dir), "CMakeLists.txt")
    if not os.path.isfile(cmake_path):
        return False

    with open(cmake_path, "r", encoding="utf-8") as f:
        text = f.read()

    if _AKG_CATLASS_CMAKE_MARKER in text or "# Priority: cmake -DCATLASS_ROOT" in text:
        return False

    if not _RELATIVE_CATLASS_ROOT_RE.search(text):
        return False

    text = _RELATIVE_CATLASS_ROOT_RE.sub(_AKG_CATLASS_ROOT_BLOCK.rstrip(), text, count=1)
    with open(cmake_path, "w", encoding="utf-8") as f:
        f.write(text)
    return True
