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

"""Manifest 目录定位规则。"""

from __future__ import annotations

import hashlib
import inspect
import re
from pathlib import Path

_FUTURE_IMPORT_RE = re.compile(
    r"^\s*from\s+__future__\s+import\s+[^\n]*\n",
    re.MULTILINE,
)


def default_manifest_root() -> Path:
    return Path.home() / ".cache" / "akg_agents" / "dynamic_tune" / "manifests"


def manifest_dir_for_source(source_path: str | Path) -> Path:
    path = Path(source_path).expanduser().resolve()
    text = path.read_text(encoding="utf-8")
    return manifest_dir_for_source_text(text)


def manifest_dir_for_source_text(source_text: str) -> Path:
    digest = hashlib.sha256(_normalize_source(source_text).encode("utf-8")).hexdigest()
    return default_manifest_root() / digest[:16]


def manifest_dir_for_caller() -> Path:
    frame = inspect.currentframe()
    caller = frame.f_back.f_back if frame is not None and frame.f_back is not None else None
    if caller is None:
        raise RuntimeError("无法定位 load_deployed_selector 调用方")
    filename = caller.f_code.co_filename
    if filename.startswith("<"):
        raise RuntimeError("load_deployed_selector() 无参调用需要可读取的调用方源码文件")
    return manifest_dir_for_source(filename)


def _normalize_source(source_text: str) -> str:
    return _FUTURE_IMPORT_RE.sub("", source_text).replace("\r\n", "\n")


__all__ = [
    "default_manifest_root",
    "manifest_dir_for_caller",
    "manifest_dir_for_source",
    "manifest_dir_for_source_text",
]
