# Copyright 2025 Huawei Technologies Co., Ltd
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

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from secrets import token_hex

from ai_kernel_generator.cli.cli.utils.paths import get_panel_log_base_dir
from textual import log


@dataclass(frozen=True)
class SessionRoots:
    saved_root: Path
    tmp_root: Path


def get_session_roots() -> SessionRoots:
    base = get_panel_log_base_dir()
    return SessionRoots(
        saved_root=base / "sessions",
        tmp_root=base / "sessions_tmp",
    )


def new_session_id() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    pid = os.getpid()
    return f"{ts}_{pid}_{token_hex(4)}"


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def tmp_session_dir(session_id: str) -> Path:
    roots = get_session_roots()
    return _ensure_dir(roots.tmp_root) / str(session_id)


def saved_session_dir(session_id: str) -> Path:
    roots = get_session_roots()
    return _ensure_dir(roots.saved_root) / str(session_id)


def commit_session(session_id: str) -> Path:
    """把临时会话目录移动到 saved 目录，返回最终目录。"""
    src = tmp_session_dir(session_id)
    dst = saved_session_dir(session_id)
    if not src.exists():
        raise FileNotFoundError(f"临时会话不存在: {src}")
    if dst.exists():
        raise FileExistsError(f"会话已存在: {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    return dst


def discard_session(session_id: str) -> None:
    """删除临时会话目录（best-effort）。"""
    src = tmp_session_dir(session_id)
    try:
        if src.exists():
            shutil.rmtree(src, ignore_errors=True)
    except Exception as e:
        log.debug(
            "[Session] discard_session failed; ignore",
            session_id=str(session_id or ""),
            exc_info=e,
        )
        return


def resolve_session_dir_for_resume(session_id: str) -> Path:
    """resume 时只允许读取已保存的会话。"""
    p = saved_session_dir(session_id)
    if not p.exists():
        raise FileNotFoundError(f"未找到已保存会话: {p}")
    return p
