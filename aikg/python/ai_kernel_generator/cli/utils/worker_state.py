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

import json
import os
import signal
import time
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from .paths import get_process_log_dir

logger = logging.getLogger(__name__)


_STATE_VERSION = 1


def _state_path() -> Path:
    return get_process_log_dir() / "worker_state.json"


def _default_state() -> Dict[str, Any]:
    return {"version": _STATE_VERSION, "workers": {}}


def load_worker_state() -> Dict[str, Any]:
    path = _state_path()
    if not path.exists():
        return _default_state()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.debug("[WorkerState] load failed; fallback default", exc_info=e)
        return _default_state()
    if not isinstance(data, dict):
        return _default_state()
    workers = data.get("workers")
    if not isinstance(workers, dict):
        data["workers"] = {}
    return data


def save_worker_state(state: Dict[str, Any]) -> None:
    path = _state_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(state, ensure_ascii=True, indent=2), encoding="utf-8"
        )
    except Exception as e:
        logger.debug("[WorkerState] save failed", exc_info=e)


def get_worker_entry(state: Dict[str, Any], port: int) -> Optional[Dict[str, Any]]:
    workers = state.get("workers")
    if not isinstance(workers, dict):
        return None
    entry = workers.get(str(port))
    return entry if isinstance(entry, dict) else None


def set_worker_entry(state: Dict[str, Any], port: int, entry: Dict[str, Any]) -> None:
    workers = state.get("workers")
    if not isinstance(workers, dict):
        workers = {}
        state["workers"] = workers
    workers[str(port)] = entry


def remove_worker_entry(state: Dict[str, Any], port: int) -> None:
    workers = state.get("workers")
    if not isinstance(workers, dict):
        return
    workers.pop(str(port), None)


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _send_signal(pid: int, sig: int) -> None:
    if hasattr(os, "killpg"):
        try:
            pgid = os.getpgid(pid)
            os.killpg(pgid, sig)
            return
        except Exception:
            pass
    os.kill(pid, sig)


def terminate_pid(pid: int, timeout: float = 5.0) -> bool:
    if not pid_alive(pid):
        return True
    try:
        _send_signal(pid, signal.SIGTERM)
    except ProcessLookupError:
        return True
    deadline = time.time() + float(timeout)
    while time.time() < deadline:
        if not pid_alive(pid):
            return True
        time.sleep(0.2)
    try:
        _send_signal(pid, signal.SIGKILL)
    except ProcessLookupError:
        return True
    return not pid_alive(pid)
