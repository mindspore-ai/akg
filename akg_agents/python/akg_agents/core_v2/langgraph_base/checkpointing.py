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

"""LangGraph debug checkpoint helpers.

LangGraph 1.x exposes the checkpointing contract through a checkpointer
passed to ``StateGraph.compile`` and a ``configurable.thread_id`` at
invocation time.  AKG keeps that native contract, but provides a lightweight
file-backed saver so save/resume works without requiring optional database
packages in developer environments.
"""

from __future__ import annotations

import logging
import os
import pickle
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from langgraph.checkpoint.memory import MemorySaver

logger = logging.getLogger(__name__)

_DEFAULT_CHECKPOINT_DIR = "~/.akg/langgraph_checkpoints"


class FileCheckpointSaver(MemorySaver):
    """Persistent LangGraph checkpointer backed by one local pickle file.

    The serialized file contains LangGraph's own typed blobs as produced by
    ``MemorySaver``.  It is intended for trusted local debug artefacts only;
    callers should not load checkpoint files from untrusted sources.
    """

    def __init__(self, checkpoint_file: str):
        super().__init__()
        self.checkpoint_file = Path(checkpoint_file).expanduser()
        self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        self._load_from_disk()

    def _load_from_disk(self) -> None:
        if not self.checkpoint_file.exists():
            return
        try:
            with self.checkpoint_file.open("rb") as f:
                payload = pickle.load(f)
            self.storage.update(payload.get("storage", {}))
            self.writes.update(payload.get("writes", {}))
            self.blobs.update(payload.get("blobs", {}))
            logger.info(f"[LangGraphDebug] loaded checkpoint: {self.checkpoint_file}")
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load LangGraph checkpoint file {self.checkpoint_file}: {exc}"
            ) from exc

    def _flush_to_disk(self) -> None:
        payload = {
            "storage": dict(self.storage),
            "writes": dict(self.writes),
            "blobs": dict(self.blobs),
        }
        fd, tmp_name = tempfile.mkstemp(
            prefix=self.checkpoint_file.name + ".",
            suffix=".tmp",
            dir=str(self.checkpoint_file.parent),
        )
        try:
            with os.fdopen(fd, "wb") as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_name, self.checkpoint_file)
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)

    def put(self, config, checkpoint, metadata, new_versions):
        next_config = super().put(config, checkpoint, metadata, new_versions)
        self._flush_to_disk()
        return next_config

    def put_writes(self, config, writes, task_id, task_path=""):
        result = super().put_writes(config, writes, task_id, task_path)
        self._flush_to_disk()
        return result

    async def aput(self, config, checkpoint, metadata, new_versions):
        next_config = await super().aput(config, checkpoint, metadata, new_versions)
        self._flush_to_disk()
        return next_config

    async def aput_writes(self, config, writes, task_id, task_path=""):
        result = await super().aput_writes(config, writes, task_id, task_path)
        self._flush_to_disk()
        return result


def get_debug_config(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    raw = dict((config or {}).get("debug") or {})
    # Backward-compatible flat keys are useful for CLI and tests.
    for flat_key, nested_key in {
        "debug_enabled": "enabled",
        "debug_resume": "resume",
        "debug_session_id": "session_id",
        "debug_checkpoint_dir": "checkpoint_dir",
        "debug_checkpoint_file": "checkpoint_file",
    }.items():
        if flat_key in (config or {}) and nested_key not in raw:
            raw[nested_key] = (config or {})[flat_key]
    return raw


def debug_enabled(config: Optional[Dict[str, Any]]) -> bool:
    dbg = get_debug_config(config)
    return bool(dbg.get("enabled") or dbg.get("resume"))


def debug_resume_requested(config: Optional[Dict[str, Any]]) -> bool:
    return bool(get_debug_config(config).get("resume"))


def get_debug_thread_id(config: Optional[Dict[str, Any]]) -> str:
    dbg = get_debug_config(config)
    thread_id = (
        dbg.get("thread_id")
        or dbg.get("session_id")
        or (config or {}).get("_langgraph_debug_thread_id")
        or (config or {}).get("session_id")
        or (config or {}).get("_langgraph_debug_task_id")
        or "akg-langgraph-debug"
    )
    return str(thread_id)


def get_debug_checkpoint_file(config: Optional[Dict[str, Any]]) -> str:
    dbg = get_debug_config(config)
    explicit_file = dbg.get("checkpoint_file")
    if explicit_file:
        return str(Path(str(explicit_file)).expanduser())

    checkpoint_dir = Path(str(dbg.get("checkpoint_dir") or _DEFAULT_CHECKPOINT_DIR)).expanduser()
    thread_id = _safe_file_component(get_debug_thread_id(config))
    workflow_name = _safe_file_component(str((config or {}).get("_langgraph_debug_workflow") or "workflow"))
    return str(checkpoint_dir / f"{workflow_name}_{thread_id}.pkl")


def build_debug_checkpointer(config: Optional[Dict[str, Any]]) -> Optional[FileCheckpointSaver]:
    if not debug_enabled(config):
        return None
    return FileCheckpointSaver(get_debug_checkpoint_file(config))


def build_invoke_config(config: Dict[str, Any], recursion_limit: int) -> Dict[str, Any]:
    invoke_config: Dict[str, Any] = {"recursion_limit": recursion_limit}
    if debug_enabled(config):
        invoke_config["configurable"] = {"thread_id": get_debug_thread_id(config)}
    return invoke_config


async def get_existing_debug_state(app: Any, invoke_config: Dict[str, Any]) -> Dict[str, Any]:
    snapshot = await app.aget_state(invoke_config)
    values = getattr(snapshot, "values", {}) or {}
    return dict(values)


def _safe_file_component(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)
    return safe.strip("._") or "debug"

