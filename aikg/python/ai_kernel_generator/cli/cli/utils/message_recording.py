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
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Optional

from textual import log


@dataclass(frozen=True)
class RecordedFrame:
    seq: int
    ts: float  # time.time()
    mono: float  # time.monotonic()
    payload: dict[str, Any]


class MessageRecorder:
    """录制 server->cli 的原始 JSON frame（jsonl）。"""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._seq = 0
        self._disabled = False
        self._error_logged = False

    def _disable(self, reason: str, exc: BaseException) -> None:
        self._disabled = True
        if self._error_logged:
            return
        self._error_logged = True
        log.warning(
            "[Recorder] disabled",
            reason=str(reason or ""),
            path=str(self.path),
            exc_info=exc,
        )

    def record(self, payload: dict[str, Any]) -> None:
        if self._disabled:
            return
        try:
            obj = {
                "seq": int(self._seq),
                "ts": float(time.time()),
                "mono": float(time.monotonic()),
                "direction": "server->cli",
                "payload": payload,
            }
            self._seq += 1
            line = json.dumps(obj, ensure_ascii=False)
            with self._lock:
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
        except Exception as e:
            self._disable("write_frame_failed", e)
            return


def load_recorded_frames(path: Path) -> list[RecordedFrame]:
    p = Path(path)
    if not p.exists():
        return []
    frames: list[RecordedFrame] = []
    try:
        for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
            s = (line or "").strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            payload = obj.get("payload")
            if not isinstance(payload, dict):
                continue
            try:
                frames.append(
                    RecordedFrame(
                        seq=int(obj.get("seq") or 0),
                        ts=float(obj.get("ts") or 0.0),
                        mono=float(obj.get("mono") or 0.0),
                        payload=payload,
                    )
                )
            except (TypeError, ValueError):
                continue
    except OSError as e:
        log.warning("[Recorder] load_recorded_frames failed", path=str(p), exc_info=e)
        return []
    frames.sort(key=lambda x: x.seq)
    return frames


@dataclass(frozen=True)
class ReplaySpeed:
    speed: float = 1.0  # 0=尽快
    llm_speed: Optional[float] = None
    verify_speed: Optional[float] = None
    max_sleep: float = 2.0
    min_sleep: float = 0.0


def choose_speed_multiplier(message: object, cfg: ReplaySpeed) -> float:
    """按消息类型选择倍速（返回 multiplier，越大越快）。"""
    try:
        if cfg.speed <= 0:
            return float("inf")
    except (TypeError, ValueError):
        return float("inf")

    base = float(cfg.speed)
    name = type(message).__name__

    if cfg.llm_speed is not None:
        if name in ["LLMStreamMessage"]:
            try:
                v = float(cfg.llm_speed)
                return float("inf") if v <= 0 else v
            except (TypeError, ValueError):
                return base

    if cfg.verify_speed is not None:
        if name in ["NodeStartMessage", "NodeEndMessage"]:
            try:
                node = str(getattr(message, "node", "") or "")
            except Exception as e:
                log.debug("[Recorder] read node failed; fallback empty", exc_info=e)
                node = ""
            if node == "verifier":
                try:
                    v = float(cfg.verify_speed)
                    return float("inf") if v <= 0 else v
                except (TypeError, ValueError):
                    return base

    return base
