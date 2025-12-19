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

from typing import Any

from textual import log


class MetadataInjector:
    """把 evolve 的并发信息注入到 summary result 里（便于上层保存/展示）。"""

    def __init__(self, store) -> None:
        self._s = store

    def inject_evolve_metadata(self, result: dict[str, Any]) -> None:
        try:
            meta = result.get("metadata")
            if not isinstance(meta, dict):
                meta = {}
                result["metadata"] = meta

            if not (self._s.evolve_round_snapshots or self._s.evolve_task_summary):
                return

            # 汇总每个 task 的最终状态：优先使用 verifier node_end 的信息
            tasks: dict[str, dict[str, Any]] = {}
            # 先从 progress 快照里拿状态（done/fail/running/queued）
            for rd, snap in sorted(self._s.evolve_round_snapshots.items()):
                tmap = snap.get("tasks") if isinstance(snap.get("tasks"), dict) else {}
                for tid, st in tmap.items():
                    tasks.setdefault(str(tid), {})["status"] = str(st)
                    tasks[str(tid)]["round"] = int(rd)
            # 再合并 verifier 错误摘要
            for tid, info in (self._s.evolve_task_summary or {}).items():
                tasks.setdefault(str(tid), {}).update(info)

            meta["evolve"] = {
                "watch_task_id": self._s.watch_task_id or "",
                "round_snapshots": dict(self._s.evolve_round_snapshots),
                "tasks": tasks,
            }
        except Exception as e:
            log.warning("[Evolve] inject metadata failed; ignore", exc_info=e)
