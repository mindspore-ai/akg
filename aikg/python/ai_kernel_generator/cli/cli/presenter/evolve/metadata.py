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
    """把并发任务信息注入到 summary result 里（便于上层保存/展示）。"""

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

            # 规范化字段（兼容 summary 契约）
            for tid, info in tasks.items():
                if not isinstance(info, dict):
                    tasks[tid] = {"task_id": str(tid)}
                    continue
                info.setdefault("task_id", str(tid))
                if "verification_result" not in info and "verifier_result" in info:
                    info["verification_result"] = info.get("verifier_result")
                if "error_log" not in info:
                    ve = info.get("verifier_error")
                    if isinstance(ve, str) and ve.strip():
                        info["error_log"] = ve
            # 识别子 agent 名称（可来自 result 或 metadata）
            def _pick_group_name() -> str:
                for key in (
                    "subagent",
                    "sub_agent",
                    "sub_workflow",
                    "workflow_name",
                ):
                    val = result.get(key)
                    if isinstance(val, str) and val.strip():
                        return val.strip()
                for key in ("subagent", "sub_agent", "sub_workflow", "workflow_name"):
                    val = meta.get(key)
                    if isinstance(val, str) and val.strip():
                        return val.strip()
                # 有 evolve 轮次信息时可判定为 evolve
                if self._s.evolve_round_snapshots:
                    return "evolve"
                return "subagent"

            group_name = _pick_group_name()
            group_payload = {
                "name": group_name,
                "label": group_name,
                "watch_task_id": self._s.watch_task_id or "",
                "round_snapshots": dict(self._s.evolve_round_snapshots),
                "tasks": tasks,
            }

            # 新契约：统一并发任务摘要
            subagents = meta.get("subagents")
            if isinstance(subagents, dict):
                subagents[group_name] = group_payload
            elif isinstance(subagents, list):
                subagents.append(group_payload)
            else:
                meta["subagents"] = {group_name: group_payload}

            # 兼容旧字段
            meta["evolve"] = {
                "watch_task_id": self._s.watch_task_id or "",
                "round_snapshots": dict(self._s.evolve_round_snapshots),
                "tasks": tasks,
            }
        except Exception as e:
            log.warning("[Evolve] inject metadata failed; ignore", exc_info=e)
