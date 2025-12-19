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

from typing import TYPE_CHECKING, Any

from .metadata import MetadataInjector
from .state import EvolveTaskStore, TaskStateRepository
from .stream_session import TaskStreamSession
from .task_id import TaskIdPolicy
from .trace import TraceController
from .ui import UiIntentController
from .watch import WatchController

if TYPE_CHECKING:
    from ..core import CLIPresenter


class EvolveTaskManager:
    """并发 task 管理：task_id 归一化、watch 切换、事件回放、Trace/Progress 渲染。"""

    def __init__(self, presenter: "CLIPresenter") -> None:
        self._p = presenter
        self._store = EvolveTaskStore()

        self._repo = TaskStateRepository(self._store)
        self._task_id = TaskIdPolicy(
            presenter, get_watch_task_id=lambda: self._store.watch_task_id
        )
        self._trace = TraceController(
            presenter,
            self._store,
            sorted_task_ids=self._repo.sorted_task_ids,
            get_watch_task_id=lambda: self._store.watch_task_id,
        )
        self._watch = WatchController(
            presenter,
            self._store,
            ensure_task_known=self.ensure_task_known,
            sorted_task_ids=self._repo.sorted_task_ids,
            apply_task_node_state=self.apply_task_node_state,
            refresh_task_tabs=self._trace.refresh_task_tabs,
            refresh_trace_panel=self._trace.refresh_trace_panel,
            refresh_progress_panel=self.refresh_progress_panel,
            task_events=self._repo.task_events,
        )
        self._ui = UiIntentController(
            presenter,
            self._store,
            watch_next=self._watch.watch_next,
            set_watch_task=self._watch.set_watch_task,
            refresh_progress_panel=self.refresh_progress_panel,
            refresh_trace_panel=self._trace.refresh_trace_panel,
        )
        self._meta = MetadataInjector(self._store)

    # ======= 对外读取（用于 summary / presenter）=======

    @property
    def replaying(self) -> bool:
        return bool(self._store.replaying)

    @property
    def evolve_round_snapshots(self) -> dict[int, dict[str, Any]]:
        return self._store.evolve_round_snapshots

    @property
    def evolve_task_summary(self) -> dict[str, dict[str, Any]]:
        return self._store.evolve_task_summary

    @property
    def latest_progress_text(self) -> str:
        return self._store.latest_progress_text

    @latest_progress_text.setter
    def latest_progress_text(self, value: str) -> None:
        self._store.latest_progress_text = str(value or "")

    @property
    def latest_progress_data(self) -> dict[str, Any]:
        return self._store.latest_progress_data

    @latest_progress_data.setter
    def latest_progress_data(self, value: dict[str, Any]) -> None:
        self._store.latest_progress_data = dict(value or {})

    @property
    def watch_task_id(self) -> str:
        return str(self._store.watch_task_id or "")

    @watch_task_id.setter
    def watch_task_id(self, value: str) -> None:
        self._store.watch_task_id = str(value or "").strip()

    # ======= UI intents / pump =======

    def start_ui_pump(self) -> None:
        self._ui.start_ui_pump()

    def stop_ui_pump(self) -> None:
        self._ui.stop_ui_pump()

    def poll_ui_events(self) -> None:
        self._ui.poll_ui_events()

    # ======= task_id / render policy =======

    def should_render_task(self, task_id: str) -> bool:
        return self._task_id.should_render_task(task_id)

    def normalize_task_id(self, raw_task_id: str, *, node_hint: str = "") -> str:
        return self._task_id.normalize_task_id(raw_task_id, node_hint=node_hint)

    def task_id_from_message(self, message: Any, *, node_hint: str = "") -> str:
        return self._task_id.task_id_from_message(message, node_hint=node_hint)

    # ======= task state / events =======

    def ensure_task_node_state(self, task_id: str) -> dict[str, Any]:
        return self._repo.ensure_task_node_state(task_id)

    def apply_task_node_state(self, task_id: str) -> None:
        self._repo.apply_task_node_state(self._p, task_id)

    def ensure_task_known(self, task_id: str) -> None:
        before = len(self._store.known_task_ids)
        self._repo.ensure_task_known(task_id)
        if len(self._store.known_task_ids) != before:
            self._trace.refresh_task_tabs()

    def record_task_event(self, task_id: str, payload: dict[str, Any]) -> int | None:
        return self._repo.record_task_event(task_id, payload)

    def task_events(self, task_id: str) -> list[dict[str, Any]]:
        return self._repo.task_events(task_id)

    def task_llm_state(self, task_id: str) -> dict[str, Any]:
        return self._repo.task_llm_state(task_id)

    def stream_session(self, task_id: str) -> TaskStreamSession:
        """获取某个 task 的流式渲染会话（每个 task 唯一）。"""
        tid = str(task_id or "").strip()
        if not tid:
            tid = "main"
        self.ensure_task_known(tid)
        sess = self._store.task_stream_sessions.get(tid)
        if isinstance(sess, TaskStreamSession):
            return sess
        sess2 = TaskStreamSession(
            self._p,
            tid,
            append_main_content=self._p._handlers._emit_main,
        )
        self._store.task_stream_sessions[tid] = sess2
        return sess2

    def append_task_main_content(self, task_id: str, content: Any) -> None:
        self._repo.append_task_main_content(task_id, content)

    def task_main_contents(self, task_id: str) -> list[Any]:
        return self._repo.task_main_contents(task_id)

    def sorted_task_ids(self) -> list[str]:
        return self._repo.sorted_task_ids()

    # ======= watch / replay =======

    def set_watch_task(
        self, task_id: str, *, force: bool = False, upto_event_idx: int | None = None
    ) -> None:
        self._watch.set_watch_task(task_id, force=force, upto_event_idx=upto_event_idx)

    def watch_next(self) -> None:
        self._watch.watch_next()

    def refresh_progress_panel(self) -> None:
        self._watch.refresh_progress_panel()

    def replay_task(self, task_id: str, *, upto_event_idx: int | None = None) -> None:
        self._watch.replay_task(task_id, upto_event_idx=upto_event_idx)

    # ======= trace / tabs =======

    def refresh_trace_panel(self) -> None:
        self._trace.refresh_trace_panel()

    def refresh_task_tabs(self) -> None:
        self._trace.refresh_task_tabs()

    def record_trace_node_start(
        self, *, task_id: str, node: str, run_no: int, event_idx: int
    ) -> None:
        self._trace.record_trace_node_start(
            task_id=task_id, node=node, run_no=run_no, event_idx=event_idx
        )

    # ======= metadata =======

    def inject_evolve_metadata(self, result: dict[str, Any]) -> None:
        self._meta.inject_evolve_metadata(result)
