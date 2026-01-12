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

from rich.text import Text

from ai_kernel_generator.cli.cli.constants import DisplayStyle
from ...utils.i18n import t

from textual import log


class WatchController:
    """watch 选择/回放逻辑：切换 task、清屏并回放历史事件、刷新右侧面板与 Trace。"""

    def __init__(
        self,
        presenter,
        store,
        *,
        ensure_task_known,
        sorted_task_ids,
        apply_task_node_state,
        refresh_task_tabs,
        refresh_trace_panel,
        refresh_progress_panel,
        task_events,
    ) -> None:
        self._p = presenter
        self._s = store
        self._ensure_task_known = ensure_task_known
        self._sorted_task_ids = sorted_task_ids
        self._apply_task_node_state = apply_task_node_state
        self._refresh_task_tabs = refresh_task_tabs
        self._refresh_trace_panel = refresh_trace_panel
        self._refresh_progress_panel = refresh_progress_panel
        self._task_events = task_events

    def set_watch_task(
        self, task_id: str, *, force: bool = False, upto_event_idx: int | None = None
    ) -> None:
        p = self._p
        tid = (task_id or "").strip()
        if not tid:
            return
        self._ensure_task_known(tid)
        old = str(self._s.watch_task_id or "")
        if (not force) and self._s.watch_task_id == tid:
            log.debug(
                "[Watch] set_watch_task no-op",
                task_id=tid,
                force=bool(force),
            )
            return
        self._s.watch_task_id = tid
        try:
            log.info(
                "[Watch] set_watch_task",
                old=old,
                new=tid,
                force=bool(force),
                upto_event_idx=upto_event_idx if upto_event_idx is not None else "",
            )
        except Exception as e:
            log.debug("[Watch] log set_watch_task failed; ignore", exc_info=e)
        self._refresh_task_tabs()
        try:
            if hasattr(p.layout_manager, "set_active_task_tab"):
                p.layout_manager.set_active_task_tab(tid)
        except Exception as e:
            log.warning("[Watch] set_active_task_tab failed", task_id=tid, exc_info=e)

        # 切换：清空 Chat，回放该 task 历史输出，然后继续追底
        try:
            if hasattr(p.layout_manager, "clear_log"):
                p.layout_manager.clear_log()
        except Exception as e:
            log.warning("[Watch] clear_log failed", exc_info=e)

        # 切换 Tab 时不自动改变焦点，便于连续左右切换

        # 注意：流式渲染的 renderer/buffer 状态由 TaskStreamSession 持有，
        # watch 切换不再直接操作 StreamRenderer，避免重复 start 导致重编号/分割线问题。
        p.llm_buffer = ""
        p.llm_running = False
        p.current_agent = ""
        p.current_model = ""

        try:
            self.replay_task(tid, upto_event_idx=upto_event_idx)
        except Exception as e:
            import traceback

            log.error(
                "[Watch] replay_task failed",
                task_id=tid,
                error=f"{type(e).__name__}: {e}",
                exc_info=e,
            )
            log.error(traceback.format_exc())
        # 切换后：同步该 task 的 current node / seen nodes，再刷新右侧
        self._apply_task_node_state(tid)
        # 切换后：把该 task 已累计的 llm_stream buffer 一次性补渲染到当前 UI（避免后台逐 chunk 渲染）
        try:
            if hasattr(p, "_handlers") and hasattr(
                p._handlers, "render_task_stream_buffer"
            ):
                p._handlers.render_task_stream_buffer(tid)
        except Exception as e:
            log.warning(
                "[Watch] render_task_stream_buffer failed", task_id=tid, exc_info=e
            )
        self._refresh_progress_panel()
        self._refresh_trace_panel()

    def watch_next(self) -> None:
        p = self._p
        ids = self._sorted_task_ids()
        if not ids or len(ids) <= 1:
            # 还没进入 evolve 并发（或只有 1 个 task）：给出提示
            log.debug(
                "[Watch] watch_next unavailable",
                known=len(ids or []),
                current=str(self._s.watch_task_id or ""),
            )
            return
        if not self._s.watch_task_id:
            self.set_watch_task(ids[-1])
            return
        try:
            i = ids.index(self._s.watch_task_id)
        except ValueError:
            i = -1
        self.set_watch_task(ids[(i + 1) % len(ids)])

    def refresh_progress_panel(self) -> None:
        """兼容旧逻辑：progress 现在渲染到 Workflow 面板。"""
        p = self._p
        p._refresh_info_panel()
        p._refresh_workflow_panel()

    def replay_task(self, task_id: str, *, upto_event_idx: int | None = None) -> None:
        """回放指定 task 的历史输出（优先使用当时真实渲染的 main_content 序列）。"""
        p = self._p
        tid = (task_id or "").strip()
        if not tid:
            return
        contents = []
        try:
            contents = p.tasks.task_main_contents(tid) or []
        except Exception as e:
            log.warning(
                "[Watch] read task_main_contents failed; fallback empty",
                task_id=tid,
                exc_info=e,
            )
            contents = []

        if upto_event_idx is not None:
            try:
                i = int(upto_event_idx)
                if i >= 0:
                    contents = contents[: min(len(contents), i + 1)]
            except (TypeError, ValueError) as e:
                log.debug("[Watch] upto_event_idx cast failed; ignore", exc_info=e)

        log.info(
            "[Watch] replay_task_from_cache",
            task_id=tid,
            contents=len(contents),
            upto_event_idx=upto_event_idx if upto_event_idx is not None else "",
        )

        self._s.replaying = True
        try:
            for c in contents:
                p._handlers._render_main_direct(c)
        finally:
            self._s.replaying = False
